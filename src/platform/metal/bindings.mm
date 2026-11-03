// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include <armigo/common/diagnostics.h>
// Live descriptor facts gate every binding. Same-dimension arrayness differences
// use layer-zero views in the shader's target. A colour-coded sampled descriptor
// may resolve to an existing depth texture only for an admitted storage alias.
// Storage buffers bind their explicit descriptor extent and retain GPU writes in
// the coherence registry. Constant-buffer windows copy bounded physical
// components and zero-fill missing ranges.

#include "bindings.h"
#include <armigo/memory/write_tracker.h>
#include "texture_format.h"
#include "texture_fact.h"
#include "texture_residency.h"
#include "../../graphics/texture_residency.h"
#include "device_context.h"
#include "native_resources.h"
#include <unistd.h>
#include <string_view>

#include <armigo/graphics/texture_format.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>

namespace armigo::platform::metal {
namespace {

// The decoder's message plus the field, descriptor word and value it rejected,
// so a reached-profile stop names exactly what the guest programmed.
std::string texture_error_detail(const graphics::TextureError& error) {
    std::string detail = error.message();
    if (error.field != graphics::TextureField::None) {
        char extra[96];
        std::snprintf(extra, sizeof extra, " (field %u word %u value %#llx)",
                      unsigned(error.field), error.word,
                      static_cast<unsigned long long>(error.value));
        detail += extra;
    }
    return detail;
}

using namespace graphics;
namespace gm = shader::graphics_msl;
constexpr std::uint64_t staging_limit = 256u << 20;

// Host-owned null resources never enter the guest coherence registry.
class NullTexture final : public TextureResource {
public:
    explicit NullTexture(id<MTLTexture> texture) : native_(texture) {}
    BackingRange coverage() const noexcept override { return {}; }
    id<MTLTexture> texture() const noexcept override { return native_; }
    id<MTLTexture> allocation() const noexcept override { return native_; }
    RendererStatus read_gpu(BackingRange, std::span<std::byte>) const override {
        return RendererError{{CommandStatus::MemoryFault, "null texture has no guest backing"}};
    }
private:
    id<MTLTexture> native_;
};

[[noreturn]] void reject(CommandStatus status, std::string detail,
                         std::uint64_t address = 0) {
    throw RendererError{CommandResult{status, detail, address}};
}

void require(bool condition, const char* detail,
             CommandStatus status = CommandStatus::Unsupported) {
    if (!condition) reject(status, detail);
}

template <class T>
T take(RendererResult<T> result) {
    if (auto* error = std::get_if<RendererError>(&result)) throw *error;
    return std::get<T>(std::move(result));
}

// A uniform window's bytes taken straight into a new upload slice: the copied
// prefix from guest memory, zeros past it.
// A CPU-backed window known by its retained range is copied without a read
// transaction.
ResidentBuffer snapshot_window(ResourceMemory& memory, BufferCache& buffers, std::uint64_t address,
                               std::uint64_t copied, std::uint64_t window_bytes,
                               const RetainedRange* range = nullptr) {
    struct Fill {
        ResourceMemory& memory;
        std::uint64_t address, copied;
        const RetainedRange* range;
    } fill{memory, address, copied, range};
    return take(buffers.upload_filled(window_bytes, 4, [](void* raw, std::span<std::byte> output) -> RendererStatus {
        auto& source = *static_cast<Fill*>(raw);
        std::fill(output.begin() + std::ptrdiff_t(source.copied), output.end(), std::byte{});
        auto* registry = source.range && source.range->bytes.size == source.copied
            ? dynamic_cast<ResourceMemoryRegistry*>(&source.memory) : nullptr;
        if (registry) {
            auto lean = registry->read_cpu_backed(source.range->bytes, output.first(source.copied));
            if (auto* failure = std::get_if<RendererError>(&lean)) return *failure;
            if (std::get<bool>(lean)) return renderer_success();
        }
        return snapshot_buffer_into(source.memory, source.address, output.first(source.copied),
                                    BufferSnapshotPolicy::ZeroMissingConstantBufferBytes);
    }, &fill));
}

void append_dependencies(std::vector<CompletionToken>& destination,
                         const std::vector<CompletionToken>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

std::size_t index_size(IndexFormat format) {
    switch (format) {
    case IndexFormat::U8: return 1;
    case IndexFormat::U16: return 2;
    case IndexFormat::U32: return 4;
    }
    reject(CommandStatus::Unsupported, "binding plan has an unknown index format");
}

std::uint32_t load_index(const std::byte* bytes, IndexFormat format) {
    const auto byte = [&](std::size_t offset) {
        return std::to_integer<std::uint32_t>(bytes[offset]);
    };
    switch (format) {
    case IndexFormat::U8: return byte(0);
    case IndexFormat::U16: return byte(0) | (byte(1) << 8);
    case IndexFormat::U32:
        return byte(0) | (byte(1) << 8) | (byte(2) << 16) | (byte(3) << 24);
    }
    reject(CommandStatus::Unsupported, "binding plan has an unknown index format");
}

void read_checked(const GpuAddressSpace& address_space, std::uint64_t address,
                  std::span<std::byte> destination) {
    require(destination.size() <= std::numeric_limits<std::uint64_t>::max() - address,
            "binding guest-memory range overflows", CommandStatus::MemoryFault);
    while (!destination.empty()) {
        std::size_t size = 0;
        for (const auto& mapping : address_space.mappings()) {
            if (address >= mapping.gpu_base &&
                address - mapping.gpu_base < mapping.mapped_size) {
                size = std::min<std::uint64_t>(
                    destination.size(),
                    mapping.mapped_size - (address - mapping.gpu_base));
                break;
            }
        }
        require(size != 0, "binding guest-memory range is unmapped",
                CommandStatus::MemoryFault);
        const auto result = address_space.read(address, destination.first(size));
        require(result.ok(), "binding guest-memory read failed",
                CommandStatus::MemoryFault);
        address += size;
        destination = destination.subspan(size);
    }
}

std::vector<std::byte> snapshot(ResourceMemory& memory, std::uint64_t address,
                                std::uint64_t size) {
    require(size != 0 && size <= std::numeric_limits<std::size_t>::max(),
            "binding snapshot size is invalid", CommandStatus::BudgetExceeded);
    auto retained = take(memory.retain(address, size, ResourceAccess::Read));
    auto transaction = take(memory.prepare_cpu_read(
        retained.bytes, ResolveReason::CpuEngineRead));
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    read_checked(memory.address_space(), address, bytes);
    return bytes;
}

void store_u32_le(std::byte* destination, std::uint32_t value) {
    destination[0] = std::byte(value & 0xffu);
    destination[1] = std::byte((value >> 8) & 0xffu);
    destination[2] = std::byte((value >> 16) & 0xffu);
    destination[3] = std::byte((value >> 24) & 0xffu);
}

bool float_sampled(TextureFormat format) {
    switch (format) {
    case TextureFormat::R8Uint: case TextureFormat::R8Sint:
    case TextureFormat::R16Uint: case TextureFormat::R16Sint:
    case TextureFormat::R32Uint: case TextureFormat::R32Sint:
    case TextureFormat::R8G8Uint: case TextureFormat::R8G8Sint:
    case TextureFormat::R16G16Uint: case TextureFormat::R16G16Sint:
    case TextureFormat::R32G32Uint: case TextureFormat::R32G32Sint:
    case TextureFormat::R32G32B32Uint: case TextureFormat::R32G32B32Sint:
    case TextureFormat::R10G10B10A2Uint:
    case TextureFormat::R8G8B8A8Uint: case TextureFormat::R8G8B8A8Sint:
    case TextureFormat::R16G16B16A16Uint: case TextureFormat::R16G16B16A16Sint:
    case TextureFormat::R32G32B32A32Uint: case TextureFormat::R32G32B32A32Sint:
    case TextureFormat::D16Unorm: case TextureFormat::D32Float:
    case TextureFormat::D24UnormS8Uint: case TextureFormat::X8UintD24Unorm:
    case TextureFormat::S8UintD24Unorm: case TextureFormat::D32FloatS8Uint:
        return false;
    default:
        return true;
    }
}

const DerivedTextureFact* texture_fact(const CompiledGraphicsStage& stage,
                                       std::uint32_t offset) {
    const auto found = std::find_if(
        stage.texture_facts.begin(), stage.texture_facts.end(),
        [&](const DerivedTextureFact& fact) {
            return fact.handle_byte_offset == offset;
        });
    return found == stage.texture_facts.end() ? nullptr : &*found;
}

std::string texture_handle(const DerivedTextureFact& fact) {
    return "cbuf " + std::to_string(unsigned(fact.cbuf_slot)) + " offset " +
           std::to_string(fact.handle_byte_offset);
}

TextureTargetPlan verify_texture_fact(const DerivedTextureFact& fact,
                                      const ResolvedTextureBinding& resolved, bool uses_sampler) {
    const auto target = texture_target_plan(fact, resolved);
    const auto mismatch = texture_fact_mismatch(fact, resolved, uses_sampler);
    if (!mismatch.empty())
        reject(CommandStatus::Unsupported,
               "texture_descriptor fact mismatch: " + mismatch +
               " for handle " + texture_handle(fact));
    return std::get<TextureTargetPlan>(target);
}

// A compare-sampled colour-coded TIC whose format aliases depth storage
// (R16Unorm/D16Unorm,
// R32Float/D32Float) is a depth texture's sampling view — NVN describes the
// depth target the game rendered through the colour format. The binding's
// effective descriptor adopts the depth format so fact admission, residency
// keys and the host allocation all use the depth representation; non-compare
// samplers keep the colour-coded key and resolve onto depth storage through
// the texture cache's sampler alias instead. `aliased` must outlive the
// returned reference.
const graphics::ResolvedTextureBinding& depth_alias_binding(
    const graphics::ResolvedTextureBinding& binding,
    const DerivedTextureFact* fact,
    graphics::ResolvedTextureBinding& aliased) {
    if (fact && fact->comparison && !binding.is_null) {
        if (const auto depth = graphics::depth_alias_storage(binding.texture.format)) {
            aliased = binding;
            aliased.texture.format = *depth;
            return aliased;
        }
    }
    return binding;
}

DerivedTextureFact resource_texture_fact(const CompiledShaderResource& resource,
                                         std::uint32_t offset) {
    require(resource.texture_descriptor.has_value(),
            "sampled resource is missing its texture descriptor");
    const auto& descriptor = *resource.texture_descriptor;
    DerivedTextureFact fact;
    fact.cbuf_slot = resource.cbuf_slot;
    fact.handle_byte_offset = offset;
    fact.normalized_coordinates = descriptor.normalized_coordinates;
    fact.comparison = descriptor.comparison;
    fact.array = descriptor.array;
    fact.multisample = descriptor.multisample;
    fact.component_format = descriptor.component_format;
    switch (descriptor.dimension) {
    case CompiledShaderTextureDimension::d1: fact.dimension = TextureFactDimension::d1; break;
    case CompiledShaderTextureDimension::d2: fact.dimension = TextureFactDimension::d2; break;
    case CompiledShaderTextureDimension::d3: fact.dimension = TextureFactDimension::d3; break;
    case CompiledShaderTextureDimension::cube: fact.dimension = TextureFactDimension::cube; break;
    default: reject(CommandStatus::Unsupported, "buffer textures have no runtime binding path");
    }
    switch (descriptor.scalar) {
    case CompiledShaderTextureScalar::f32: fact.scalar = TextureSampledScalar::f32; break;
    case CompiledShaderTextureScalar::u32: fact.scalar = TextureSampledScalar::u32; break;
    case CompiledShaderTextureScalar::s32: fact.scalar = TextureSampledScalar::s32; break;
    default: reject(CommandStatus::Unsupported, "texture sampled scalar has no runtime binding path");
    }
    require(resource.comparison == descriptor.comparison &&
                resource.normalized_coordinates == descriptor.normalized_coordinates,
            "resource and texture descriptor contracts disagree");
    return fact;
}

void validate_stage_resources(const CompiledGraphicsStage& stage,
                              shader_input::Stage expected,
                              std::uint32_t texture_selector,
                              const std::set<std::uint32_t>& vertex_indices) {
    require(stage.stage == expected, "compiled graphics stage kind is invalid");
    std::set<unsigned> guest, buffers, texture_indices, sampler_indices, handles;
    for (const auto& binding : stage.constant_buffers) {
        require(binding.guest_binding < 18 && binding.metal_buffer_index < 18 &&
                    binding.minimum_size >= 4 && binding.minimum_size <= 65536 &&
                    !(binding.minimum_size & 3u) &&
                    guest.insert(binding.guest_binding).second &&
                    buffers.insert(binding.metal_buffer_index).second,
                "constant-buffer reflection is invalid");
    }
    require(stage.constant_buffer_windows.size() <= 18 &&
                stage.constant_buffer_support_index.has_value() ==
                    !stage.constant_buffer_windows.empty() &&
                (stage.constant_buffer_windows.empty() || stage.constant_buffers.empty()),
            "runtime constant-buffer policy is invalid or mixed with legacy bindings");
    for (const auto& binding : stage.constant_buffer_windows) {
        require(binding.guest_binding < 18 && binding.metal_buffer_index < 31 &&
                    binding.window_bytes >= 4 && binding.window_bytes <= 65536 &&
                    !(binding.window_bytes & 3u) &&
                    guest.insert(binding.guest_binding).second &&
                    buffers.insert(binding.metal_buffer_index).second,
                "constant-buffer window reflection is invalid");
    }
    if (stage.constant_buffer_support_index) {
        require(*stage.constant_buffer_support_index < 31 &&
                    buffers.insert(*stage.constant_buffer_support_index).second,
                "constant-buffer support index is invalid or duplicated");
    }
    std::set<std::pair<unsigned, unsigned>> storage_descriptors;
    require(stage.storage_buffers.size() <= 31,
            "storage-buffer reflection exceeds the Metal buffer namespace");
    for (const auto& binding : stage.storage_buffers) {
        const auto window = std::ranges::find_if(stage.constant_buffer_windows,
            [&](const auto& candidate) { return candidate.guest_binding == binding.cbuf_slot; });
        require(binding.cbuf_slot < 18 && binding.descriptor_byte_offset <= 65524 &&
                    !(binding.descriptor_byte_offset & 7u) && binding.alignment == 4 &&
                    binding.access <= CompiledShaderResourceAccess::ReadWrite &&
                    binding.metal_buffer_index < 31 &&
                    buffers.insert(binding.metal_buffer_index).second &&
                    storage_descriptors.emplace(binding.cbuf_slot, binding.descriptor_byte_offset).second &&
                    window != stage.constant_buffer_windows.end() &&
                    binding.descriptor_byte_offset + 12 <= window->window_bytes,
                "storage-buffer reflection is invalid or aliases another buffer");
    }
    if (expected == shader_input::Stage::Vertex) {
        for (auto index : vertex_indices)
            require(index < 31 && buffers.insert(index).second,
                    "vertex stream aliases or exceeds a shader buffer binding");
    }
    if (stage.emulates_depth_bias)
        require(expected == shader_input::Stage::Fragment && !stage.writes_fragment_depth &&
                    buffers.insert(30).second,
                "shader depth bias ABI requires an exclusive fragment buffer 30");
    if (stage.custom_border)
        require(stage.custom_border->buffer_index < 31 &&
                    buffers.insert(stage.custom_border->buffer_index).second &&
                    valid_custom_border_pairs(stage),
                "custom border support reflection is invalid or aliases a buffer");
    if (stage.texture_handle_cbuf)
        require(texture_selector == *stage.texture_handle_cbuf,
                "texture_handle_cbuf fact mismatch: the live selector differs from "
                "the compiled handle constant buffer");
    if (stage.argument_buffer) {
        const auto& layout = *stage.argument_buffer;
        require(layout.abi_version == 1 && layout.buffer_index < 31 &&
                    layout.byte_extent && !(layout.byte_extent & 7u) &&
                    layout.alignment >= 8 && !(layout.alignment & (layout.alignment - 1)) &&
                    buffers.insert(layout.buffer_index).second && !stage.resources.empty(),
                "argument-buffer layout is invalid or aliases another buffer");
        require(std::uint64_t(layout.byte_extent) + layout.alignment - 1 <= staging_limit,
                "argument-buffer layout exceeds the staging budget", CommandStatus::BudgetExceeded);
    }
    if (!stage.resources.empty()) {
        std::set<std::uint32_t> ids;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> members;
        for (const auto& resource : stage.resources) {
            require(ids.insert(resource.resource_id).second && resource.array_length &&
                        resource.cbuf_slot < 18 && !(resource.handle_byte_offset & 3u) &&
                        std::uint64_t(resource.handle_byte_offset) +
                            std::uint64_t(resource.array_length) * 4 <= 65536 &&
                        resource.location.count == resource.array_length,
                    "resource reflection provenance or array length is invalid");
            require(resource.access == CompiledShaderResourceAccess::Read &&
                        (resource.kind == CompiledShaderResourceKind::SampledTexture ||
                         resource.kind == CompiledShaderResourceKind::Sampler),
                    "storage image resources have no runtime binding path");
            const bool sampler = resource.kind == CompiledShaderResourceKind::Sampler;
            if (!sampler) (void)resource_texture_fact(resource, resource.handle_byte_offset);
            else require(!resource.texture_descriptor, "sampler resource carries a texture descriptor");
            const auto& location = resource.location;
            if (stage.argument_buffer) {
                const auto end = std::uint64_t(location.byte_offset) +
                                 std::uint64_t(location.count) * 8;
                require(location.kind == CompiledShaderResourceLocationKind::ArgumentBufferMember &&
                            location.stride == 8 && !(location.byte_offset & 7u) &&
                            end <= stage.argument_buffer->byte_extent,
                        "argument-buffer member is outside its declared layout");
                members.emplace_back(location.byte_offset, end);
            } else {
                const auto end = std::uint64_t(location.first) + location.count;
                require(location.kind == CompiledShaderResourceLocationKind::Direct &&
                            location.resource_namespace == (sampler
                                ? CompiledShaderResourceNamespace::Sampler
                                : CompiledShaderResourceNamespace::Texture) &&
                            end <= (sampler ? 16u : 128u),
                        "direct resource location exceeds its namespace");
                auto& indices = sampler ? sampler_indices : texture_indices;
                for (auto index = location.first; index != end; ++index)
                    require(indices.insert(index).second, "direct resource locations overlap");
            }
        }
        std::sort(members.begin(), members.end());
        std::uint64_t end = 0;
        for (const auto& member : members) {
            require(member.first >= end, "argument-buffer members overlap");
            end = member.second;
        }
        return;
    }
    require(stage.legacy_texture_bindings_compatible,
            "stage has no independent resource reflection for its bindings");
    for (const auto& binding : stage.textures) {
        require(binding.texture_index < 128 &&
                    (binding.sampler_index < 16 || binding.sampler_index == gm::TextureBinding::no_sampler) &&
                    !(binding.handle_byte_offset & 3u) &&
                    handles.insert(binding.handle_byte_offset).second &&
                    texture_indices.insert(binding.texture_index).second &&
                    (binding.sampler_index == gm::TextureBinding::no_sampler ||
                     sampler_indices.insert(binding.sampler_index).second),
                "texture reflection is invalid or duplicated");
    }
    std::set<std::uint32_t> facts;
    for (const auto& fact : stage.texture_facts) {
        require(fact.cbuf_slot < 18 && !(fact.handle_byte_offset & 3u) &&
                    facts.insert(fact.handle_byte_offset).second &&
                    handles.contains(fact.handle_byte_offset) &&
                    stage.texture_handle_cbuf &&
                    fact.cbuf_slot == *stage.texture_handle_cbuf,
                "texture descriptor fact is stale or inconsistent");
    }
    if (!stage.texture_facts.empty())
        require(stage.texture_facts.size() == stage.textures.size(),
                "texture descriptor facts do not cover every binding");
}

void validate_stage_interfaces(const AdmittedGraphicsStages& admitted,
                               const DrawPlan& draw) {
    require(admitted.stages.vertex && admitted.stages.fragment,
            "compiled graphics stage pair is missing");
    const auto& vertex = *admitted.stages.vertex;
    const auto& fragment = *admitted.stages.fragment;
    require(vertex.stage == shader_input::Stage::Vertex &&
                fragment.stage == shader_input::Stage::Fragment &&
                !vertex.metallib.empty() && !fragment.metallib.empty() &&
                !vertex.entry_point.empty() && !fragment.entry_point.empty() &&
                vertex.entry_point.find('\0') == std::string::npos &&
                fragment.entry_point.find('\0') == std::string::npos,
            "compiled stage library or entry point is invalid");
    require(vertex.vertex_depth == gm::VertexDepthConvention::NegativeOneToOne &&
                vertex.fragment_position == gm::FragmentPositionConvention::Unspecified &&
                fragment.vertex_depth == gm::VertexDepthConvention::Unspecified &&
                fragment.vertex_attributes.empty() &&
                (fragment.fragment_position ==
                     gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW ||
                 (fragment.fragment_position == gm::FragmentPositionConvention::Unspecified &&
                  !fragment.interface.position_input_mask)),
            "compiled coordinate variant does not match the renderer profile");
    require(vertex.interface.position_input_mask == 0 &&
                vertex.interface.position_output_mask == 15 &&
                vertex.interface.color_output_mask == 0 &&
                !vertex.writes_fragment_depth && !vertex.writes_sample_mask &&
                fragment.interface.position_output_mask == 0 &&
                !(fragment.interface.position_input_mask & ~15u) &&
                fragment.interface.generic_outputs.empty(),
            "compiled stage interface is unsupported");

    std::uint32_t color_summary = 0;
    std::set<std::uint32_t> color_locations;
    for (const auto& output : fragment.color_outputs) {
        require(output.location < 8 && output.component_mask &&
                    output.components >= 1 && output.components <= 4 &&
                    !(output.component_mask >> output.components) &&
                    output.scalar <= ColorOutputScalar::s32 &&
                    color_locations.insert(output.location).second,
                "fragment color output is invalid or duplicated");
        color_summary |= std::uint32_t(output.component_mask) << (4 * output.location);
    }
    require(color_summary == fragment.interface.color_output_mask,
            "fragment color outputs disagree with their interface summary");

    const bool class_inputs = !vertex.vertex_class_inputs.empty();
    require(!class_inputs || vertex.vertex_attributes.empty(),
            "vertex input reflection mixes legacy and class-typed inputs");
    require(vertex.interface.generic_inputs.size() ==
                (class_inputs ? vertex.vertex_class_inputs.size()
                              : vertex.vertex_attributes.size()),
            "vertex input reflection does not match its format records");
    require(draw.vertex.attributes.size() ==
                (class_inputs ? vertex.vertex_class_inputs.size()
                              : vertex.vertex_attributes.size()),
            "draw vertex plan does not cover every reflected input");

    std::set<unsigned> locations, attribute_indices;
    if (class_inputs) {
        for (const auto& input : vertex.vertex_class_inputs) {
            require(input.location < 31 && locations.insert(input.location).second,
                    "vertex class input is invalid or duplicated");
            const auto planned = std::find_if(
                draw.vertex.attributes.begin(), draw.vertex.attributes.end(),
                [&](const VertexAttributePlan& attribute) {
                    return attribute.location == input.location &&
                           attribute.format.format_class == input.input_class;
                });
            require(planned != draw.vertex.attributes.end(),
                    "draw vertex format class differs from compiled input");
        }
    } else {
        for (const auto& input : vertex.vertex_attributes) {
            require(input.generic_slot < 32 && input.attribute_index < 31 &&
                        input.format == gm::VertexFormat::Float32x3 &&
                        locations.insert(input.generic_slot).second &&
                        attribute_indices.insert(input.attribute_index).second,
                    "legacy vertex input is invalid or duplicated");
            const auto planned = std::find_if(
                draw.vertex.attributes.begin(), draw.vertex.attributes.end(),
                [&](const VertexAttributePlan& attribute) {
                    return attribute.location == input.generic_slot &&
                           attribute.format.format ==
                               VertexAttribFormat::R32G32B32Float;
                });
            require(planned != draw.vertex.attributes.end(),
                    "legacy vertex input format differs from compiled reflection");
        }
    }
    std::set<unsigned> generic_inputs;
    for (const auto& input : vertex.interface.generic_inputs) {
        require(input.slot < 32 && input.component_mask &&
                    !(input.component_mask & ~15u) &&
                    (class_inputs || !(input.component_mask & ~7u)) &&
                    generic_inputs.insert(input.slot).second &&
                    locations.contains(input.slot),
                "vertex generic input is missing a format record");
    }

    std::map<unsigned, std::uint8_t> outputs;
    for (const auto& output : vertex.interface.generic_outputs) {
        require(output.slot < 32 && output.component_mask &&
                    !(output.component_mask & ~15u) &&
                    outputs.emplace(output.slot, output.component_mask).second,
                "vertex generic output is invalid or duplicated");
    }
    std::set<unsigned> fragment_inputs;
    for (const auto& input : fragment.interface.generic_inputs) {
        require(input.slot < 32 && input.component_mask &&
                    !(input.component_mask & ~15u) &&
                    fragment_inputs.insert(input.slot).second &&
                    (input.interpolation == shader::graphics::Interpolation::Constant ||
                     input.interpolation == shader::graphics::Interpolation::Perspective ||
                     input.interpolation == shader::graphics::Interpolation::ScreenLinear),
                "fragment generic input is invalid or duplicated");
        require(input.interpolation != shader::graphics::Interpolation::Perspective ||
                    fragment.fragment_position ==
                        gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW,
                "perspective input requires the admitted position convention");
        const auto output = outputs.find(input.slot);
        require(output != outputs.end() &&
                    (output->second & input.component_mask) == input.component_mask,
                "vertex and fragment generic interfaces do not link");
    }
}

struct StreamGroup {
    std::uint32_t stream = 0;
    std::uint32_t reached = 0;
    bool constant_attribute = false;
    const VertexStreamPlan* plan = nullptr;
    const VertexStreamStep* step = nullptr;
};

struct DirectBufferRequest {
    std::uint64_t address = 0;
    RetainedRange retained;
    BufferUse use = BufferUse::Vertex;
    std::uint64_t alignment = 1;
    std::optional<std::uint32_t> vertex_index;
    std::optional<std::pair<shader_input::Stage, std::uint32_t>> shader_binding;
    ResourceAccess access = ResourceAccess::Read;
    std::uint64_t native_extent = 0;
};

bool needs_instance_expansion(const StreamGroup& group, const DrawPlan& draw) {
    if (!group.step || group.step->step != VertexStep::PerInstance ||
        group.step->rate <= 1) return false;
    return std::any_of(draw.segments.begin(), draw.segments.end(),
        [](const DrawSegment& segment) {
            return segment.count && segment.instance_count && segment.base_instance;
        });
}

ResidentBuffer expand_instance_stream(const StreamGroup& group,
                                      const DrawPlan& draw,
                                      ResourceMemory& memory,
                                      BufferCache& buffers) {
    std::uint64_t maximum_guest = 0, maximum_native = 0;
    bool any = false;
    for (const auto& segment : draw.segments) {
        if (!segment.count || !segment.instance_count) continue;
        const auto last_native = std::uint64_t(segment.base_instance) +
                                 segment.instance_count - 1;
        const auto last_guest = last_native / group.step->rate;
        maximum_native = any ? std::max(maximum_native, last_native) : last_native;
        maximum_guest = any ? std::max(maximum_guest, last_guest) : last_guest;
        any = true;
    }
    require(any, "instance expansion has no live draw segment");
    const auto host_stride = std::max(group.plan->stride, group.reached);
    require(host_stride && host_stride <= staging_limit && maximum_native <=
                (staging_limit / host_stride) - 1,
            "expanded instance stream exceeds the staging limit",
            CommandStatus::BudgetExceeded);
    require(!group.plan->stride || maximum_guest <=
                (std::numeric_limits<std::uint64_t>::max() - group.reached) /
                    group.plan->stride,
            "guest instance stream extent overflows", CommandStatus::MemoryFault);
    const auto source_size = maximum_guest * group.plan->stride + group.reached;
    require(source_size && source_size <= staging_limit &&
                source_size - 1 <= group.plan->inclusive_limit - group.plan->address,
            "guest instance stream exceeds its limit or staging budget",
            source_size > staging_limit ? CommandStatus::BudgetExceeded
                                        : CommandStatus::MemoryFault);
    const auto source = snapshot(memory, group.plan->address, source_size);
    std::vector<std::byte> expanded((maximum_native + 1) * host_stride);
    for (const auto& segment : draw.segments) {
        if (!segment.count || !segment.instance_count) continue;
        for (std::uint64_t local = 0; local != segment.instance_count; ++local) {
            const auto native = std::uint64_t(segment.base_instance) + local;
            const auto guest = native / group.step->rate;
            std::copy_n(source.data() + guest * group.plan->stride,
                        group.reached,
                        expanded.data() + native * host_stride);
        }
    }
    return take(buffers.upload(expanded, 4));
}

std::vector<ResidentBuffer> preflight_overlapping_buffers(
    ResourceMemory& memory, BufferCache& buffers,
    const std::vector<DirectBufferRequest>& requests) {
    std::vector<std::size_t> order(requests.size());
    for (std::size_t i = 0; i != order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        const auto& a = requests[left].retained.bytes;
        const auto& b = requests[right].retained.bytes;
        return std::tie(a.backing.owner, a.backing.allocation,
                        a.backing.generation, a.offset, a.size) <
               std::tie(b.backing.owner, b.backing.allocation,
                        b.backing.generation, b.offset, b.size);
    });
    std::vector<ResidentBuffer> held;
    for (std::size_t first = 0; first != order.size();) {
        const auto& initial = requests[order[first]];
        const auto backing = initial.retained.bytes.backing;
        const auto start = initial.retained.bytes.offset;
        auto end = start + initial.retained.bytes.size;
        auto alignment = initial.alignment;
        std::size_t last = first + 1;
        while (last != order.size()) {
            const auto& next = requests[order[last]];
            const auto& bytes = next.retained.bytes;
            if (bytes.backing != backing || bytes.offset >= end) break;
            end = std::max(end, bytes.offset + bytes.size);
            alignment = std::max(alignment, next.alignment);
            ++last;
        }
        if (last - first > 1) {
            for (std::size_t item = first; item != last; ++item) {
                const auto& request = requests[order[item]];
                require(!((request.retained.bytes.offset - start) % request.alignment),
                        "overlapping buffer bindings have incompatible alignments");
            }
            std::optional<RetainedRange> retained;
            std::optional<RendererError> failure;
            for (std::size_t item = first; item != last; ++item) {
                const auto& request = requests[order[item]];
                if (request.retained.bytes.offset != start) continue;
                auto candidate = memory.retain(
                    request.address, end - start, ResourceAccess::Read);
                if (auto* error = std::get_if<RendererError>(&candidate)) {
                    failure = *error;
                    continue;
                }
                retained = std::get<RetainedRange>(std::move(candidate));
                break;
            }
            if (!retained) throw *failure;
            held.push_back(take(buffers.acquire(
                {*retained, BufferUse::Transfer, ResourceAccess::Read, alignment})));
        }
        first = last;
    }
    return held;
}

// Constant attributes read zero (Armigo binds them to a zero source) from one
// shared zero stream: an all-constant stream of their own when there is one,
// otherwise the lowest native stream slot that no fetched attribute uses.
std::optional<std::uint32_t> constant_stream(const DrawPlan& draw) {
    std::set<std::uint32_t> fetched, constant;
    for (const auto& attribute : draw.vertex.attributes)
        (attribute.constant ? constant : fetched).insert(attribute.stream);
    if (constant.empty()) return std::nullopt;
    for (const auto stream : constant)
        if (!fetched.contains(stream)) return stream;
    for (std::uint32_t stream = 0; stream <= 12; ++stream)
        if (!fetched.contains(stream)) return stream;
    return std::nullopt;
}

std::uint32_t attribute_stream(const VertexAttributePlan& attribute,
                               const std::optional<std::uint32_t>& constant) {
    return attribute.constant && constant ? *constant : attribute.stream;
}

std::vector<StreamGroup> stream_groups(const DrawPlan& draw) {
    std::map<std::uint32_t, StreamGroup> groups;
    std::set<std::uint32_t> locations;
    const auto constant = constant_stream(draw);
    for (const auto& attribute : draw.vertex.attributes) {
        require(!attribute.constant || constant,
                "constant vertex attributes have no free native stream slot");
        require(attribute.location < 31 && locations.insert(attribute.location).second &&
                    attribute.format.components >= 1 &&
                    attribute.format.components <= 4 &&
                    attribute.format.size_bytes != 0 &&
                    attribute.offset <= std::numeric_limits<std::uint32_t>::max() -
                                            attribute.format.size_bytes,
                "draw vertex attribute is invalid or duplicated");
        const auto stream = attribute_stream(attribute, constant);
        auto& group = groups[stream];
        if (group.reached == 0) {
            group.stream = stream;
            group.constant_attribute = attribute.constant;
        } else {
            require(group.constant_attribute == attribute.constant,
                    "vertex stream mixes constant and fetched attributes");
        }
        group.reached = std::max(
            group.reached,
            attribute.offset + std::uint32_t(attribute.format.size_bytes));
    }
    std::set<std::uint32_t> planned_streams, stepped_streams;
    for (const auto& stream : draw.vertex.streams)
        require(planned_streams.insert(stream.stream).second,
                "draw vertex stream is duplicated");
    for (const auto& step : draw.vertex.steps)
        require(stepped_streams.insert(step.stream).second,
                "draw vertex step is duplicated");
    std::size_t fetched_groups = 0;
    for (auto& [stream, group] : groups) {
        require(stream <= 12, "vertex stream exceeds the native binding range");
        if (group.constant_attribute) continue;
        ++fetched_groups;
        const auto planned = std::find_if(
            draw.vertex.streams.begin(), draw.vertex.streams.end(),
            [&](const VertexStreamPlan& item) { return item.stream == stream; });
        const auto step = std::find_if(
            draw.vertex.steps.begin(), draw.vertex.steps.end(),
            [&](const VertexStreamStep& item) { return item.stream == stream; });
        require(planned != draw.vertex.streams.end() &&
                    step != draw.vertex.steps.end() &&
                    planned->address && planned->address <= planned->inclusive_limit &&
                    step->rate != 0,
                "vertex stream or divisor plan is missing or invalid");
        group.plan = &*planned;
        group.step = &*step;
    }
    require(draw.vertex.streams.size() == draw.vertex.steps.size() &&
                draw.vertex.streams.size() == fetched_groups &&
                planned_streams == stepped_streams,
            "vertex stream and step plans disagree with consumed attributes");
    std::vector<StreamGroup> result;
    result.reserve(groups.size());
    for (const auto& [stream, group] : groups) result.push_back(group);
    return result;
}

std::optional<std::uint64_t> maximum_vertex(const DrawPlan& draw,
                                            std::span<const std::byte> indices) {
    if (!draw.indices) {
        std::optional<std::uint64_t> maximum;
        for (const auto& segment : draw.segments) {
            if (!segment.count) continue;
            const std::uint64_t last = std::uint64_t(segment.first) + segment.count - 1;
            if (!maximum || last > *maximum) maximum = last;
        }
        return maximum;
    }
    const auto element = index_size(draw.indices->encoded_format);
    require(!(indices.size() % element), "encoded index bytes are truncated");
    const auto count = indices.size() / element;
    std::optional<std::uint64_t> maximum;
    for (const auto& segment : draw.segments) {
        require(segment.first <= count && segment.count <= count - segment.first,
                "draw segment exceeds its encoded index buffer");
        for (std::size_t item = segment.first;
             item != std::size_t(segment.first) + segment.count; ++item) {
            const auto index = load_index(indices.data() + item * element,
                                          draw.indices->encoded_format);
            if (draw.indices->restart_enabled &&
                index == draw.indices->restart_index) continue;
            const auto resolved = std::int64_t(index) + segment.base_vertex;
            require(resolved >= 0,
                    "native base vertex makes an index negative",
                    CommandStatus::MemoryFault);
            const auto fetch = std::uint64_t(resolved);
            if (!maximum || fetch > *maximum) maximum = fetch;
        }
    }
    return maximum;
}

class Planner final : public BindingPlanner {
public:
    Planner(std::shared_ptr<PipelineStateCache> states, std::shared_ptr<DeviceContext> device)
        : states_(std::move(states)), device_(std::move(device)) {}

    RendererResult<Bindings> prepare(
        const RenderStop& stop, const AdmittedGraphicsStages& stages,
        const DrawPlan& draw, ResourceMemory& memory,
        BufferCache& buffers, TextureCache& textures, const BindingReuse& reuse) override {
        try {
            return prepare_checked(stop, stages, draw, memory, buffers, textures, reuse);
        } catch (const RendererError& error) {
            return error;
        } catch (const std::bad_alloc&) {
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "binding preparation exceeded the host allocation budget"}};
        } catch (const std::length_error&) {
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "binding preparation exceeds a supported container extent"}};
        }
    }

private:
    // Index and vertex stream requests of a draw (shared by full preparation
    // and draws repeating the last draw's state with new buffers).
    void vertex_index_requests(const DrawPlan& draw, const std::vector<StreamGroup>& groups,
                               ResourceMemory& memory, BufferCache& buffers, Bindings& result,
                               std::vector<DirectBufferRequest>& direct_buffers) {
        std::vector<std::byte> index_bytes;
        std::optional<DirectBufferRequest> direct_index;
        if (draw.indices) {
            const auto element = index_size(draw.indices->encoded_format);
            if (draw.indices->converted.empty()) {
                require(draw.indices->encoded_format == draw.indices->source_format &&
                            draw.indices->address &&
                            draw.indices->address <= draw.indices->inclusive_limit,
                        "direct index range is invalid");
                const auto size = draw.indices->inclusive_limit -
                                  draw.indices->address + 1;
                require(size <= std::numeric_limits<std::size_t>::max(),
                        "direct index range exceeds the host size budget",
                        CommandStatus::BudgetExceeded);
                require(!(size % element), "direct index range is not element aligned");
                // Guest indices bind in place and are never read on the CPU:
                // per-vertex streams of this draw bind their whole guest range.
                auto retained = take(memory.retain(draw.indices->address, size,
                                                   ResourceAccess::Read));
                direct_index = DirectBufferRequest{
                    draw.indices->address, retained, BufferUse::Index, element,
                    std::nullopt};
            } else {
                require(!(draw.indices->converted.size() % element),
                        "converted index bytes are not element aligned");
                index_bytes = draw.indices->converted;
                auto resident = take(buffers.upload(index_bytes, element));
                append_dependencies(result.dependencies, resident.dependencies);
                result.index_buffer = std::move(resident);
            }
        }
        const bool unbounded_fetch = draw.indices && draw.indices->converted.empty();
        const auto max_vertex = unbounded_fetch ? std::nullopt : maximum_vertex(draw, index_bytes);

        if (direct_index) direct_buffers.push_back(*direct_index);
        for (const auto& group : groups) {
            if (group.constant_attribute) {
                std::vector<std::byte> zeros(group.reached);
                auto resident = take(buffers.upload(zeros, 4));
                append_dependencies(result.dependencies, resident.dependencies);
                result.vertex_buffers.push_back(
                    {shader_input::Stage::Vertex, 18 + group.stream,
                     std::move(resident)});
                continue;
            }
            if (needs_instance_expansion(group, draw)) {
                auto resident = expand_instance_stream(
                    group, draw, memory, buffers);
                append_dependencies(result.dependencies, resident.dependencies);
                result.vertex_buffers.push_back(
                    {shader_input::Stage::Vertex, 18 + group.stream,
                     std::move(resident)});
                continue;
            }
            std::uint64_t last = 0;
            if (unbounded_fetch && group.step->step == VertexStep::PerVertex) {
                // Bind a vertex buffer from its address to the stream limit
                // within the mapping that holds the address.
                const auto mappings = memory.address_space().mappings();
                const auto after = std::upper_bound(mappings.begin(), mappings.end(), group.plan->address,
                    [](std::uint64_t value, const auto& mapping) { return value < mapping.gpu_base; });
                require(after != mappings.begin(), "vertex stream address is unmapped", CommandStatus::MemoryFault);
                const auto& mapping = *std::prev(after);
                require(group.plan->address - mapping.gpu_base < mapping.mapped_size,
                        "vertex stream address is unmapped", CommandStatus::MemoryFault);
                const auto backed = mapping.backing_size > mapping.buffer_offset
                    ? mapping.backing_size - mapping.buffer_offset : 0;
                const auto mapped_end = mapping.gpu_base + std::min<std::uint64_t>(mapping.mapped_size, backed) - 1;
                const auto size = std::min(group.plan->inclusive_limit, mapped_end) - group.plan->address + 1;
                require(size >= group.reached, "vertex stream is smaller than one vertex",
                        CommandStatus::MemoryFault);
                auto retained = take(memory.retain(group.plan->address, size, ResourceAccess::Read));
                direct_buffers.push_back({group.plan->address, retained, BufferUse::Vertex, 4,
                                          18 + group.stream});
                continue;
            }
            switch (group.step->step) {
            case VertexStep::PerVertex:
                last = max_vertex.value_or(0);
                break;
            case VertexStep::PerInstance: {
                std::uint64_t maximum = 0;
                bool any = false;
                if (max_vertex || unbounded_fetch) {
                    for (const auto& segment : draw.segments) {
                        if (!segment.count || !segment.instance_count) continue;
                        const auto instance =
                            (std::uint64_t(segment.base_instance) +
                             segment.instance_count - 1) / group.step->rate;
                        maximum = any ? std::max(maximum, instance) : instance;
                        any = true;
                    }
                }
                last = any ? maximum : 0;
                break;
            }
            case VertexStep::Constant:
                last = 0;
                break;
            }
            require(!group.plan->stride ||
                        last <= (std::numeric_limits<std::uint64_t>::max() -
                                 group.reached) / group.plan->stride,
                    "vertex binding extent overflows", CommandStatus::MemoryFault);
            const auto size = last * group.plan->stride + group.reached;
            require(size && size - 1 <=
                                group.plan->inclusive_limit - group.plan->address,
                    "vertex fetch exceeds its inclusive stream limit",
                    CommandStatus::MemoryFault);
            auto retained = take(memory.retain(group.plan->address, size,
                                               ResourceAccess::Read));
            direct_buffers.push_back({group.plan->address, retained,
                                      BufferUse::Vertex, 4,
                                      18 + group.stream});
        }

    }

    Bindings prepare_checked(const RenderStop& stop,
                             const AdmittedGraphicsStages& stages,
                             const DrawPlan& draw, ResourceMemory& memory,
                             BufferCache& buffers, TextureCache& textures,
                             const BindingReuse& reuse) {
        require(states_ != nullptr && states_->context().device != 0,
                "binding planner has no valid pipeline state service");
        require(stop.engine == engine_class::threed &&
                    stop.kind == RenderKind::Draw && stop.draw.has_value(),
                "binding preparation requires a stopped 3D draw");
        // The sets are validated by stamp below; repeats check stamps once more.
        reset_quiet();
        const auto& stopped_draw = *stop.draw;
        require(stopped_draw.indexed == draw.source.indexed &&
                    stopped_draw.topology == draw.source.topology &&
                    stopped_draw.first == draw.source.first &&
                    stopped_draw.count == draw.source.count &&
                    stopped_draw.base_vertex == draw.source.base_vertex &&
                    stopped_draw.base_instance == draw.source.base_instance &&
                    stopped_draw.instance_index == draw.source.instance_index &&
                    stopped_draw.instance_count == draw.source.instance_count,
                "draw plan does not belong to the supplied render stop");
        const ResidencyDomain expected{states_->context().device, &memory};
        require(buffers.domain() == expected && textures.domain() == expected,
                "binding caches do not share the planner residency domain");
        // A draw that changed no register group in the same epoch, with the
        // same stages and instancing and guest indices bound in place, binds
        // exactly what the previous draw bound.
        // Only a CPU store into what those bindings read forces a rebuild.
        auto* replay_registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
        const bool direct_indices = !draw.indices || draw.indices->converted.empty();
        if (replay_registry && !replay_registry->has_deferred_host_store() &&
            reuse.epoch && reuse.dirty == 0 && direct_indices && last_bindings_ &&
            last_bindings_->epoch == reuse.epoch && last_bindings_->vertex == stages.stages.vertex &&
            last_bindings_->fragment == stages.stages.fragment &&
            last_bindings_->indexed == bool(draw.indices) &&
            last_bindings_->base_instance == draw.source.base_instance &&
            last_bindings_->instance_count == draw.source.instance_count &&
            std::ranges::none_of(last_bindings_->reads, [&](const RetainedRange& range) {
                return replay_registry->cpu_written_since(range.bytes, last_bindings_->stamp);
            })) {
            auto replayed = last_bindings_->bindings;
            replayed.stages = stages;
            return replayed;
        }
        last_bindings_.reset();
        const auto replay_stamp = replay_registry ? replay_registry->cpu_write_stamp() : 0;
        std::vector<RetainedRange> replay_reads;
        replay_reads_ = &replay_reads;
        replayable_ = direct_indices && replay_stamp && reuse.epoch;
        struct StopRecording {
            std::vector<RetainedRange>*& target;
            ~StopRecording() { target = nullptr; }
        } stop_recording{replay_reads_};
        const auto groups = stream_groups(draw);
        // Stage validation depends on the stage pair, the texture handle
        // selector and the vertex stream set: skip it while all three equal
        // the last validated draw's.
        std::vector<std::uint32_t> streams;
        streams.reserve(groups.size());
        for (const auto& group : groups) streams.push_back(group.stream);
        if (validated_.vertex != stages.stages.vertex || validated_.fragment != stages.stages.fragment ||
            validated_.selector != stop.registers[0x982] || validated_.streams != streams ||
            validated_.attributes != draw.vertex.attributes.size()) {
            // Every combination validated before stays valid: the stages are
            // immutable and retained by the stage caches.
            auto key = std::tuple{static_cast<const void*>(stages.stages.vertex.get()),
                                  static_cast<const void*>(stages.stages.fragment.get()),
                                  stop.registers[0x982], streams, draw.vertex.attributes.size()};
            if (!validated_sets_.contains(key)) {
                validated_ = {};
                validate_stage_interfaces(stages, draw);
                std::set<std::uint32_t> vertex_indices;
                for (const auto& group : groups) vertex_indices.insert(18 + group.stream);
                validate_stage_resources(*stages.stages.vertex, shader_input::Stage::Vertex,
                                         stop.registers[0x982], vertex_indices);
                validate_stage_resources(*stages.stages.fragment, shader_input::Stage::Fragment,
                                         stop.registers[0x982], {});
                if (validated_sets_.size() >= 4096) validated_sets_.clear();
                validated_sets_.insert(std::move(key));
                validated_stages_.insert(stages.stages.vertex);
                validated_stages_.insert(stages.stages.fragment);
            }
            validated_ = {stages.stages.vertex, stages.stages.fragment, stop.registers[0x982],
                          std::move(streams), draw.vertex.attributes.size()};
        }

        Bindings result;
        result.stages = stages;

        std::vector<DirectBufferRequest> direct_buffers;
        vertex_index_requests(draw, groups, memory, buffers, result, direct_buffers);

        stage_buffers(stop, 0, shader_input::Stage::Vertex, stages.stages.vertex,
                      memory, buffers, direct_buffers, result, reuse);
        stage_buffers(stop, 4, shader_input::Stage::Fragment, stages.stages.fragment,
                      memory, buffers, direct_buffers, result, reuse);
        acquire_direct(memory, buffers, direct_buffers, result);

        stage_textures(stop, 0, shader_input::Stage::Vertex, stages.stages.vertex,
                       memory, buffers, textures, result, reuse);
        stage_textures(stop, 4, shader_input::Stage::Fragment, stages.stages.fragment,
                       memory, buffers, textures, result, reuse);
        // Unchanged draws take the renderer's repeat path; the complete
        // bindings are no longer kept for a second replay path.
        return result;
    }

    RendererResult<Bindings> prepare_compute(
        const RenderStop& stop, const CompiledGraphicsStage& stage, ResourceMemory& memory,
        BufferCache& buffers, TextureCache& textures) override {
        try {
            require(states_ != nullptr && states_->context().device != 0,
                    "binding planner has no valid pipeline state service");
            require(stop.engine == engine_class::compute && stop.kind == RenderKind::Compute &&
                        stop.compute.has_value(),
                    "compute binding preparation requires a stopped compute launch");
            const ResidencyDomain expected{states_->context().device, &memory};
            require(buffers.domain() == expected && textures.domain() == expected,
                    "binding caches do not share the planner residency domain");
            validate_stage_resources(stage, shader_input::Stage::Compute,
                                     stop.registers[0x982], {});
            // The compute class programs its texture header pool, sampler pool
            // and bindless constant-buffer selector at the 3D class's method
            // offsets; the queue
            // meta data supplies the constant buffers and the sampler index
            // mode. Resolution runs on a 3D-shaped view of that state, with
            // the launch's slots in stage 0.
            const auto& launch = *stop.compute;
            RenderStop view = stop;
            view.engine = engine_class::threed;
            view.kind = RenderKind::Draw;
            view.constant_buffers = {};
            for (std::size_t slot = 0; slot != launch.constant_buffers.size(); ++slot) {
                const auto& binding = launch.constant_buffers[slot];
                view.constant_buffers[0][slot] = {binding.address, binding.size, binding.enabled};
            }
            view.registers[0x48d] = launch.sampler_index ? 1u : 0u;
            Bindings result;
            std::vector<DirectBufferRequest> direct_buffers;
            prepare_stage_buffers(view, 0, shader_input::Stage::Compute, stage, memory, buffers,
                                  direct_buffers, result);
            acquire_direct(memory, buffers, direct_buffers, result);
            prepare_stage_textures(view, 0, shader_input::Stage::Compute, stage, memory, buffers,
                                   textures, result);
            return result;
        } catch (const RendererError& error) {
            return error;
        } catch (const std::bad_alloc&) {
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "compute binding preparation exceeded the host allocation budget"}};
        } catch (const std::length_error&) {
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "compute binding preparation exceeds a supported container extent"}};
        }
    }

private:
    // L5 cannot replace a partially overlapping image while an earlier
    // binding retains it. Prepare each complete overlap component across
    // index, vertex and every stage's storage before acquiring its views.
    void acquire_direct(ResourceMemory& memory, BufferCache& buffers,
                        const std::vector<DirectBufferRequest>& direct_buffers,
                        Bindings& result) {
        // Vertex and index ranges bind their backing's zero-copy pool at
        // their own offsets (Armigo binds every buffer as pool plus
        // offset); the rest take resident copies, preflighted as one union
        // per overlap component.
        std::vector<std::optional<GuestBinding>> pooled(direct_buffers.size());
        std::vector<DirectBufferRequest> copied;
        for (std::size_t index = 0; index != direct_buffers.size(); ++index) {
            const auto& request = direct_buffers[index];
            if (request.use == BufferUse::Vertex || request.use == BufferUse::Index) {
                auto pool = guest_pool_slice(memory, request.retained);
                if (pool && !(pool->offset % request.alignment)) {
                    pooled[index] = std::move(pool);
                    continue;
                }
            }
            copied.push_back(request);
        }
        const auto preflight = preflight_overlapping_buffers(memory, buffers, copied);
        (void)preflight;
        for (std::size_t index = 0; index != direct_buffers.size(); ++index) {
            const auto& request = direct_buffers[index];
            ResidentBuffer resident;
            if (const auto& pool = pooled[index]) {
                resident.key.bytes = request.retained.bytes;
                resident.storage = pool->slice;
                resident.offset = pool->offset;
                resident.size = request.retained.bytes.size;
            } else {
                resident = take(buffers.acquire(
                    {request.retained, request.use, request.access,
                     request.alignment, request.native_extent}));
                append_dependencies(result.dependencies, resident.dependencies);
            }
            result.accesses.push_back({request.retained, request.access});
            if (request.shader_binding) {
                result.buffers.push_back(
                    {request.shader_binding->first, request.shader_binding->second,
                     std::move(resident), request.use, request.access});
            } else if (request.vertex_index) {
                result.vertex_buffers.push_back(
                    {shader_input::Stage::Vertex, *request.vertex_index,
                     std::move(resident)});
            } else {
                result.index_buffer = std::move(resident);
            }
        }
    }

    // A stage's constant-buffer bindings of an earlier draw, reused while the
    // stage, its constant-buffer binds and the bytes they read are unchanged.
    // Stages with storage descriptors or GPU-written windows always rebuild.
    struct WindowRecord {
        std::uint32_t index = 0;
        std::uint8_t slot = 0;
        std::uint64_t address = 0;
        std::uint32_t copied = 0, window_bytes = 0;
        RetainedRange range;
    };
    // Few stamp checks per cache: reads on one backing closer than 64 KiB
    // share a covering range. A store inside a covering range costs a
    // rebuild, never staleness; distant reads stay separate so stores to
    // unrelated parts of a large heap leave the cache valid.
    static void merge_nearby(std::vector<RetainedRange>& reads) {
        constexpr std::uint64_t gap = 64 * 1024;
        std::ranges::sort(reads, [](const RetainedRange& a, const RetainedRange& b) {
            return std::tie(a.bytes.backing.owner, a.bytes.backing.allocation, a.bytes.backing.generation,
                            a.bytes.offset) <
                   std::tie(b.bytes.backing.owner, b.bytes.backing.allocation, b.bytes.backing.generation,
                            b.bytes.offset);
        });
        std::vector<RetainedRange> merged;
        for (auto& range : reads) {
            if (!merged.empty() && merged.back().bytes.backing == range.bytes.backing &&
                range.bytes.offset <= merged.back().bytes.offset + merged.back().bytes.size + gap) {
                auto& last = merged.back().bytes;
                last.size = std::max(last.offset + last.size, range.bytes.offset + range.bytes.size) - last.offset;
                continue;
            }
            merged.push_back(std::move(range));
        }
        reads = std::move(merged);
    }

    struct StageBuffers {
        std::shared_ptr<const CompiledGraphicsStage> stage;
        std::uint64_t epoch = 0, stamp = 0;
        std::vector<RetainedRange> reads;
        // Per uniform window and direct constant buffer, for repeats.
        std::vector<WindowRecord> windows;
        std::vector<RetainedRange> direct;
        std::vector<BufferBinding> buffers;
        std::vector<CompletionToken> dependencies;
        // The stage's enabled constant buffers: snapshots and descriptor words
        // come from them, so later epochs reuse the set while GPU writes
        // logged since miss these ranges.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> gpu_reads;
        // The stage's constant-buffer bindings the set was built from.
        std::array<ConstantBufferDescription, 18> bindings{};
    };
    std::array<StageBuffers, 5> stage_buffers_;
    static std::vector<RetainedRange> window_ranges(const StageBuffers& stage) {
        std::vector<RetainedRange> ranges;
        ranges.reserve(stage.windows.size());
        for (const auto& window : stage.windows) ranges.push_back(window.range);
        return ranges;
    }
    // Earlier stages' buffer sets by compiled stage (see texture_sets_).
    std::array<std::unordered_map<const CompiledGraphicsStage*, StageBuffers>, 5> buffer_sets_;
    static bool same_bindings(const std::array<ConstantBufferDescription, 18>& a,
                              const std::array<ConstantBufferDescription, 18>& b) {
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].address != b[i].address || a[i].size != b[i].size || a[i].enabled != b[i].enabled)
                return false;
        return true;
    }

    void stage_buffers(const RenderStop& stop, std::size_t stage_index, shader_input::Stage shader_stage,
                       const std::shared_ptr<const CompiledGraphicsStage>& stage, ResourceMemory& memory,
                       BufferCache& buffers, std::vector<DirectBufferRequest>& requests, Bindings& result,
                       const BindingReuse& reuse) {
        auto& cache = stage_buffers_[stage_index];
        auto* registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
        auto& sets = buffer_sets_[stage_index];
        switch_set(cache, sets, stage);
        const auto checked_stamp = registry ? registry->cpu_write_stamp() : 0;
        const bool unchanged = registry && reuse.epoch && cache.stamp && cache.stage == stage &&
            (cache.epoch == reuse.epoch || (reuse.log && reuse.log->clean_since(cache.epoch, cache.gpu_reads, cache.reads) &&
                                            reuse.log->clean_since(cache.epoch, {}, window_ranges(cache)))) &&
            std::ranges::none_of(cache.reads, [&](const RetainedRange& range) {
                return registry->cpu_written_since(range.bytes, cache.stamp);
            });
        // Uniform windows a CPU store reached are taken again, as repeats
        // refresh them, instead of rebuilding the stage. Windows a repeat
        // moved are tracked by their records only.
        std::vector<std::size_t> stale;
        if (unchanged)
            for (std::size_t i = 0; i < cache.windows.size(); ++i)
                if (registry->cpu_written_since(cache.windows[i].range.bytes, cache.stamp)) stale.push_back(i);
        bool reusable = unchanged && same_bindings(cache.bindings, stop.constant_buffers[stage_index]);
        // Outcome counts: same bindings, moved windows, no cache, stage,
        // logged GPU writes, stored reads, unknown GPU writes, rebound;
        // plus draws with stale windows.
        static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        static std::array<std::uint64_t, 9> outcomes{};
        std::size_t outcome = 0;
        if (!unchanged) {
            if (!registry || !reuse.epoch || !cache.stamp) outcome = 2;
            else if (cache.stage != stage) outcome = 3;
            else if (cache.epoch != reuse.epoch && reuse.log && reuse.log->last_unknown >= cache.epoch) outcome = 6;
            else if (cache.epoch != reuse.epoch &&
                     !(reuse.log && reuse.log->clean_since(cache.epoch, cache.gpu_reads, cache.reads) &&
                       reuse.log->clean_since(cache.epoch, {}, window_ranges(cache)))) outcome = 4;
            else outcome = 5;
        } else if (!reusable) {
            outcome = 7;
        }
        struct Report {
            const bool& trace;
            std::array<std::uint64_t, 9>& outcomes;
            std::size_t& outcome;
            bool stale;
            ~Report() {
                if (!trace) return;
                ++outcomes[outcome];
                if (stale) ++outcomes[8];
                const auto total = outcomes[0] + outcomes[1] + outcomes[2] + outcomes[3] + outcomes[4] +
                                   outcomes[5] + outcomes[6] + outcomes[7];
                if (total % 50000) return;
                armigo::diag::opt_in( "stage_buffer_reuse same=%llu moved=%llu none=%llu stage=%llu logged=%llu "
                    "stored=%llu unknown=%llu rebound=%llu stale=%llu\n", (unsigned long long)outcomes[0],
                    (unsigned long long)outcomes[1], (unsigned long long)outcomes[2], (unsigned long long)outcomes[3],
                    (unsigned long long)outcomes[4], (unsigned long long)outcomes[5], (unsigned long long)outcomes[6],
                    (unsigned long long)outcomes[7], (unsigned long long)outcomes[8]);
            }
        } report{trace, outcomes, outcome, !stale.empty()};
        if (unchanged && !reusable && checked_stamp) {
            // Only uniform windows moved to other addresses of their size: take
            // those windows again instead of rebuilding the stage.
            std::vector<Moved> moved;
            if (moved_windows(stop, stage_index, memory, moved)) {
                try {
                    for (auto& change : moved) {
                        std::erase(stale, change.record);
                        auto& window = cache.windows[change.record];
                        window.address = stop.constant_buffers[stage_index][window.slot].address;
                        window.range = std::move(change.range);
                        auto resident = snapshot_window(memory, buffers, window.address, window.copied,
                                                        window.window_bytes, &window.range);
                        for (auto& bound : cache.buffers)
                            if (bound.stage == shader_stage && bound.index == window.index) bound.buffer = resident;
                    }
                    cache.bindings = stop.constant_buffers[stage_index];
                    cache.gpu_reads.clear();
                    for (const auto& bound : cache.bindings)
                        if (bound.enabled) cache.gpu_reads.emplace_back(bound.address, bound.size);
                    // Every read was verified unwritten after this stamp was taken.
                    cache.stamp = checked_stamp;
                    reusable = true;
                    outcome = 1;
                } catch (const RendererError&) {
                    reusable = false;
                }
            }
        }
        if (reusable && !stale.empty()) {
            try {
                if (!checked_stamp) reject(CommandStatus::Unsupported, "stale uniform windows need a write stamp");
                for (const auto i : stale) {
                    const auto& window = cache.windows[i];
                    auto resident = snapshot_window(memory, buffers, window.address, window.copied,
                                                    window.window_bytes, &window.range);
                    for (auto& bound : cache.buffers)
                        if (bound.stage == shader_stage && bound.index == window.index) bound.buffer = resident;
                }
                cache.stamp = checked_stamp;
            } catch (const RendererError&) {
                reusable = false;
            }
        }
        if (reusable) {
            cache.epoch = reuse.epoch;
            result.buffers.insert(result.buffers.end(), cache.buffers.begin(), cache.buffers.end());
            append_dependencies(result.dependencies, cache.dependencies);
            if (replay_reads_) {
                replay_reads_->insert(replay_reads_->end(), cache.reads.begin(), cache.reads.end());
                for (const auto& window : cache.windows) replay_reads_->push_back(window.range);
            }
            return;
        }
        if (cache.stage != stage) keep_set(cache, sets);
        cache = {};
        const auto stamp = registry ? registry->cpu_write_stamp() : 0;
        const auto first_buffer = result.buffers.size(), first_request = requests.size();
        const auto first_dependency = result.dependencies.size();
        std::vector<RetainedRange> reads, direct;
        std::vector<WindowRecord> windows;
        auto* outer = replay_reads_;
        const bool outer_replayable = replayable_;
        replay_reads_ = &reads;
        direct_reads_ = &direct;
        window_records_ = &windows;
        replayable_ = true;
        struct Restore {
            std::vector<RetainedRange>*& target;
            std::vector<RetainedRange>* previous;
            std::vector<RetainedRange>*& direct;
            std::vector<WindowRecord>*& windows;
            ~Restore() {
                target = previous;
                direct = nullptr;
                windows = nullptr;
            }
        } restore{replay_reads_, outer, direct_reads_, window_records_};
        prepare_stage_buffers(stop, stage_index, shader_stage, *stage, memory, buffers, requests, result);
        const bool cacheable = replayable_ && requests.size() == first_request && stamp && reuse.epoch;
        replayable_ = outer_replayable && replayable_;
        if (outer) outer->insert(outer->end(), reads.begin(), reads.end());
        if (!cacheable) return;
        auto merged = std::move(reads);
        merge_nearby(merged);
        cache.stage = stage;
        cache.epoch = reuse.epoch;
        cache.stamp = stamp;
        cache.reads = std::move(merged);
        cache.bindings = stop.constant_buffers[stage_index];
        for (const auto& bound : stop.constant_buffers[stage_index])
            if (bound.enabled) cache.gpu_reads.emplace_back(bound.address, bound.size);
        cache.windows = std::move(windows);
        cache.direct = std::move(direct);
        cache.buffers.assign(result.buffers.begin() + std::ptrdiff_t(first_buffer), result.buffers.end());
        cache.dependencies.assign(result.dependencies.begin() + std::ptrdiff_t(first_dependency),
                                  result.dependencies.end());
    }

    void prepare_stage_buffers(const RenderStop& stop, std::size_t stage_index,
                       shader_input::Stage shader_stage, const CompiledGraphicsStage& stage,
                       ResourceMemory& memory, BufferCache& buffers,
                       std::vector<DirectBufferRequest>& requests, Bindings& result) {
        struct StorageDescriptor {
            const CompiledStorageBufferBinding* binding;
            std::uint64_t address;
            std::uint32_t size;
        };
        std::vector<StorageDescriptor> storage;
        for (const auto& binding : stage.constant_buffers) {
            const auto& guest = stop.constant_buffers[stage_index][binding.guest_binding];
            require(guest.enabled && guest.size >= binding.minimum_size,
                    "constant buffer is disabled or smaller than reflection");
            auto retained = take(memory.retain(
                guest.address, binding.minimum_size, ResourceAccess::Read));
            if (replay_reads_) replay_reads_->push_back(retained);
            if (direct_reads_) direct_reads_->push_back(retained);
            auto resident = take(buffers.acquire(
                {retained, BufferUse::Uniform, ResourceAccess::Read, 4}));
            append_dependencies(result.dependencies, resident.dependencies);
            result.buffers.push_back(
                {shader_stage, binding.metal_buffer_index, std::move(resident)});
        }
        for (const auto& binding : stage.constant_buffer_windows) {
            const auto& guest = stop.constant_buffers[stage_index][binding.guest_binding];
            std::vector<std::byte> bytes(binding.window_bytes);
            // Whole words below the bound size; the rest of the window reads
            // zero (a word only partly inside the bound reads zero).
            const auto copied = guest.enabled
                                    ? std::min(guest.size, binding.window_bytes) & ~3u
                                    : 0u;
            // GPU-produced constants (a compute pass writing uniform data)
            // bind their resident image with a zero tail to the window size,
            // because constant buffers remain in GPU memory, instead of
            // a CPU snapshot that would first write the results back. Windows
            // holding storage descriptors keep the snapshot their CPU decode
            // reads.
            // CPU-backed windows reuse the bytes of their previous snapshot
            // while no store reached them, because dirty-page stamps prove
            // that pool memory is unchanged. The upload itself stays per draw.
            std::optional<WindowKey> window;
            std::vector<WindowInterval> window_state;
            auto* registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
            if (copied && std::ranges::none_of(stage.storage_buffers, [&](const auto& descriptor) {
                    return descriptor.cbuf_slot == binding.guest_binding;
                })) {
                auto retained = memory.retain(guest.address, copied, ResourceAccess::Read);
                if (auto* range = std::get_if<RetainedRange>(&retained); range && replay_reads_)
                    replay_reads_->push_back(*range);
                else if (copied) replayable_ = false;
                if (auto* range = std::get_if<RetainedRange>(&retained)) {
                    auto state = memory.inspect(range->bytes);
                    if (auto* intervals = std::get_if<std::vector<DirtyRange>>(&state);
                        intervals && std::ranges::any_of(*intervals, [](const auto& interval) {
                            return interval.gpu_dirty;
                        })) {
                        requests.push_back({guest.address, std::move(*range), BufferUse::Uniform, 4,
                            std::nullopt, std::pair{shader_stage, std::uint32_t(binding.metal_buffer_index)},
                            ResourceAccess::Read, binding.window_bytes});
                        continue;
                    } else if (intervals && registry) {
                        if (window_records_) {
                            window_records_->push_back({std::uint32_t(binding.metal_buffer_index),
                                std::uint8_t(binding.guest_binding), guest.address, copied, binding.window_bytes,
                                *range});
                            // The record tracks the window: a store into it
                            // takes the window again instead of rebuilding
                            // the stage.
                            if (replay_reads_ && !replay_reads_->empty() &&
                                replay_reads_->back().bytes == range->bytes)
                                replay_reads_->pop_back();
                        }
                        const auto& bytes = range->bytes;
                        window = WindowKey{bytes.backing.owner, bytes.backing.allocation,
                            bytes.backing.generation, bytes.offset, bytes.size, binding.window_bytes};
                        for (const auto& interval : *intervals)
                            window_state.push_back({interval.bytes.offset, interval.bytes.size, interval.version});
                        const auto found = windows_.find(*window);
                        if (found != windows_.end() && found->second.state == window_state &&
                            !registry->cpu_written_since(bytes, found->second.stamp)) {
                            // The slot's last upload of this very snapshot is
                            // still valid: ring slices are never rewritten.
                            auto& last = last_uploads_[stage_index][binding.metal_buffer_index % 32];
                            if (last && last->serial == found->second.serial) {
                                result.buffers.push_back({shader_stage, binding.metal_buffer_index, last->resident});
                                continue;
                            }
                            auto resident = take(buffers.upload(found->second.bytes, 4));
                            last = LastUpload{found->second.serial, resident};
                            append_dependencies(result.dependencies, resident.dependencies);
                            result.buffers.push_back(
                                {shader_stage, binding.metal_buffer_index, std::move(resident)});
                            continue;
                        }
                    }
                }
            }
            // Take the stamp before the snapshot reads any byte it keeps.
            const auto stamp = window ? registry->cpu_write_stamp() : 0;
            if (copied) {
                try {
                    take(snapshot_buffer_into(memory, guest.address, std::span(bytes).first(copied),
                        BufferSnapshotPolicy::ZeroMissingConstantBufferBytes));
                } catch (const RendererError& error) {
                    reject(error.command().status,
                        "constant buffer stage=" + std::to_string(stage_index) +
                        " slot=" + std::to_string(binding.guest_binding) + ": " +
                        std::string(error.command().detail), guest.address);
                }
            }
            // Decode the exact immutable words passed to the shader. In
            // particular, the bounds guard and resident range share one size.
            for (const auto& descriptor : stage.storage_buffers) {
                if (descriptor.cbuf_slot != binding.guest_binding) continue;
                replayable_ = false;
                require(guest.enabled && descriptor.descriptor_byte_offset + 12 <= copied,
                        "storage descriptor constant buffer is disabled or truncated",
                        CommandStatus::MemoryFault);
                // Missing ordinary cbuf bytes may be synthesized, but a
                // descriptor must be physically present. Decode and upload the
                // same strict snapshot so zero-filled holes cannot bind null.
                auto descriptor_bytes = take(snapshot_buffer_range(memory,
                    guest.address + descriptor.descriptor_byte_offset, 12));
                std::copy(descriptor_bytes.begin(), descriptor_bytes.end(),
                    bytes.begin() + descriptor.descriptor_byte_offset);
                const auto* words = descriptor_bytes.data();
                const auto address = std::uint64_t(load_index(words, IndexFormat::U32)) |
                    (std::uint64_t(load_index(words + 4, IndexFormat::U32)) << 32);
                storage.push_back({&descriptor, address, load_index(words + 8, IndexFormat::U32)});
                if ([] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }()) {
                    const auto size = storage.back().size;
                    armigo::diag::opt_in(
                        "storage_descriptor stage=%zu cbuf=%u offset=%x metal=%u address=%llx size=%x\n",
                        stage_index, unsigned(descriptor.cbuf_slot), descriptor.descriptor_byte_offset,
                        unsigned(descriptor.metal_buffer_index), static_cast<unsigned long long>(address), size);
                    if (size <= std::numeric_limits<std::uint64_t>::max() - address) {
                        for (const auto& mapping : memory.address_space().mappings()) {
                            if (mapping.gpu_base >= address + size ||
                                mapping.gpu_base + mapping.mapped_size <= address) continue;
                            armigo::diag::opt_in(
                                "storage_mapping gpu=%llx size=%llx offset=%llx backing_size=%llx cpu=%llx\n",
                                static_cast<unsigned long long>(mapping.gpu_base),
                                static_cast<unsigned long long>(mapping.mapped_size),
                                static_cast<unsigned long long>(mapping.buffer_offset),
                                static_cast<unsigned long long>(mapping.backing_size),
                                static_cast<unsigned long long>(mapping.guest_address));
                        }
                    }
                }
            }
            if (window && stamp) {
                if (window_bytes_ > window_budget) {
                    windows_.clear();
                    window_bytes_ = 0;
                }
                auto& cached = windows_[*window];
                window_bytes_ += bytes.size() - cached.bytes.size();
                cached = {stamp, ++window_serial_, std::move(window_state), bytes};
            }
            auto resident = take(buffers.upload(bytes, 4));
            if (window && stamp)
                last_uploads_[stage_index][binding.metal_buffer_index % 32] =
                    LastUpload{window_serial_, resident};
            append_dependencies(result.dependencies, resident.dependencies);
            result.buffers.push_back(
                {shader_stage, binding.metal_buffer_index, std::move(resident)});
        }
        if (stage.constant_buffer_support_index) {
            std::array<std::byte, 18 * sizeof(std::uint32_t)> bytes{};
            for (std::size_t slot = 0; slot != 18; ++slot) {
                const auto& guest = stop.constant_buffers[stage_index][slot];
                store_u32_le(bytes.data() + slot * sizeof(std::uint32_t),
                             guest.enabled ? guest.size : 0u);
            }
            auto resident = take(buffers.upload(bytes, 4));
            append_dependencies(result.dependencies, resident.dependencies);
            result.buffers.push_back(
                {shader_stage, *stage.constant_buffer_support_index,
                 std::move(resident)});
        }

        for (const auto& descriptor : storage) {
            const auto& binding = *descriptor.binding;
            if (!descriptor.size) {
                // A null binding has no guest access. MSL still requires a
                // valid pointer; its descriptor-size guard forbids dereference.
                const std::array<std::byte, 4> zero{};
                auto resident = take(buffers.upload(zero, binding.alignment));
                append_dependencies(result.dependencies, resident.dependencies);
                result.buffers.push_back({shader_stage, binding.metal_buffer_index,
                    std::move(resident), BufferUse::Storage, ResourceAccess::Read});
                continue;
            }
            require(descriptor.address <= std::numeric_limits<std::uint64_t>::max() - descriptor.size,
                    "storage descriptor range overflows", CommandStatus::MemoryFault);
            const auto& mappings = memory.address_space().mappings();
            const auto mapping = std::ranges::find_if(mappings, [&](const auto& candidate) {
                return descriptor.address >= candidate.gpu_base &&
                    descriptor.address - candidate.gpu_base < candidate.mapped_size &&
                    descriptor.size <= candidate.mapped_size - (descriptor.address - candidate.gpu_base);
            });
            require(mapping != mappings.end(), "storage descriptor range is unmapped or crosses a mapping boundary",
                    CommandStatus::MemoryFault);
            // A big-page mapping can extend past its memory handle. Bind the
            // backed prefix with a zero native tail up to the descriptor size
            // the shader guards against; the tail is never guest-visible.
            const auto backed_start = mapping->buffer_offset + (descriptor.address - mapping->gpu_base);
            const std::uint64_t backed = backed_start < mapping->backing_size
                ? std::min<std::uint64_t>(descriptor.size, mapping->backing_size - backed_start) : 0;
            require(backed != 0, "storage descriptor starts past its mapping's guest backing",
                    CommandStatus::MemoryFault);
            // A shader store may touch only part of the descriptor. Preserve
            // the remaining bytes and publish the allocation as GPU-written.
            const auto access = binding.access == CompiledShaderResourceAccess::Read
                ? ResourceAccess::Read : ResourceAccess::ReadWrite;
            auto retained_result = memory.retain(descriptor.address, backed, access);
            if (const auto* failure = std::get_if<RendererError>(&retained_result))
                reject(failure->command().status,
                    "storage buffer stage=" + std::to_string(stage_index) +
                    " cbuf=" + std::to_string(binding.cbuf_slot) +
                    " descriptor=" + std::to_string(binding.descriptor_byte_offset) +
                    " metal=" + std::to_string(binding.metal_buffer_index) + ": " +
                    std::string(failure->command().detail), descriptor.address);
            auto retained = std::get<RetainedRange>(std::move(retained_result));
            // A fully backed range binds guest memory in place: the shader's
            // writes land there, with no resident copy or growth.
            if (backed == descriptor.size) {
                if (auto pool = guest_pool_slice(memory, retained); pool && !(pool->offset & 3u)) {
                    ResidentBuffer resident;
                    resident.key.bytes = retained.bytes;
                    resident.storage = pool->slice;
                    resident.offset = pool->offset;
                    resident.size = backed;
                    result.accesses.push_back({retained, access});
                    result.buffers.push_back({shader_stage, std::uint32_t(binding.metal_buffer_index),
                        std::move(resident), BufferUse::Storage, access});
                    continue;
                }
            }
            requests.push_back({descriptor.address, std::move(retained), BufferUse::Storage,
                binding.alignment, std::nullopt,
                std::pair{shader_stage, std::uint32_t(binding.metal_buffer_index)}, access,
                backed < descriptor.size ? std::uint64_t{descriptor.size} : 0});
        }
    }

    // A stage's texture and sampler bindings of an earlier draw may reuse the
    // previous argument buffer within the same pass while no binding,
    // registration or dirty-page generation changed. Here the stage, its
    // constant-buffer bindings and texture pools must be clean, the renderer
    // epoch unchanged (no GPU work other than this batch's draws), no CPU
    // store may have reached the handle words, pools or textures since the
    // stamp, and no sampled texture may overlap a current attachment.
    // A 32-byte TIC or TSC entry with the bytes it held when read.
    struct PoolEntry {
        RetainedRange range;
        const std::byte* host = nullptr;
        std::array<std::byte, 32> bytes{};
    };
    // Whether a pool entry still holds its bytes, peeked only after a store
    // reached its page since `stamp`.
    static bool same_entry(const PoolEntry& entry, ResourceMemoryRegistry& registry, std::uint64_t stamp) {
        // A deferred store has not reached the bytes yet: it counts as a change.
        if (registry.deferred_host_store(entry.range.bytes)) return false;
        return !registry.cpu_written_since(entry.range.bytes, stamp) ||
               std::memcmp(entry.host, entry.bytes.data(), entry.bytes.size()) == 0;
    }
    // Retains a pool entry and captures its bytes from host-addressable
    // guest memory; false when the entry has no host view.
    static bool capture_entry(ResourceMemory& memory, ResourceMemoryRegistry& registry, std::uint64_t address,
                              PoolEntry& entry) {
        auto retained = memory.retain(address, 32, ResourceAccess::Read);
        auto* range = std::get_if<RetainedRange>(&retained);
        if (!range) return false;
        if (registry.deferred_host_store(range->bytes)) return false;
        const auto host = registry.backing_host_range(range->bytes.backing);
        if (!host || range->bytes.offset > host->second || host->second - range->bytes.offset < 32) return false;
        entry.host = reinterpret_cast<const std::byte*>(host->first + range->bytes.offset);
        std::memcpy(entry.bytes.data(), entry.host, entry.bytes.size());
        entry.range = std::move(*range);
        return true;
    }

    static EpochLog::ResidentRead resident_read(const ResidentTexture& texture) {
        const auto& descriptor = texture.key.storage.descriptor;
        return {descriptor.address, descriptor.backing_extent, texture.key.storage.bytes,
                resident_texture_identity(texture)};
    }

    // acquire_checked returns every Accepted/Submitted GPU writer and propagates
    // producer failures. A same-allocation replay needs the same ordering without
    // descriptor resolution, allocation selection or content revalidation.
    static void refresh_texture_dependencies(ResourceMemory& memory,
                                             std::span<const BackingRange> writes,
                                             std::vector<CompletionToken>& dependencies) {
        if (writes.empty()) return;
        std::erase_if(dependencies, [](const CompletionToken& token) {
            return token.snapshot().phase == SubmissionPhase::Complete;
        });
        for (const auto& bytes : writes) {
            const auto intervals = take(memory.inspect(bytes));
            for (const auto& interval : intervals) {
                if (!interval.gpu_dirty) continue;
                const auto snapshot = interval.last_writer.snapshot();
                if (snapshot.error) throw *snapshot.error;
                if ((snapshot.phase == SubmissionPhase::Accepted || snapshot.phase == SubmissionPhase::Submitted) &&
                    std::ranges::none_of(dependencies, [&](const CompletionToken& token) {
                        return token.id() == interval.last_writer.id();
                    })) dependencies.push_back(interval.last_writer);
            }
        }
    }

    struct StageTextures {
        std::shared_ptr<const CompiledGraphicsStage> stage;
        std::uint64_t epoch = 0, stamp = 0;
        std::vector<RetainedRange> reads;
        // Bindless handle words, checked by value: their constant buffers share
        // pages with per-frame uniform stores, so a page stamp alone would
        // rebuild the set although the handles kept their values.
        std::vector<RetainedRange> handle_ranges;
        std::vector<std::uint64_t> handle_addresses;
        std::vector<std::uint32_t> handle_values;
        // Constant-buffer slots the handles were read through, with the
        // binding each had: other slots' binds leave the textures valid.
        std::vector<std::pair<std::uint8_t, ConstantBufferDescription>> handle_slots;
        // Descriptor pool entries the handles index, checked by value once a
        // store reaches their page: pools share pages with entries other
        // draws rewrite.
        std::vector<PoolEntry> entries;
        std::array<std::uint32_t, 8> registers{};
        std::vector<std::pair<std::uint64_t, std::uint64_t>> sampled;
        // Handle/pool reads require unchanged bytes. Texture reads may retain
        // their binding across writes to the exact same resident allocation.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> gpu_reads;
        std::vector<EpochLog::ResidentRead> resident_reads;
        std::vector<ArgumentBufferBinding> argument_buffers;
        std::vector<SamplerBinding> samplers;
        std::vector<TextureBinding> textures;
        std::vector<GpuAccessRequest> accesses;
        std::vector<CompletionToken> dependencies;
        std::vector<RetainedResource> border_textures;
        std::optional<std::uint32_t> border_index;
        std::vector<std::byte> border_bytes;
    };
    std::array<StageTextures, 5> stage_textures_;
    // Earlier stages' texture sets, most recent last, so alternating programs
    // find theirs again; each is validated like the current one.
    std::array<std::unordered_map<const CompiledGraphicsStage*, StageTextures>, 5> texture_sets_;
    // A title alternates among hundreds of programs; past the limit the sets
    // start over. Each set retains its stage, keeping the key valid.
    static constexpr std::size_t texture_set_limit = 1024;
    // Swaps in this stage's earlier set; the current one is kept by its stage.
    template<class Set>
    static void switch_set(Set& cache, std::unordered_map<const CompiledGraphicsStage*, Set>& sets,
                           const std::shared_ptr<const CompiledGraphicsStage>& stage) {
        if (cache.stage == stage) return;
        auto found = sets.find(stage.get());
        if (found == sets.end()) return;
        Set earlier = std::move(found->second);
        sets.erase(found);
        keep_set(cache, sets);
        cache = std::move(earlier);
    }
    template<class Set>
    static void keep_set(Set& cache, std::unordered_map<const CompiledGraphicsStage*, Set>& sets) {
        if (!cache.stamp || !cache.stage) return;
        if (sets.size() >= texture_set_limit) sets.clear();
        sets.insert_or_assign(cache.stage.get(), std::move(cache));
    }
    // Registers texture and sampler resolution reads: sampler binding mode,
    // both descriptor pools and the bindless handle constant-buffer slot.
    static constexpr std::array<std::uint16_t, 8> texture_registers{
        0x48d, 0x557, 0x558, 0x559, 0x55d, 0x55e, 0x55f, 0x982};
    static std::array<std::uint32_t, 8> texture_inputs(const RenderStop& stop) {
        std::array<std::uint32_t, 8> values{};
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = stop.registers[texture_registers[i]];
        return values;
    }
    // The previous draw's complete bindings for replay, the guest ranges
    // they read and the write stamp taken before reading them.
    struct LastBindings {
        std::uint64_t epoch = 0;
        std::shared_ptr<const CompiledGraphicsStage> vertex, fragment;
        bool indexed = false;
        std::uint32_t base_instance = 0, instance_count = 0;
        std::uint64_t stamp = 0;
        std::vector<RetainedRange> reads;
        Bindings bindings;
    };
    std::optional<LastBindings> last_bindings_;
    std::vector<RetainedRange>* replay_reads_ = nullptr;
    std::vector<RetainedRange>* direct_reads_ = nullptr;
    std::vector<WindowRecord>* window_records_ = nullptr;
    bool replayable_ = false;
    struct Validated {
        std::shared_ptr<const CompiledGraphicsStage> vertex, fragment;
        std::uint32_t selector = 0;
        std::vector<std::uint32_t> streams;
        std::size_t attributes = 0;
    } validated_;
    std::set<std::tuple<const void*, const void*, std::uint32_t, std::vector<std::uint32_t>, std::size_t>>
        validated_sets_;
    // Keeps validated stages alive so their addresses stay unique identities.
    std::set<std::shared_ptr<const CompiledGraphicsStage>> validated_stages_;
    // For a draw repeating the last one: everything the stages bound is
    // still valid except uniform windows a CPU store reached, which are
    // snapshotted again; any other change needs a full preparation.
    // A window whose slot moved to another address (same size): the new
    // window range, retained and checked before anything is replaced.
    struct Moved { std::size_t record; RetainedRange range; };
    // A constant-buffer rebind that only moves CPU-backed uniform windows to
    // other addresses of the same size re-snapshots those windows instead of
    // rebuilding the stage. Changed slots holding direct buffers or storage
    // descriptors, size or enable changes, and GPU-written targets need the
    // full rebuild (false).
    bool moved_windows(const RenderStop& stop, std::size_t index, ResourceMemory& memory,
                       std::vector<Moved>& out) {
        const auto& stage = stage_buffers_[index];
        if (!stage.stage) return false;
        const auto& compiled = *stage.stage;
        for (std::uint8_t slot = 0; slot < stage.bindings.size(); ++slot) {
            const auto& before = stage.bindings[slot];
            const auto& now = stop.constant_buffers[index][slot];
            if (before.address == now.address && before.size == now.size && before.enabled == now.enabled)
                continue;
            if (before.size != now.size || before.enabled != now.enabled) return false;
            if (std::ranges::any_of(compiled.constant_buffers, [&](const auto& b) { return b.guest_binding == slot; }) ||
                std::ranges::any_of(compiled.storage_buffers, [&](const auto& b) { return b.cbuf_slot == slot; }))
                return false;
            bool used = false;
            for (std::size_t record = 0; record < stage.windows.size(); ++record) {
                const auto& window = stage.windows[record];
                if (window.slot != slot) continue;
                used = true;
                auto retained = memory.retain(now.address, window.copied, ResourceAccess::Read);
                auto* range = std::get_if<RetainedRange>(&retained);
                if (!range) return false;
                auto written = memory.gpu_written(range->bytes);
                auto* gpu_written = std::get_if<bool>(&written);
                if (!gpu_written || *gpu_written) return false;
                out.push_back({record, std::move(*range)});
            }
            // A changed slot the stage reads through no recorded window.
            if (!used && std::ranges::any_of(compiled.constant_buffer_windows,
                                             [&](const auto& b) { return b.guest_binding == slot; }))
                return false;
        }
        return true;
    }

    // Engine stores reported since the last stamp-checked refresh, by GPU
    // range (contiguous stores coalesce). While the tracker's unnoted clock
    // holds still, they are the only writes repeats must account for.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> engine_stores_;
    bool engine_stores_reported_ = false, engine_stores_overflow_ = false;
    std::uint64_t quiet_clock_ = 0;
    void note_engine_store(std::uint64_t gpu_address, std::uint64_t size) override {
        if (!engine_stores_reported_ || !size) return;
        if (!engine_stores_.empty() && engine_stores_.back().first + engine_stores_.back().second == gpu_address) {
            engine_stores_.back().second += size;
            return;
        }
        if (engine_stores_.size() >= 256) {
            engine_stores_overflow_ = true;
            return;
        }
        engine_stores_.emplace_back(gpu_address, size);
    }
    void set_engine_stores_reported(bool reported) override {
        engine_stores_reported_ = reported;
        reset_quiet();
    }
    // The next repeat checks every stamp again (sets were rebuilt or dropped).
    void reset_quiet() noexcept {
        quiet_clock_ = 0;
        engine_stores_.clear();
        engine_stores_overflow_ = false;
    }
    // Whether an engine store reached what a stage buffer set read other than
    // its recorded uniform windows (see refresh_repeat).
    static bool quiet_store_rebuilds(const StageBuffers& stage,
                                     const std::pair<std::uint64_t, std::uint64_t>& store) noexcept {
        if (!stage.stage) return true;
        const auto& compiled = *stage.stage;
        const auto hit = [&](std::uint64_t address, std::uint64_t size) {
            return store.first < address + size && address < store.first + store.second;
        };
        for (const auto& binding : compiled.constant_buffers) {
            const auto& bound = stage.bindings[binding.guest_binding];
            if (bound.enabled && hit(bound.address, binding.minimum_size)) return true;
        }
        for (const auto& binding : compiled.constant_buffer_windows) {
            const auto& bound = stage.bindings[binding.guest_binding];
            if (!bound.enabled || !hit(bound.address, std::min<std::uint64_t>(bound.size, binding.window_bytes)))
                continue;
            const bool recorded = std::ranges::any_of(stage.windows, [&](const WindowRecord& window) {
                return window.slot == binding.guest_binding;
            });
            const bool descriptors = std::ranges::any_of(compiled.storage_buffers, [&](const auto& descriptor) {
                return descriptor.cbuf_slot == binding.guest_binding;
            });
            if (!recorded || descriptors) return true;
        }
        return false;
    }
    bool engine_stored(std::uint64_t address, std::uint64_t size) const noexcept {
        return std::ranges::any_of(engine_stores_, [&](const auto& store) {
            return store.first < address + size && address < store.first + store.second;
        });
    }
    // Tracing: which refresh_repeat check failed, by exit order.
    std::array<std::uint64_t, 24> refresh_failures_{};
    bool refresh_failed(std::size_t exit) {
        static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        if (!trace) return false;
        ++refresh_failures_[std::min<std::size_t>(exit, refresh_failures_.size() - 1)];
        std::uint64_t total = 0;
        for (const auto value : refresh_failures_) total += value;
        if (total % 20000) return false;
        armigo::diag::opt_in( "refresh_failures");
        for (std::size_t i = 0; i < refresh_failures_.size(); ++i)
            if (refresh_failures_[i]) armigo::diag::opt_in( " e%zu=%llu", i, (unsigned long long)refresh_failures_[i]);
        armigo::diag::newline();
        return false;
    }
    bool refresh_repeat(const RenderStop& stop, const DrawPlan& draw, std::uint64_t dirty,
                        ResourceMemory& memory, BufferCache& buffers, RepeatBindings& out) override {
        namespace groups = graphics::threed_dirty;
        auto* registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
        if (!registry) return refresh_failed(1);
        if (registry->has_deferred_host_store()) return refresh_failed(2);
        const auto stamp = registry->cpu_write_stamp();
        if (!stamp) return refresh_failed(3);
        const auto written = [&](const RetainedRange& range, std::uint64_t since) {
            return registry->cpu_written_since(range.bytes, since);
        };
        // Quiet: no guest store, registration change or unreported store
        // reached a tracked page since the last stamp-checked refresh, so only
        // the reported engine stores can have changed what the sets read.
        const auto unnoted = armigo::memory::WriteTracker::unnoted_clock();
        const bool quiet = engine_stores_reported_ && !engine_stores_overflow_ && quiet_clock_ &&
                           unnoted == quiet_clock_;
        std::array<std::vector<std::size_t>, 5> stale;
        std::array<bool, 5> rebind{};
        std::array<std::vector<Moved>, 5> moved;
        for (const std::size_t index : {0u, 4u}) {
            auto& textures = stage_textures_[index];
            const auto& stage = stage_buffers_[index];
            if (!textures.stamp || !stage.stamp) return refresh_failed(4);
            const bool textures_quiet = quiet && std::ranges::none_of(textures.gpu_reads, [&](const auto& read) {
                return engine_stored(read.first, read.second);
            });
            if (!textures_quiet) {
                if (std::ranges::any_of(textures.reads, [&](const auto& range) { return written(range, textures.stamp); }))
                    return refresh_failed(5);
                if (!std::ranges::all_of(textures.entries, [&](const PoolEntry& entry) {
                        return same_entry(entry, *registry, textures.stamp);
                    })) return refresh_failed(6);
                if (!same_handles(textures, memory, *registry)) return refresh_failed(7);
            }
            if (dirty & groups::bit(groups::ConstantBuffers0 + std::uint32_t(index))) {
                // Rebinding the stage's constant buffers keeps its textures
                // only while the handle slots keep their bindings.
                for (const auto& [slot, binding] : textures.handle_slots) {
                    const auto& current = stop.constant_buffers[index][slot];
                    if (current.address != binding.address || current.size != binding.size ||
                        current.enabled != binding.enabled) return refresh_failed(8);
                }
                if (!moved_windows(stop, index, memory, moved[index])) {
                    moved[index].clear();
                    rebind[index] = true;
                    continue;
                }
            }
            if (quiet) {
                // Engine stores the set's reads may have seen: a recorded
                // window they reach is taken again; directly bound constant
                // buffers, descriptor windows and unrecorded windows need a
                // full preparation; bytes the stage never reads are ignored.
                if (std::ranges::any_of(engine_stores_, [&](const auto& store) {
                        return quiet_store_rebuilds(stage, store);
                    })) return refresh_failed(13);
                for (std::size_t i = 0; i < stage.windows.size(); ++i)
                    if (engine_stored(stage.windows[i].address, stage.windows[i].copied)) stale[index].push_back(i);
                continue;
            }
            if (std::ranges::any_of(stage.direct, [&](const auto& range) { return written(range, stage.stamp); }))
                return refresh_failed(9);
            for (std::size_t i = 0; i < stage.windows.size(); ++i)
                if (written(stage.windows[i].range, stage.stamp)) stale[index].push_back(i);
        }
        try {
            for (const std::size_t index : {0u, 4u}) {
                auto& stage = stage_buffers_[index];
                const auto shader_stage = index == 0 ? shader_input::Stage::Vertex : shader_input::Stage::Fragment;
                if (rebind[index]) {
                    Bindings rebuilt;
                    std::vector<DirectBufferRequest> requests;
                    const auto compiled = stage.stage;
                    stage_buffers(stop, index, shader_stage, compiled, memory, buffers, requests, rebuilt,
                                  BindingReuse{dirty, stage.epoch, {}});
                    if (!requests.empty() || !rebuilt.dependencies.empty() || !stage_buffers_[index].stamp)
                        return refresh_failed(10);
                    out.changed.insert(out.changed.end(), rebuilt.buffers.begin(), rebuilt.buffers.end());
                    continue;
                }
                for (auto& change : moved[index]) {
                    auto& window = stage.windows[change.record];
                    window.address = stop.constant_buffers[index][window.slot].address;
                    window.range = change.range;
                    if (std::ranges::find(stale[index], change.record) == stale[index].end())
                        stale[index].push_back(change.record);
                }
                if (!moved[index].empty()) {
                    stage.bindings = stop.constant_buffers[index];
                    stage.gpu_reads.clear();
                    for (const auto& bound : stage.bindings)
                        if (bound.enabled) stage.gpu_reads.emplace_back(bound.address, bound.size);
                }
                for (const auto i : stale[index]) {
                    const auto& window = stage.windows[i];
                    auto resident = snapshot_window(memory, buffers, window.address, window.copied,
                                                    window.window_bytes, &window.range);
                    for (auto& bound : stage.buffers)
                        if (bound.stage == shader_stage && bound.index == window.index) bound.buffer = resident;
                    out.changed.push_back({shader_stage, window.index, std::move(resident)});
                }
                if (!stale[index].empty()) stage.stamp = stamp;
            }
            if (dirty & (groups::bit(groups::IndexBuffer) | groups::bit(groups::VertexStreams))) {
                Bindings streams;
                std::vector<DirectBufferRequest> direct;
                vertex_index_requests(draw, stream_groups(draw), memory, buffers, streams, direct);
                acquire_direct(memory, buffers, direct, streams);
                if (!streams.dependencies.empty() || !streams.buffers.empty()) return refresh_failed(11);
                out.vertex_buffers = std::move(streams.vertex_buffers);
                out.index_buffer = std::move(streams.index_buffer);
            }
        } catch (const RendererError&) {
            return refresh_failed(12);
        }
        last_bindings_.reset();
        // Everything the sets read is accounted for up to this point.
        if (!quiet && engine_stores_reported_) quiet_clock_ = unnoted;
        engine_stores_.clear();
        engine_stores_overflow_ = false;
        return true;
    }

    void drop_reuse() override {
        reset_quiet();
        cached_textures_.clear();
        cached_samplers_.clear();
        stage_textures_ = {};
        texture_sets_ = {};
        buffer_sets_ = {};
        stage_buffers_ = {};
        last_bindings_.reset();
    }

    // Handle words whose pages took a store since the set's stamp must keep
    // their values. They are peeked straight from host-addressable guest
    // memory: re-arming write protection there would only make the next
    // per-frame uniform store fault again.
    static bool same_handles(const StageTextures& cache, ResourceMemory& memory,
                             ResourceMemoryRegistry& registry) {
        for (std::size_t i = 0; i < cache.handle_ranges.size(); ++i) {
            const auto& bytes = cache.handle_ranges[i].bytes;
            if (registry.deferred_host_store(bytes)) return false;
            if (!registry.cpu_written_since(bytes, cache.stamp)) continue;
            std::uint32_t value = 0;
            const auto host = registry.backing_host_range(bytes.backing);
            if (host && bytes.offset <= host->second && host->second - bytes.offset >= sizeof value) {
                std::memcpy(&value, reinterpret_cast<const void*>(host->first + bytes.offset), sizeof value);
            } else {
                std::array<std::byte, 4> word{};
                if (!memory.address_space().read(cache.handle_addresses[i], word)) return false;
                std::memcpy(&value, word.data(), sizeof value);
            }
            if (value != cache.handle_values[i]) return false;
        }
        return true;
    }

    void stage_textures(const RenderStop& stop, std::size_t stage_index,
                        shader_input::Stage shader_stage,
                        const std::shared_ptr<const CompiledGraphicsStage>& stage,
                        ResourceMemory& memory, BufferCache& buffers, TextureCache& textures,
                        Bindings& result, const BindingReuse& reuse) {
        auto& cache = stage_textures_[stage_index];
        auto* registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
        // Switch to this stage's earlier set; the current one joins the list.
        auto& sets = texture_sets_[stage_index];
        switch_set(cache, sets, stage);
        const auto same_handle_slots = [&] {
            return std::ranges::all_of(cache.handle_slots, [&](const auto& entry) {
                const auto& current = stop.constant_buffers[stage_index][entry.first];
                return current.address == entry.second.address && current.size == entry.second.size &&
                       current.enabled == entry.second.enabled;
            });
        };
        // Outcome counts: hit, no cache, epoch, stage, dirty, attachment, store.
        static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        static std::array<std::uint64_t, 7> outcomes{};
        const auto outcome = [&](std::size_t index) {
            if (!trace) return;
            ++outcomes[index];
            if ((outcomes[0] + outcomes[1] + outcomes[2] + outcomes[3] + outcomes[4] + outcomes[5] +
                 outcomes[6]) % 20000) return;
            armigo::diag::opt_in( "stage_texture_reuse hit=%llu none=%llu epoch=%llu stage=%llu dirty=%llu "
                "attachment=%llu stored=%llu\n", (unsigned long long)outcomes[0], (unsigned long long)outcomes[1],
                (unsigned long long)outcomes[2], (unsigned long long)outcomes[3], (unsigned long long)outcomes[4],
                (unsigned long long)outcomes[5], (unsigned long long)outcomes[6]);
        };
        std::vector<BackingRange> resident_writes;
        std::size_t miss = 0;
        if (!registry || !reuse.epoch || !cache.stamp) miss = 1;
        else if (cache.stage != stage) miss = 3;
        else if (cache.epoch != reuse.epoch &&
                 !(reuse.log && reuse.log->clean_since(cache.epoch, cache.gpu_reads, cache.reads) &&
                   reuse.log->clean_since(cache.epoch, {}, cache.handle_ranges) &&
                   reuse.log->textures_clean_since(cache.epoch, cache.resident_reads, &resident_writes)))
            miss = 2;
        else if (cache.registers != texture_inputs(stop) || !same_handle_slots()) miss = 4;
        else {
            static std::array<std::uint64_t, 2> kinds{};
            const auto stored = [&](std::size_t kind) {
                miss = 6;
                // Which kind of read a store reached: handle value, other bytes.
                if (trace && ++kinds[kind] % 20000 == 0)
                    armigo::diag::opt_in( "stage_texture_store handle=%llu other=%llu\n",
                                 (unsigned long long)kinds[0], (unsigned long long)kinds[1]);
            };
            if (std::ranges::any_of(cache.reads, [&](const RetainedRange& range) {
                    return registry->cpu_written_since(range.bytes, cache.stamp);
                })) stored(1);
            else if (!std::ranges::all_of(cache.entries, [&](const PoolEntry& entry) {
                         return same_entry(entry, *registry, cache.stamp);
                     })) stored(1);
            else if (!same_handles(cache, memory, *registry)) stored(0);
        }
        outcome(miss);
        if (!miss) {
            refresh_texture_dependencies(memory, resident_writes, cache.dependencies);
            cache.epoch = reuse.epoch;
            result.argument_buffers.insert(result.argument_buffers.end(),
                cache.argument_buffers.begin(), cache.argument_buffers.end());
            result.samplers.insert(result.samplers.end(), cache.samplers.begin(), cache.samplers.end());
            result.textures.insert(result.textures.end(), cache.textures.begin(), cache.textures.end());
            result.accesses.insert(result.accesses.end(), cache.accesses.begin(), cache.accesses.end());
            append_dependencies(result.dependencies, cache.dependencies);
            result.custom_border_textures.insert(result.custom_border_textures.end(),
                cache.border_textures.begin(), cache.border_textures.end());
            if (cache.border_index) {
                auto support = take(buffers.upload(cache.border_bytes, 16));
                append_dependencies(result.dependencies, support.dependencies);
                result.buffers.push_back({shader_stage, *cache.border_index, std::move(support),
                                          BufferUse::Uniform, ResourceAccess::Read});
            }
            return;
        }
        // Another stage's valid set is kept for when that program returns.
        if (cache.stage != stage) keep_set(cache, sets);
        cache = {};
        const auto stamp = registry ? registry->cpu_write_stamp() : 0;
        const auto argument_buffers = result.argument_buffers.size(), samplers = result.samplers.size(),
                   bound = result.textures.size(), accesses = result.accesses.size(),
                   border_textures = result.custom_border_textures.size();
        std::vector<CompletionToken> dependencies;
        std::vector<std::uint64_t> handles;
        std::vector<std::uint8_t> handle_slot_ids;
        std::vector<std::byte> border_bytes;
        texture_reuse_ = &reuse;
        struct ClearReuse { const BindingReuse*& reuse; ~ClearReuse() { reuse = nullptr; } } clear_reuse{texture_reuse_};
        prepare_stage_textures(stop, stage_index, shader_stage, *stage, memory, buffers, textures,
                               result, &handles, &border_bytes, &dependencies, &handle_slot_ids);
        if (!stamp || !reuse.epoch) return;
        // Retain every guest range the preparation read: the handle words,
        // both descriptor pools and each sampled texture's bytes.
        StageTextures built;
        const auto read = [&](std::uint64_t address, std::uint64_t size) {
            auto retained = memory.retain(address, size, ResourceAccess::Read);
            if (auto* range = std::get_if<RetainedRange>(&retained)) {
                built.reads.push_back(std::move(*range));
                return true;
            }
            return false;
        };
        for (const auto address : handles) {
            auto retained = memory.retain(address, 4, ResourceAccess::Read);
            auto* range = std::get_if<RetainedRange>(&retained);
            std::array<std::byte, 4> word{};
            if (!range || !memory.address_space().read(address, word)) return;
            std::uint32_t value = 0;
            std::memcpy(&value, word.data(), sizeof value);
            built.handle_ranges.push_back(std::move(*range));
            built.handle_addresses.push_back(address);
            built.handle_values.push_back(value);
            built.gpu_reads.emplace_back(address, 4);
        }
        // Captured values are the ones the preparation used only while no
        // store reached a handle after the stamp.
        if (std::ranges::any_of(built.handle_ranges, [&](const RetainedRange& range) {
                return registry->cpu_written_since(range.bytes, stamp);
            })) return;
        // Descriptor reads: when every bound texture came through a handle,
        // only the TIC and TSC entries those handles index (low 20 bits and
        // high 12 bits); stores elsewhere in the pools then leave the set valid.
        const auto pool_base = [&](std::uint32_t pool) {
            return (std::uint64_t(stop.registers[pool]) << 32) | stop.registers[pool + 1];
        };
        const auto pool_entries = [&](std::uint32_t pool) { return std::uint64_t(stop.registers[pool + 2]) + 1; };
        if (!built.handle_values.empty() && built.handle_values.size() >= result.textures.size() - bound) {
            std::vector<std::uint64_t> entries;
            for (const auto value : built.handle_values) {
                for (const auto [pool, index] : {std::pair{0x55dU, std::uint64_t(value & 0xfffffu)},
                                                 std::pair{0x557U, std::uint64_t(value >> 20)}}) {
                    if (index >= pool_entries(pool)) continue;
                    const auto address = pool_base(pool) + index * 32;
                    if (std::ranges::find(entries, address) != entries.end()) continue;
                    entries.push_back(address);
                    PoolEntry entry;
                    if (capture_entry(memory, *registry, address, entry)) {
                        // The captured bytes are the ones preparation used only
                        // while no store reached the entry after the stamp.
                        if (registry->cpu_written_since(entry.range.bytes, stamp)) return;
                        built.entries.push_back(std::move(entry));
                    } else if (!read(address, 32)) {
                        return;
                    }
                    built.gpu_reads.emplace_back(address, 32);
                }
            }
        } else {
            for (const std::uint32_t pool : {0x55dU, 0x557U}) {
                const auto size = pool_entries(pool) * 32;
                if (!read(pool_base(pool), size)) return;
                built.gpu_reads.emplace_back(pool_base(pool), size);
            }
        }
        for (auto it = result.textures.begin() + std::ptrdiff_t(bound); it != result.textures.end(); ++it) {
            const auto& descriptor = it->texture.key.storage.descriptor;
            if (!it->texture.storage || !descriptor.address) continue;
            built.sampled.emplace_back(descriptor.address, descriptor.backing_extent);
            built.resident_reads.push_back(resident_read(it->texture));
            built.reads.push_back(RetainedRange{{}, it->texture.key.storage.bytes, {}});
        }
        std::ranges::sort(handle_slot_ids);
        handle_slot_ids.erase(std::unique(handle_slot_ids.begin(), handle_slot_ids.end()), handle_slot_ids.end());
        for (const auto slot : handle_slot_ids)
            built.handle_slots.emplace_back(slot, stop.constant_buffers[stage_index][slot]);
        merge_nearby(built.reads);
        built.stage = stage;
        built.epoch = reuse.epoch;
        built.stamp = stamp;
        built.registers = texture_inputs(stop);
        built.argument_buffers.assign(result.argument_buffers.begin() + std::ptrdiff_t(argument_buffers),
                                      result.argument_buffers.end());
        built.samplers.assign(result.samplers.begin() + std::ptrdiff_t(samplers), result.samplers.end());
        built.textures.assign(result.textures.begin() + std::ptrdiff_t(bound), result.textures.end());
        built.accesses.assign(result.accesses.begin() + std::ptrdiff_t(accesses), result.accesses.end());
        built.dependencies = std::move(dependencies);
        built.border_textures.assign(result.custom_border_textures.begin() + std::ptrdiff_t(border_textures),
                                     result.custom_border_textures.end());
        if (stage->custom_border) {
            built.border_index = stage->custom_border->buffer_index;
            built.border_bytes = std::move(border_bytes);
        }
        cache = std::move(built);
    }

    TextureDescriptorCache texture_descriptors_;

    // One resolved texture binding of a stage, reused while its handle word
    // keeps its address and value, its TIC/TSC entries and texture bytes take
    // no CPU store, and GPU writes preserve their resident allocation (as one
    // title keeps a texture binding per handle). Replays skip descriptor
    // resolution and the texture cache acquire.
    struct CachedTexture {
        std::uint32_t texture_index = 0, sampler_index = 0, handle_byte_offset = 0;
        std::uint64_t handle_address = 0;
        const std::uint32_t* handle_host = nullptr;
        RetainedRange handle_range;
        std::uint32_t handle_value = 0;
        std::array<std::uint32_t, 8> registers{};
        std::uint64_t stamp = 0, epoch = 0;
        std::vector<RetainedRange> reads;
        std::vector<PoolEntry> entries;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> gpu_reads;
        TextureBinding binding;
        std::array<EpochLog::ResidentRead, 1> resident_reads;
        std::optional<GpuAccessRequest> access;
    };
    struct CachedTextureKey {
        const CompiledGraphicsStage* stage;
        std::size_t stage_index, request;
        bool operator==(const CachedTextureKey&) const = default;
    };
    struct CachedTextureHash {
        std::size_t operator()(const CachedTextureKey& key) const noexcept {
            return std::hash<const void*>{}(key.stage) ^ (key.stage_index << 20) ^ (key.request << 24);
        }
    };
    // A binding position keeps the resolutions of its last few handle values:
    // materials that share a program alternate their handles per draw.
    struct CachedTextureVariants {
        std::array<CachedTexture, 4> entries;
        std::uint8_t next = 0;
    };
    std::unordered_map<CachedTextureKey, CachedTextureVariants, CachedTextureHash> cached_textures_;
    // Per sampler binding position: the resolved sampler with the handle word
    // and TSC entry it came from, both checked by value.
    struct CachedSampler {
        std::uint64_t handle_address = 0;
        const std::uint32_t* handle_host = nullptr;
        std::uint32_t handle_value = 0;
        RetainedRange handle_range;
        PoolEntry entry;
        std::array<std::uint32_t, 8> registers{};
        std::uint64_t epoch = 0;
        std::array<std::pair<std::uint64_t, std::uint64_t>, 2> gpu_reads{};
        SamplerDescriptor live;
        RetainedResource sampler;
    };
    // Four handle values per position, as for textures.
    struct CachedSamplerVariants {
        std::array<CachedSampler, 4> entries;
        std::uint8_t next = 0;
    };
    std::unordered_map<CachedTextureKey, CachedSamplerVariants, CachedTextureHash> cached_samplers_;
    // Outcome counts for tracing: hit, absent, address/registers, epoch,
    // deferred store, handle value, entry bytes, not remembered.
    std::array<std::uint64_t, 8> sampler_outcomes_{};
    void count_sampler(std::size_t outcome) {
        static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        if (!trace) return;
        ++sampler_outcomes_[outcome];
        std::uint64_t total = 0;
        for (const auto value : sampler_outcomes_) total += value;
        if (total % 20000) return;
        armigo::diag::opt_in( "binding_sampler_reuse hit=%llu absent=%llu fields=%llu epoch=%llu deferred=%llu "
            "handle=%llu entry=%llu unremembered=%llu\n", (unsigned long long)sampler_outcomes_[0],
            (unsigned long long)sampler_outcomes_[1], (unsigned long long)sampler_outcomes_[2],
            (unsigned long long)sampler_outcomes_[3], (unsigned long long)sampler_outcomes_[4],
            (unsigned long long)sampler_outcomes_[5], (unsigned long long)sampler_outcomes_[6],
            (unsigned long long)sampler_outcomes_[7]);
    }
    const CachedSampler* reuse_sampler(const CachedTextureKey& key, std::uint64_t handle_address,
                                       const RenderStop& stop, ResourceMemoryRegistry& registry) {
        const auto found = cached_samplers_.find(key);
        if (found == cached_samplers_.end()) return count_sampler(1), nullptr;
        const auto inputs = texture_inputs(stop);
        std::size_t reason = 1;
        for (auto& cached : found->second.entries) {
            if (!cached.handle_host) continue;
            // The epoch check comes first: a mapping change logs the remapped
            // ranges, so both host pages are still mapped when they are peeked.
            if (cached.handle_address != handle_address || cached.registers != inputs) {
                reason = std::max<std::size_t>(reason, 2);
                continue;
            }
            if (!(cached.epoch == texture_reuse_->epoch ||
                  (texture_reuse_->log && texture_reuse_->log->clean_since(cached.epoch, cached.gpu_reads)))) {
                reason = std::max<std::size_t>(reason, 3);
                continue;
            }
            if (registry.deferred_host_store(cached.handle_range.bytes) ||
                registry.deferred_host_store(cached.entry.range.bytes)) {
                reason = std::max<std::size_t>(reason, 4);
                continue;
            }
            std::uint32_t value = 0;
            std::memcpy(&value, cached.handle_host, sizeof value);
            if (value != cached.handle_value) {
                reason = std::max<std::size_t>(reason, 5);
                continue;
            }
            if (std::memcmp(cached.entry.host, cached.entry.bytes.data(), cached.entry.bytes.size()) != 0) {
                reason = std::max<std::size_t>(reason, 6);
                continue;
            }
            count_sampler(0);
            cached.epoch = texture_reuse_->epoch;
            return &cached;
        }
        return count_sampler(reason), nullptr;
    }
    void remember_sampler(const CachedTextureKey& key, std::uint64_t handle_address, const RenderStop& stop,
                          ResourceMemory& memory, ResourceMemoryRegistry& registry, std::uint64_t stamp,
                          const SamplerDescriptor& live, const RetainedResource& sampler) {
        CachedSampler cached;
        cached.handle_address = handle_address;
        struct Unremembered {
            Planner& planner;
            bool remembered = false;
            ~Unremembered() { if (!remembered) planner.count_sampler(7); }
        } unremembered{*this};
        auto retained = memory.retain(handle_address, 4, ResourceAccess::Read);
        auto* range = std::get_if<RetainedRange>(&retained);
        if (!range) return;
        const auto host = registry.backing_host_range(range->bytes.backing);
        if (!host || range->bytes.offset + 4 > host->second) return;
        cached.handle_range = *range;
        cached.handle_host = reinterpret_cast<const std::uint32_t*>(host->first + range->bytes.offset);
        std::memcpy(&cached.handle_value, cached.handle_host, sizeof cached.handle_value);
        const auto index = std::uint64_t(cached.handle_value >> 20);
        if (index > stop.registers[0x557 + 2]) return;
        const auto entry_address = ((std::uint64_t(stop.registers[0x557]) << 32) | stop.registers[0x558]) + index * 32;
        if (!capture_entry(memory, registry, entry_address, cached.entry)) return;
        // Only values the resolution itself read: nothing stored after the stamp.
        if (registry.cpu_written_since(cached.handle_range.bytes, stamp) ||
            registry.cpu_written_since(cached.entry.range.bytes, stamp)) return;
        cached.registers = texture_inputs(stop);
        cached.epoch = texture_reuse_->epoch;
        cached.gpu_reads = {std::pair{handle_address, std::uint64_t{4}}, std::pair{entry_address, std::uint64_t{32}}};
        cached.live = live;
        cached.sampler = sampler;
        if (cached_samplers_.size() >= 8192) cached_samplers_.clear();
        auto& variants = cached_samplers_[key];
        auto slot = std::ranges::find_if(variants.entries, [&](const CachedSampler& entry) {
            return entry.handle_host && entry.handle_address == cached.handle_address &&
                   entry.handle_value == cached.handle_value;
        });
        if (slot == variants.entries.end()) {
            slot = variants.entries.begin() + variants.next;
            variants.next = std::uint8_t((variants.next + 1) % variants.entries.size());
        }
        *slot = std::move(cached);
        unremembered.remembered = true;
    }
    // Set by stage_textures for the call in progress; per-binding reuse needs
    // its epoch and write log.
    const BindingReuse* texture_reuse_ = nullptr;

    void remember_texture(const CachedTextureKey& key, std::uint32_t texture_index, std::uint32_t sampler_index,
                          std::uint32_t handle_byte_offset, std::uint64_t handle_address, const RenderStop& stop,
                          ResourceMemory& memory, ResourceMemoryRegistry& registry, std::uint64_t stamp,
                          const RetainedRange* texture_bytes, const Bindings& result) {
        CachedTexture cached;
        cached.texture_index = texture_index;
        cached.sampler_index = sampler_index;
        cached.handle_byte_offset = handle_byte_offset;
        cached.handle_address = handle_address;
        cached.registers = texture_inputs(stop);
        cached.stamp = stamp;
        cached.epoch = texture_reuse_->epoch;
        const auto retain = [&](std::uint64_t address, std::uint64_t size) -> const RetainedRange* {
            auto retained = memory.retain(address, size, ResourceAccess::Read);
            auto* range = std::get_if<RetainedRange>(&retained);
            if (!range) return nullptr;
            cached.reads.push_back(std::move(*range));
            cached.gpu_reads.emplace_back(address, size);
            return &cached.reads.back();
        };
        const auto* handle_range = retain(handle_address, 4);
        if (!handle_range) return;
        cached.handle_range = *handle_range;
        if (registry.deferred_host_store(handle_range->bytes)) return;
        const auto host = registry.backing_host_range(handle_range->bytes.backing);
        if (!host || handle_range->bytes.offset + 4 > host->second) return;
        cached.handle_host = reinterpret_cast<const std::uint32_t*>(host->first + handle_range->bytes.offset);
        std::memcpy(&cached.handle_value, cached.handle_host, sizeof cached.handle_value);
        // Keep the handle lease separately for deferred-store checks. Ordinary
        // page stores still use the value comparison instead of invalidating.
        cached.reads.pop_back();
        cached.gpu_reads.pop_back();
        cached.gpu_reads.emplace_back(handle_address, 4);
        for (const auto [pool, index] : {std::pair{0x55dU, std::uint64_t(cached.handle_value & 0xfffffu)},
                                         std::pair{0x557U, std::uint64_t(cached.handle_value >> 20)}}) {
            if (index > stop.registers[pool + 2]) continue;
            const auto base = (std::uint64_t(stop.registers[pool]) << 32) | stop.registers[pool + 1];
            PoolEntry entry;
            if (capture_entry(memory, registry, base + index * 32, entry)) {
                cached.entries.push_back(std::move(entry));
                cached.gpu_reads.emplace_back(base + index * 32, 32);
            } else if (!retain(base + index * 32, 32)) {
                return;
            }
        }
        if (texture_bytes) {
            cached.reads.push_back(*texture_bytes);
            cached.resident_reads[0] = resident_read(result.textures.back().texture);
        }
        // Only values the resolution itself read: nothing stored after the stamp.
        std::uint32_t value = 0;
        std::memcpy(&value, cached.handle_host, sizeof value);
        if (value != cached.handle_value ||
            std::ranges::any_of(cached.reads, [&](const RetainedRange& range) {
                return registry.cpu_written_since(range.bytes, stamp);
            }) ||
            std::ranges::any_of(cached.entries, [&](const PoolEntry& entry) {
                return registry.cpu_written_since(entry.range.bytes, stamp);
            })) return;
        cached.binding = result.textures.back();
        if (texture_bytes && !result.accesses.empty()) cached.access = result.accesses.back();
        if (cached_textures_.size() >= 8192) cached_textures_.clear();
        auto& variants = cached_textures_[key];
        auto slot = std::ranges::find_if(variants.entries, [&](const CachedTexture& entry) {
            return entry.handle_host && entry.handle_address == cached.handle_address &&
                   entry.handle_value == cached.handle_value;
        });
        if (slot == variants.entries.end()) {
            slot = variants.entries.begin() + variants.next;
            variants.next = std::uint8_t((variants.next + 1) % variants.entries.size());
        }
        *slot = std::move(cached);
    }

    void prepare_stage_textures(const RenderStop& stop, std::size_t stage_index,
                       shader_input::Stage shader_stage, const CompiledGraphicsStage& stage,
                       ResourceMemory& memory, BufferCache& buffers, TextureCache& textures,
                       Bindings& result, std::vector<std::uint64_t>* handles = nullptr,
                       std::vector<std::byte>* border_bytes = nullptr,
                       std::vector<CompletionToken>* dependencies = nullptr,
                       std::vector<std::uint8_t>* handle_slots = nullptr) {
        // The guest address of a handle word, recorded for binding reuse.
        const auto handle = [&](std::optional<std::uint8_t> cbuf_slot, std::uint32_t offset) {
            if (!handles) return;
            const auto slot = cbuf_slot.value_or(std::uint8_t(stop.registers[0x982]));
            if (slot < stop.constant_buffers[stage_index].size()) {
                handles->push_back(stop.constant_buffers[stage_index][slot].address + offset);
                if (handle_slots) handle_slots->push_back(slot);
            }
        };
        struct TextureRequest {
            std::uint32_t handle_byte_offset, texture_index, sampler_index;
            std::optional<std::uint8_t> cbuf_slot;
            std::optional<DerivedTextureFact> fact;
            std::optional<std::uint32_t> argument_byte_offset;
            std::optional<std::uint32_t> resource_id;
        };
        std::vector<TextureRequest> requests;
        std::map<std::uint32_t, SamplerDescriptor> live_samplers;
        struct alignas(16) BorderRecord {
            std::uint64_t coverage = 0, reserved = 0;
            std::array<float, 4> color{};
            std::array<std::uint32_t, 4> swizzle{};
            std::uint32_t enabled = 0;
            std::array<std::uint32_t, 3> padding{};
        };
        static_assert(sizeof(BorderRecord) == 64 && offsetof(BorderRecord, color) == 16 &&
                      offsetof(BorderRecord, swizzle) == 32 && offsetof(BorderRecord, enabled) == 48);
        std::vector<BorderRecord> border_records(stage.custom_border ? stage.custom_border->pairs.size() : 0);
        if (stage.argument_buffer)
            result.argument_buffers.push_back({shader_stage, *stage.argument_buffer});
        if (stage.resources.empty()) {
            for (const auto& binding : stage.textures) {
                const auto* fact = texture_fact(stage, binding.handle_byte_offset);
                requests.push_back({binding.handle_byte_offset, binding.texture_index,
                    binding.sampler_index, std::nullopt,
                    fact ? std::optional(*fact) : std::nullopt, std::nullopt, std::nullopt});
            }
        } else for (const auto& resource : stage.resources) {
            for (std::uint32_t element = 0; element != resource.array_length; ++element) {
                const auto offset = resource.handle_byte_offset + element * 4;
                const auto& location = resource.location;
                const bool indirect = location.kind ==
                    CompiledShaderResourceLocationKind::ArgumentBufferMember;
                const auto argument_offset = indirect
                    ? std::optional(location.byte_offset + element * 8) : std::nullopt;
                if (resource.kind == CompiledShaderResourceKind::Sampler) {
                    handle(resource.cbuf_slot, offset);
                    auto* sampler_registry = texture_reuse_ && texture_reuse_->epoch
                        ? dynamic_cast<ResourceMemoryRegistry*>(&memory) : nullptr;
                    const CachedTextureKey sampler_key{&stage, stage_index,
                        std::size_t(&resource - stage.resources.data()) * 4096 + element};
                    const auto sampler_slot = resource.cbuf_slot;
                    const auto sampler_handle = sampler_slot < stop.constant_buffers[stage_index].size()
                        ? stop.constant_buffers[stage_index][sampler_slot].address + offset : 0;
                    const auto sampler_stamp = sampler_registry ? sampler_registry->cpu_write_stamp() : 0;
                    if (sampler_registry && sampler_handle && sampler_stamp) {
                        if (const auto* cached = reuse_sampler(sampler_key, sampler_handle, stop, *sampler_registry)) {
                            live_samplers.emplace(resource.resource_id, cached->live);
                            result.samplers.push_back({shader_stage, location.first + element,
                                                       cached->sampler, argument_offset});
                            continue;
                        }
                    }
                    const auto resolved = resolve_sampler_binding(stop, stage_index,
                        resource.cbuf_slot, offset, memory, texture_descriptors_);
                    if (const auto* error = std::get_if<TextureError>(&resolved))
                        reject(error->code == TextureErrorCode::MemoryRead
                                   ? CommandStatus::MemoryFault : CommandStatus::Unsupported,
                               std::string("sampler binding rejection: ") + texture_error_detail(*error));
                    const auto& live = std::get<SamplerDescriptor>(resolved);
                    require(live.depth_compare == resource.comparison,
                            "sampler comparison expectation does not match the live descriptor");
                    // Metal's mapped sampler contract requires normalized coordinates.
                    require(resource.normalized_coordinates,
                            "unnormalized sampler resources are unsupported");
                    auto native_sampler = live;
                    if (live.srgb_border_conversion && std::ranges::any_of(live.wrap, [](auto wrap) {
                        return wrap == SamplerWrap::ClampToBorder || wrap == SamplerWrap::MirrorClampToBorder;
                    })) {
                        require(std::ranges::any_of(stage.resources, [&](const auto& texture) {
                            return texture.kind == CompiledShaderResourceKind::SampledTexture &&
                                texture.cbuf_slot == resource.cbuf_slot &&
                                texture.handle_byte_offset == resource.handle_byte_offset &&
                                texture.array_length == resource.array_length;
                        }), "sRGB border conversion requires paired texture provenance");
                        const auto texture = resolve_texture_binding_descriptors(stop, stage_index, offset,
                            memory, false, resource.cbuf_slot, &texture_descriptors_);
                        if (const auto* error = std::get_if<TextureError>(&texture))
                            reject(CommandStatus::Unsupported, error->message());
                        if (const auto* error = std::get_if<RendererError>(&texture)) throw *error;
                        native_sampler.border_color = metal_custom_border_color(live,
                            std::get<RetainedTextureBinding>(texture).binding.texture.srgb);
                        native_sampler.srgb_border_conversion = false;
                    }
                    const auto properties = metal_sampler_properties(native_sampler);
                    if (const auto* error = std::get_if<TextureError>(&properties))
                        reject(CommandStatus::Unsupported, error->message());
                    if (std::get<MetalSamplerProperties>(properties).custom_border) {
                        require(stage.custom_border && std::ranges::any_of(stage.custom_border->pairs,
                            [&](const auto& pair) { return pair.sampler_resource == resource.resource_id; }),
                            "custom border requires a coverage-v1 shader sampling pair");
                    }
                    live_samplers.emplace(resource.resource_id, live);
                    auto sampler = take(states_->sampler(native_sampler));
                    require(sampler != nullptr, "sampler cache returned no retained object");
                    // A border converted for an sRGB paired texture depends on that
                    // texture: not reused.
                    const bool texture_border = live.srgb_border_conversion &&
                        std::ranges::any_of(live.wrap, [](auto wrap) {
                            return wrap == SamplerWrap::ClampToBorder || wrap == SamplerWrap::MirrorClampToBorder;
                        });
                    if (sampler_registry && sampler_handle && sampler_stamp && !texture_border)
                        remember_sampler(sampler_key, sampler_handle, stop, memory, *sampler_registry,
                                         sampler_stamp, live, sampler);
                    result.samplers.push_back({shader_stage, location.first + element,
                                               std::move(sampler), argument_offset});
                } else {
                    requests.push_back({offset, location.first + element,
                        gm::TextureBinding::no_sampler, resource.cbuf_slot,
                        resource_texture_fact(resource, offset), argument_offset, resource.resource_id});
                }
            }
        }
        auto* texture_registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
        // Textures paired with a custom-border sampler rebuild their border
        // records and are never reused per binding; the stage's others are.
        const bool binding_reuse = texture_reuse_ && texture_reuse_->epoch && texture_registry;
        for (const auto& binding : requests) {
            const auto* fact = binding.fact ? &*binding.fact : nullptr;
            handle(binding.cbuf_slot, binding.handle_byte_offset);
            const CachedTextureKey cache_key{&stage, stage_index, std::size_t(&binding - requests.data())};
            const auto handle_slot = binding.cbuf_slot.value_or(std::uint8_t(stop.registers[0x982]));
            const auto handle_address = handle_slot < stop.constant_buffers[stage_index].size()
                ? stop.constant_buffers[stage_index][handle_slot].address + binding.handle_byte_offset : 0;
            const auto binding_stamp = binding_reuse ? texture_registry->cpu_write_stamp() : 0;
            const bool bordered = stage.custom_border && binding.resource_id &&
                std::ranges::any_of(stage.custom_border->pairs, [&](const auto& pair) {
                    return pair.texture_resource == *binding.resource_id;
                });
            if (binding_reuse && !bordered && binding_stamp && handle_address) {
                // Outcome counts: hit, absent, fields, epoch, handle, stored.
                static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
                static std::array<std::uint64_t, 6> binding_outcomes{};
                std::size_t reason = 1;
                const auto count = [&] {
                    if (!trace) return;
                    ++binding_outcomes[reason];
                    std::uint64_t total = 0;
                    for (const auto value : binding_outcomes) total += value;
                    if (total % 50000) return;
                    armigo::diag::opt_in( "binding_texture_reuse hit=%llu absent=%llu fields=%llu epoch=%llu "
                        "handle=%llu stored=%llu\n", (unsigned long long)binding_outcomes[0],
                        (unsigned long long)binding_outcomes[1], (unsigned long long)binding_outcomes[2],
                        (unsigned long long)binding_outcomes[3], (unsigned long long)binding_outcomes[4],
                        (unsigned long long)binding_outcomes[5]);
                };
                struct Count { const decltype(count)& count; ~Count() { count(); } } counted{count};
                if (const auto found = cached_textures_.find(cache_key); found != cached_textures_.end()) {
                    const auto inputs = texture_inputs(stop);
                    CachedTexture* hit = nullptr;
                    std::vector<BackingRange> resident_writes;
                    for (auto& cached : found->second.entries) {
                        if (trace && cached.handle_host) {
                            std::size_t why = 0;
                            if (cached.handle_address != handle_address ||
                                cached.texture_index != binding.texture_index ||
                                cached.sampler_index != binding.sampler_index ||
                                cached.handle_byte_offset != binding.handle_byte_offset || cached.registers != inputs)
                                why = 2;
                            else if (cached.epoch != texture_reuse_->epoch &&
                                     !(texture_reuse_->log && texture_reuse_->log->clean_since(cached.epoch,
                                           cached.gpu_reads, cached.reads) &&
                                       texture_reuse_->log->textures_clean_since(cached.epoch, cached.resident_reads))) {
                                why = 3;
                                // Which part failed: unknown/remap, GPU ranges, backing ranges, resident reads.
                                static std::array<std::uint64_t, 4> parts{};
                                const auto* log = texture_reuse_->log;
                                const std::size_t part = !log || log->last_unknown >= cached.epoch ? 0
                                    : !log->clean_since(cached.epoch, cached.gpu_reads) ? 1
                                    : !log->clean_since(cached.epoch, {}, cached.reads) ? 2 : 3;
                                if (++parts[part] % 5000 == 0)
                                    armigo::diag::opt_in( "binding_texture_epoch unknown=%llu gpu=%llu backing=%llu "
                                        "resident=%llu\n", (unsigned long long)parts[0], (unsigned long long)parts[1],
                                        (unsigned long long)parts[2], (unsigned long long)parts[3]);
                            }
                            else {
                                std::uint32_t value = 0;
                                std::memcpy(&value, cached.handle_host, sizeof value);
                                if (value != cached.handle_value) why = 4;
                                else if (std::ranges::any_of(cached.reads, [&](const RetainedRange& range) {
                                             return texture_registry->cpu_written_since(range.bytes, cached.stamp);
                                         })) why = 5;
                            }
                            if (why == 0) reason = 0;
                            else if (reason != 0 && (reason == 1 || why > reason)) reason = why;
                        }
                        // The epoch check comes first: a mapping change logs
                        // unknown writes, so the handle's host page is still
                        // mapped when it is peeked.
                        const auto peek = [&] {
                            std::uint32_t value = 0;
                            std::memcpy(&value, cached.handle_host, sizeof value);
                            return value;
                        };
                        resident_writes.clear();
                        if (cached.handle_host && cached.handle_address == handle_address &&
                            !texture_registry->deferred_host_store(cached.handle_range.bytes) &&
                            cached.texture_index == binding.texture_index &&
                            cached.sampler_index == binding.sampler_index &&
                            cached.handle_byte_offset == binding.handle_byte_offset &&
                            cached.registers == inputs &&
                            (cached.epoch == texture_reuse_->epoch ||
                             (texture_reuse_->log && texture_reuse_->log->clean_since(cached.epoch,
                                                                                      cached.gpu_reads,
                                                                                      cached.reads) &&
                              texture_reuse_->log->textures_clean_since(cached.epoch, cached.resident_reads,
                                                                        &resident_writes))) &&
                            peek() == cached.handle_value &&
                            std::ranges::none_of(cached.reads, [&](const RetainedRange& range) {
                                return texture_registry->cpu_written_since(range.bytes, cached.stamp);
                            }) &&
                            std::ranges::all_of(cached.entries, [&](const PoolEntry& entry) {
                                return same_entry(entry, *texture_registry, cached.stamp);
                            })) {
                            hit = &cached;
                            break;
                        }
                    }
                    if (hit) {
                        refresh_texture_dependencies(memory, resident_writes, hit->binding.texture.dependencies);
                        append_dependencies(result.dependencies, hit->binding.texture.dependencies);
                        if (dependencies) append_dependencies(*dependencies, hit->binding.texture.dependencies);
                        hit->epoch = texture_reuse_->epoch;
                        result.textures.push_back(hit->binding);
                        if (hit->access) result.accesses.push_back(*hit->access);
                        continue;
                    }
                }
            }
            auto resolved = resolve_texture_binding_descriptors(
                stop, stage_index, binding.handle_byte_offset,
                memory, binding.sampler_index != gm::TextureBinding::no_sampler,
                binding.cbuf_slot, &texture_descriptors_);
            if (const auto* error = std::get_if<TextureError>(&resolved)) {
                const auto prefix = fact
                    ? "texture_descriptor fact mismatch: live descriptor rejection: "
                    : "texture binding rejection: ";
                reject(error->code == TextureErrorCode::MemoryRead
                           ? CommandStatus::MemoryFault
                           : CommandStatus::Unsupported,
                       std::string(prefix) + texture_error_detail(*error));
            }
            if (const auto* error = std::get_if<RendererError>(&resolved)) throw *error;
            const auto& validated = std::get<RetainedTextureBinding>(resolved);
            graphics::ResolvedTextureBinding depth_tic;
            const auto& live = depth_alias_binding(validated.binding, fact, depth_tic);
            std::optional<TextureTargetPlan> target;
            if (fact) target = verify_texture_fact(*fact, live,
                binding.sampler_index != gm::TextureBinding::no_sampler);
            const auto info = texture_format_info(live.texture.format);
            std::optional<std::size_t> border_index;
            const SamplerDescriptor* border_sampler = nullptr;
            bool custom_border = false;
            if (stage.custom_border && binding.resource_id) {
                for (std::size_t i = 0; i < stage.custom_border->pairs.size(); ++i) {
                    const auto& pair = stage.custom_border->pairs[i];
                    if (pair.texture_resource != *binding.resource_id) continue;
                    border_index = i;
                    border_sampler = &live_samplers.at(pair.sampler_resource);
                    custom_border = std::get<MetalSamplerProperties>(
                        metal_sampler_properties(*border_sampler, live.texture.srgb)).custom_border;
                }
            }
            if (custom_border) {
                require(!live.is_null && live.texture.type == TextureType::Texture2D &&
                        live.texture.samples == 1 && live.texture.normalized_coordinates &&
                        !info.depth && float_sampled(live.texture.format),
                        "custom border supports normalized non-depth floating 2D textures only");
                const auto component = texture_component_format(live.texture.format);
                const auto color = metal_custom_border_color(*border_sampler, live.texture.srgb);
                float minimum = -std::numeric_limits<float>::max();
                float maximum = std::numeric_limits<float>::max();
                if (component == TextureComponentFormat::unorm) { minimum = 0.0f; maximum = 1.0f; }
                if (component == TextureComponentFormat::snorm) { minimum = -1.0f; maximum = 1.0f; }
                switch (live.texture.format) {
                case TextureFormat::R16Float: case TextureFormat::R16G16Float:
                case TextureFormat::R16G16B16A16Float: case TextureFormat::Bc6HSfloat:
                    minimum = -65504.0f; maximum = 65504.0f; break;
                case TextureFormat::Bc6HUfloat:
                    minimum = 0.0f; maximum = 65504.0f; break;
                case TextureFormat::R11G11B10Float:
                    minimum = 0.0f; maximum = 65024.0f; break;
                case TextureFormat::R9G9B9E5Float:
                    minimum = 0.0f; maximum = 65408.0f; break;
                default: break;
                }
                for (unsigned c = 0; c < info.components; ++c) {
                    const auto limit = live.texture.format == TextureFormat::R11G11B10Float && c == 2
                        ? 64512.0f : maximum;
                    require(color[c] >= minimum && color[c] <= limit,
                            "custom border outside the texture component range is unsupported");
                }
            }
            if (!fact) {
                require(live.texture.type == TextureType::Texture2D &&
                            live.texture.samples == 1 && live.texture.normalized_coordinates &&
                            binding.sampler_index != gm::TextureBinding::no_sampler,
                        "legacy binding planner samples plain single-sample 2D textures only");
                require(live.is_null ? !live.sampler.depth_compare :
                            ((info.depth && live.sampler.depth_compare) ||
                             (!info.depth && !live.sampler.depth_compare)),
                        "legacy comparison sampling and live texture format disagree");
                require(live.is_null || info.depth || float_sampled(live.texture.format),
                        "legacy integer texture sampling is unsupported by the admitted shader type");
            }

            const auto& retained = validated.backing;
            TextureStorageKey storage{retained.bytes, live.texture};
            TextureViewKey view{storage};
            view.aspect = info.depth ? TextureAspect::Depth : TextureAspect::Color;
            view.level_count = live.texture.levels;
            view.layer_count = live.texture.layers;
            view.swizzle = live.texture.swizzle;
            if (custom_border)
                view.swizzle = {TextureSwizzle::Red, TextureSwizzle::Green,
                                TextureSwizzle::Blue, TextureSwizzle::Alpha};
            if (target && !target->exact) {
                // A same-dimension arrayness difference uses a layer-zero host
                // view in the shader's target: one layer, or
                // six slices when the view target is a cube array.
                view.view_target = target->view_target;
                view.first_layer = target->first_layer;
                view.layer_count = target->layer_count;
            }
            ResidentTexture texture;
            if (live.is_null) {
                require(device_ != nullptr, "null texture binding requires a native device context");
                require(live.texture.samples == 1, "multisample null textures are unsupported");
                const bool depth = null_texture_depth(fact);
                const auto scalar = null_texture_scalar(fact);
                // A null TIC has no storage target of its own: the shader's
                // reflected declaration types the dummy resource.
                const auto declared = fact
                    ? texture_fact_target(fact->dimension, fact->array)
                    : std::optional{live.texture.type};
                require(declared.has_value(),
                        "the reflected 3D-array target cannot type a null texture");
                auto descriptor = live.texture;
                descriptor.type = *declared;
                const auto key = std::tuple{descriptor.type, scalar, depth};
                auto found = null_textures_.find(key);
                if (found == null_textures_.end()) {
                    auto created = metal_null_texture(device_->device(), device_->queue(), descriptor, scalar, depth);
                    if (const auto* error = std::get_if<CommandResult>(&created)) throw RendererError(*error);
                    found = null_textures_.emplace(key,
                        std::make_shared<NullTexture>(std::get<id<MTLTexture>>(created))).first;
                }
                view.aspect = depth ? TextureAspect::Depth : TextureAspect::Color;
                texture = {view, {}, found->second, {}};
            } else {
                auto acquired = textures.acquire({retained, view, ResourceAccess::Read,
                                                  graphics::TextureAliasScope::Sampler});
                if (std::holds_alternative<RendererError>(acquired)) {
                    // Acquire first so compatible GPU authority remains resident.
                    // An incompatible or mixed authority cannot be uploaded until
                    // its producer completes and the registry resolves its bytes.
                    const auto state = take(memory.inspect(retained.bytes));
                    if (std::ranges::any_of(state, [](const auto& interval) {
                            return interval.gpu_dirty;
                        })) {
                        if (const auto status = textures.resolve(
                                view, ResolveReason::IncompatibleAlias);
                            const auto* error = std::get_if<RendererError>(&status)) throw *error;
                        acquired = textures.acquire({retained, view, ResourceAccess::Read,
                                                     graphics::TextureAliasScope::Sampler});
                    }
                }
                texture = take(std::move(acquired));
                result.accesses.push_back({retained, ResourceAccess::Read});
            }
            graphics::RetainedResource sampler;
            if (binding.sampler_index != gm::TextureBinding::no_sampler) {
                const auto mapped = metal_sampler_properties(live.sampler, live.texture.srgb);
                if (const auto* properties = std::get_if<MetalSamplerProperties>(&mapped))
                    require(!properties->custom_border,
                            "legacy shader sampling has no custom-border support binding");
                auto native_sampler = live.sampler;
                if (std::ranges::any_of(live.sampler.wrap, [](auto wrap) {
                    return wrap == SamplerWrap::ClampToBorder || wrap == SamplerWrap::MirrorClampToBorder;
                })) native_sampler.border_color = metal_custom_border_color(live.sampler, live.texture.srgb);
                native_sampler.srgb_border_conversion = false;
                sampler = take(states_->sampler(native_sampler));
                require(sampler != nullptr, "sampler cache returned no retained object");
            }
            append_dependencies(result.dependencies, texture.dependencies);
            if (dependencies) append_dependencies(*dependencies, texture.dependencies);
            if (border_index) {
                auto& record = border_records[*border_index];
                const auto coverage = border_coverage(custom_border ? live.texture.width : 1,
                    custom_border ? live.texture.height : 1, custom_border ? live.texture.levels : 1);
                const auto native = std::dynamic_pointer_cast<const TextureResource>(coverage);
                const auto identifier = native->texture().gpuResourceID;
                static_assert(sizeof(identifier) == sizeof(record.coverage));
                std::memcpy(&record.coverage, &identifier, sizeof(identifier));
                record.enabled = custom_border;
                record.color = metal_custom_border_color(*border_sampler, live.texture.srgb);
                // Missing format channels retain the native zero/one defaults,
                // including alpha one. Only stored components receive border.
                for (unsigned c = info.components; c < 4; ++c) record.color[c] = 0.0f;
                for (unsigned c = 0; c < 4; ++c) {
                    const auto swizzle = live.texture.swizzle[c];
                    record.swizzle[c] = swizzle == TextureSwizzle::Zero ? 4u :
                        swizzle == TextureSwizzle::OneFloat || swizzle == TextureSwizzle::OneSInt ? 5u :
                        static_cast<unsigned>(swizzle) - static_cast<unsigned>(TextureSwizzle::Red);
                }
                result.custom_border_textures.push_back(coverage);
            }
            result.textures.push_back(
                {shader_stage, binding.texture_index, binding.sampler_index,
                 std::move(texture), std::move(sampler), binding.argument_byte_offset});
            if (binding_reuse && binding_stamp && handle_address && !border_index)
                remember_texture(cache_key, binding.texture_index, binding.sampler_index, binding.handle_byte_offset,
                                 handle_address, stop, memory, *texture_registry, binding_stamp,
                                 live.is_null ? nullptr : &retained, result);
        }
        if (stage.custom_border) {
            if (border_bytes) {
                const auto bytes = std::as_bytes(std::span(border_records));
                border_bytes->assign(bytes.begin(), bytes.end());
            }
            auto support = take(buffers.upload(std::as_bytes(std::span(border_records)), 16));
            append_dependencies(result.dependencies, support.dependencies);
            result.buffers.push_back({shader_stage, stage.custom_border->buffer_index,
                                      std::move(support), BufferUse::Uniform, ResourceAccess::Read});
        }
    }

    RetainedResource border_coverage(std::uint32_t width, std::uint32_t height,
                                     std::uint32_t levels) {
        require(device_ != nullptr, "custom border support requires a native device context");
        const auto key = std::tuple{width, height, levels};
        if (const auto found = border_coverage_.find(key); found != border_coverage_.end())
            return found->second;
        require(width && height && levels && levels <= 32,
                "custom border coverage dimensions are invalid");
        std::uint64_t bytes = 0;
        for (unsigned mip = 0; mip < levels; ++mip)
            bytes += std::uint64_t(std::max(1u, width >> mip)) * std::max(1u, height >> mip);
        require(width && height && levels && levels <= 32 && bytes <= staging_limit,
                "custom border coverage exceeds the host staging budget", CommandStatus::BudgetExceeded);
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
            width:width height:height mipmapped:levels > 1];
        descriptor.mipmapLevelCount = levels;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> texture = [device_->device() newTextureWithDescriptor:descriptor];
        require(texture != nil, "custom border coverage allocation failed", CommandStatus::MemoryFault);
        const auto allocation_bytes = std::uint64_t(texture.allocatedSize);
        require(allocation_bytes <= staging_limit,
                "custom border coverage allocation exceeds the host budget", CommandStatus::BudgetExceeded);
        std::vector<std::uint8_t> ones(std::uint64_t(width) * height, 255);
        for (unsigned mip = 0; mip < levels; ++mip) {
            const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
            [texture replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:mip
                         withBytes:ones.data() bytesPerRow:w];
        }
        // Eviction releases cache ownership only; in-flight bindings retain
        // their images until completion, independently of this bounded cache.
        if (allocation_bytes > staging_limit - border_coverage_bytes_) {
            border_coverage_.clear();
            border_coverage_bytes_ = 0;
        }
        auto resource = std::make_shared<NullTexture>(texture);
        border_coverage_.emplace(key, resource);
        border_coverage_bytes_ += allocation_bytes;
        return resource;
    }

    // Zero-copy guest backings use one newBufferWithBytesNoCopy buffer per
    // backing over its page-rounded host
    // range, created on first use. Bindings are buffer plus offset. Entries
    // for retired backings are dropped; encoded slices keep their buffer.
    // ARMIGO_GUEST_POOLS=0 keeps every buffer on the resident-copy path.
    // One storage identity per backing: every binding of the backing shares
    // it (accesses are matched to bindings by storage), covering the whole
    // backing, and binds at its own offset into the pool buffer.
    struct GuestPool {
        std::shared_ptr<const GuestPoolSlice> slice;
    };
    struct GuestBinding {
        std::shared_ptr<const GuestPoolSlice> slice;
        std::uint64_t offset = 0;
    };
    std::optional<GuestBinding> guest_pool_slice(ResourceMemory& memory, const RetainedRange& retained) {
        static const bool enabled = [] {
            const char* value = std::getenv("ARMIGO_GUEST_POOLS");
            return !(value && std::string_view{value} == "0");
        }();
        auto* registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
        if (!enabled || !device_ || !registry) return std::nullopt;
        const auto& backing = retained.bytes.backing;
        const auto host = registry->backing_host_range(backing);
        if (!host || !host->first || !host->second) return std::nullopt;
        const auto key = std::tuple{backing.owner, backing.allocation, backing.generation};
        auto found = guest_pools_.find(key);
        if (found == guest_pools_.end()) {
            if (guest_pools_.size() >= guest_pools_swept_ + 256) {
                std::erase_if(guest_pools_, [&](const auto& entry) {
                    const auto& [owner, allocation, generation] = entry.first;
                    return !registry->has_backing({owner, allocation, generation});
                });
                guest_pools_swept_ = guest_pools_.size();
            }
            const std::uint64_t page = static_cast<std::uint64_t>(getpagesize());
            const auto base = host->first & ~(page - 1);
            const auto end = (host->first + host->second + page - 1) & ~(page - 1);
            id<MTLBuffer> buffer = [device_->device() newBufferWithBytesNoCopy:reinterpret_cast<void*>(base)
                length:end - base options:MTLResourceStorageModeShared deallocator:nil];
            if (!buffer) return std::nullopt;
            auto slice = std::make_shared<GuestPoolSlice>();
            slice->buffer = buffer;
            slice->offset = host->first - base;
            slice->bytes = {backing, 0, host->second};
            slice->host = reinterpret_cast<const std::byte*>(host->first);
            found = guest_pools_.emplace(key, GuestPool{std::move(slice)}).first;
        }
        const auto& slice = found->second.slice;
        return GuestBinding{slice, slice->offset + retained.bytes.offset};
    }

    // Constant-buffer window snapshots keyed by backing bytes and window size,
    // valid while the interval versions match and no CPU store followed stamp.
    using WindowKey = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                 std::uint64_t, std::uint32_t>;
    struct WindowInterval {
        std::uint64_t offset = 0, size = 0;
        ResourceVersion version;
        bool operator==(const WindowInterval&) const = default;
    };
    struct WindowSnapshot {
        std::uint64_t stamp = 0, serial = 0;
        std::vector<WindowInterval> state;
        std::vector<std::byte> bytes;
    };
    static constexpr std::size_t window_budget = 32u << 20;
    std::map<WindowKey, WindowSnapshot> windows_;
    std::uint64_t window_serial_ = 0;
    // Per stage and binding slot, the last window upload and the snapshot it
    // holds; at most one ring slice per slot stays pinned.
    struct LastUpload {
        std::uint64_t serial = 0;
        ResidentBuffer resident;
    };
    std::array<std::array<std::optional<LastUpload>, 32>, 5> last_uploads_;
    std::size_t window_bytes_ = 0;

    std::shared_ptr<PipelineStateCache> states_;
    std::shared_ptr<DeviceContext> device_;
    std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>, GuestPool> guest_pools_;
    std::size_t guest_pools_swept_ = 0;
    std::map<std::tuple<TextureType, TextureNumericClass, bool>, RetainedResource> null_textures_;
    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, RetainedResource> border_coverage_;
    std::uint64_t border_coverage_bytes_ = 0;
};

} // namespace

RendererStatus populate_vertex_layout(PipelineKey& key, const DrawPlan& draw) {
    try {
        const auto groups = stream_groups(draw);
        const auto constant = constant_stream(draw);
        std::vector<VertexAttributeKey> attributes;
        attributes.reserve(draw.vertex.attributes.size());
        for (const auto& attribute : draw.vertex.attributes) {
            attributes.push_back({attribute.location, 18 + attribute_stream(attribute, constant),
                                  attribute.offset, attribute.format.format});
        }
        std::sort(attributes.begin(), attributes.end(),
                  [](const VertexAttributeKey& left,
                     const VertexAttributeKey& right) {
                      return left.location < right.location;
                  });
        std::vector<VertexBufferLayoutKey> buffers;
        buffers.reserve(groups.size());
        for (const auto& group : groups) {
            const bool expanded = !group.constant_attribute &&
                                  needs_instance_expansion(group, draw);
            const auto step = group.constant_attribute
                                  ? VertexStep::Constant
                                  : expanded ? VertexStep::PerInstance
                                             : group.step->step;
            const auto stride = group.constant_attribute
                                    ? 0u
                                    : expanded ? std::max(group.plan->stride,
                                                          group.reached)
                                               : group.plan->stride;
            const auto rate = step == VertexStep::PerInstance
                                  ? expanded ? 1u : group.step->rate
                                  : 1u;
            buffers.push_back({18 + group.stream, stride, step, rate});
        }
        key.attributes = std::move(attributes);
        key.buffers = std::move(buffers);
        return renderer_success();
    } catch (const RendererError& error) {
        return error;
    } catch (const std::bad_alloc&) {
        return RendererError{CommandResult{
            CommandStatus::BudgetExceeded,
            "vertex pipeline layout allocation failed"}};
    } catch (const std::length_error&) {
        return RendererError{CommandResult{
            CommandStatus::BudgetExceeded,
            "vertex pipeline layout exceeds a supported container extent"}};
    }
}

RendererResult<std::shared_ptr<BindingPlanner>>
create_binding_planner(std::shared_ptr<PipelineStateCache> states,
                       std::shared_ptr<DeviceContext> device) {
    if (!states || !states->context().device)
        return RendererError{CommandResult{
            CommandStatus::Unsupported,
            "binding planner requires a valid pipeline state cache"}};
    if (device && device->identity().device != states->context().device)
        return RendererError{CommandResult{
            CommandStatus::Unsupported,
            "binding planner device and pipeline cache belong to different domains"}};
    try {
        return std::shared_ptr<BindingPlanner>{
            std::make_shared<Planner>(std::move(states), std::move(device))};
    } catch (const std::bad_alloc&) {
        return RendererError{CommandResult{
            CommandStatus::BudgetExceeded,
            "binding planner allocation failed"}};
    }
}

} // namespace armigo::platform::metal
