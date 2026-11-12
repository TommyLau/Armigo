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
#include "engine_pipeline.h"
#include "fragment_output.h"

#include "native_resources.h"
#include "texture_format.h"

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <map>
#include <unordered_map>
#include <new>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace armigo::platform::metal {
namespace {

using namespace graphics;
namespace gm = shader::graphics_msl;

constexpr std::uint64_t stage_budget = 256ull << 20;

RendererError failure(std::string detail,
                      CommandStatus status = CommandStatus::Unsupported) {
    return RendererError({status, detail});
}

void require(bool condition, std::string_view detail,
             CommandStatus status = CommandStatus::Unsupported) {
    if (!condition) throw failure(std::string(detail), status);
}

template <class T>
void append_scalar(std::string& out, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
    out.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void append_text(std::string& out, std::string_view text) {
    append_scalar(out, std::uint64_t(text.size()));
    out.append(text);
}

std::string reflection_record(const CompiledGraphicsStage& stage) {
    std::string out;
    append_scalar(out, stage.stage);
    append_text(out, stage.entry_point);
    append_scalar(out, stage.vertex_depth);
    append_scalar(out, stage.fragment_position);
    append_scalar(out, std::uint64_t(stage.vertex_attributes.size()));
    for (const auto& item : stage.vertex_attributes) {
        append_scalar(out, item.generic_slot);
        append_scalar(out, item.attribute_index);
        append_scalar(out, item.format);
    }
    append_scalar(out, std::uint64_t(stage.vertex_class_inputs.size()));
    for (const auto& item : stage.vertex_class_inputs) {
        append_scalar(out, item.location);
        append_scalar(out, item.input_class);
    }
    append_scalar(out, std::uint64_t(stage.constant_buffers.size()));
    for (const auto& item : stage.constant_buffers) {
        append_scalar(out, item.guest_binding);
        append_scalar(out, item.minimum_size);
        append_scalar(out, item.metal_buffer_index);
    }
    append_scalar(out, std::uint64_t(stage.textures.size()));
    for (const auto& item : stage.textures) {
        append_scalar(out, item.handle_byte_offset);
        append_scalar(out, item.texture_index);
        append_scalar(out, item.sampler_index);
    }
    append_scalar(out, stage.interface.position_input_mask);
    append_scalar(out, stage.interface.position_output_mask);
    append_scalar(out, stage.interface.color_output_mask);
    append_scalar(out, std::uint64_t(stage.interface.generic_inputs.size()));
    for (const auto& item : stage.interface.generic_inputs) {
        append_scalar(out, item.slot);
        append_scalar(out, item.component_mask);
        append_scalar(out, item.interpolation);
    }
    append_scalar(out, std::uint64_t(stage.interface.generic_outputs.size()));
    for (const auto& item : stage.interface.generic_outputs) {
        append_scalar(out, item.slot);
        append_scalar(out, item.component_mask);
        append_scalar(out, item.interpolation);
    }
    append_scalar(out, std::uint64_t(stage.color_outputs.size()));
    for (const auto& item : stage.color_outputs) {
        append_scalar(out, item.location);
        append_scalar(out, item.component_mask);
        append_scalar(out, item.scalar);
        append_scalar(out, item.components);
    }
    append_scalar(out, stage.fragment_output_types_from_render_targets);
    append_scalar(out, stage.writes_fragment_depth);
    append_scalar(out, stage.writes_sample_mask);
    append_scalar(out, stage.texture_handle_cbuf.has_value());
    if (stage.texture_handle_cbuf) append_scalar(out, *stage.texture_handle_cbuf);
    append_scalar(out, std::uint64_t(stage.texture_facts.size()));
    for (const auto& item : stage.texture_facts) {
        append_scalar(out, item.cbuf_slot);
        append_scalar(out, item.handle_byte_offset);
        append_scalar(out, item.normalized_coordinates);
        append_scalar(out, item.comparison);
        append_scalar(out, item.dimension);
        append_scalar(out, item.array);
        append_scalar(out, item.multisample);
        append_scalar(out, item.scalar);
        append_scalar(out, item.component_format);
    }
    append_scalar(out, std::uint64_t(stage.constant_buffer_windows.size()));
    for (const auto& item : stage.constant_buffer_windows) {
        append_scalar(out, item.guest_binding);
        append_scalar(out, item.window_bytes);
        append_scalar(out, item.metal_buffer_index);
        append_scalar(out, item.zero_padding_bytes);
    }
    append_scalar(out, stage.constant_buffer_support_index.has_value());
    if (stage.constant_buffer_support_index)
        append_scalar(out, *stage.constant_buffer_support_index);
    append_scalar(out, stage.draw_support_buffer_index.has_value());
    if (stage.draw_support_buffer_index)
        append_scalar(out, *stage.draw_support_buffer_index);
    append_scalar(out, stage.emulates_depth_bias);
    return out;
}

// Derive content identities from exact bytes and reflection with the platform
// SHA-256 primitive, as package admission does.
std::array<std::byte, 32> sha256(std::span<const std::byte> input) {
    require(input.size() <= std::numeric_limits<CC_LONG>::max(),
            "Engine stage identity input exceeds the digest limit", CommandStatus::BudgetExceeded);
    std::array<std::byte, 32> result{};
    require(CC_SHA256(input.data(), static_cast<CC_LONG>(input.size()),
                     reinterpret_cast<unsigned char*>(result.data())) != nullptr,
            "Engine stage identity digest failed");
    return result;
}

std::string digest(std::span<const std::byte> bytes) {
    constexpr char digits[] = "0123456789abcdef";
    const auto value = sha256(bytes);
    std::string result = "sha256:";
    result.reserve(7 + value.size() * 2);
    for (const auto byte : value) {
        const auto v = std::to_integer<unsigned>(byte);
        result.push_back(digits[v >> 4]);
        result.push_back(digits[v & 15]);
    }
    return result;
}

std::string stage_package_component(const CompiledGraphicsStage& stage) {
    const auto record = reflection_record(stage);
    return digest(stage.metallib) + ':' +
           digest(std::as_bytes(std::span(record)));
}

ShaderFunctionKey function_key(const CompiledGraphicsStage& stage,
                               const std::string& package) {
    const auto record = reflection_record(stage);
    return {0, package, digest(std::as_bytes(std::span(record))),
            digest(stage.metallib), stage.entry_point, fragment_output_specialization(stage)};
}

void validate_stage_resources(const CompiledGraphicsStage& stage,
                              shader_input::Stage expected,
                              std::optional<std::uint32_t> texture_selector) {
    require(stage.stage == expected, "Engine pipeline stage kind is invalid");
    require(!stage.metallib.empty() && stage.metallib.size() <= stage_budget &&
                !stage.entry_point.empty() &&
                stage.entry_point.find('\0') == std::string::npos,
            "Engine pipeline stage library or entry point is invalid");
    std::set<unsigned> guest, buffers, textures, samplers, handles, facts;
    for (const auto& binding : stage.constant_buffers)
        require(binding.guest_binding < 18 && binding.metal_buffer_index < 18 &&
                    binding.minimum_size >= 4 && binding.minimum_size <= 65536 &&
                    !(binding.minimum_size & 3u) &&
                    guest.insert(binding.guest_binding).second &&
                    buffers.insert(binding.metal_buffer_index).second,
                "Engine pipeline constant-buffer reflection is invalid");
    require(stage.constant_buffer_windows.size() <= 18 &&
                stage.constant_buffer_support_index.has_value() ==
                    !stage.constant_buffer_windows.empty() &&
                (stage.constant_buffer_windows.empty() || stage.constant_buffers.empty()),
            "Engine pipeline constant-buffer policy is invalid");
    for (const auto& binding : stage.constant_buffer_windows)
        require(binding.guest_binding < 18 && binding.metal_buffer_index < 31 &&
                    binding.window_bytes >= 4 && binding.window_bytes <= 65536 &&
                    !(binding.window_bytes & 3u) &&
                    (binding.zero_padding_bytes == 0 || binding.zero_padding_bytes == 16) &&
                    guest.insert(binding.guest_binding).second &&
                    buffers.insert(binding.metal_buffer_index).second,
                "Engine pipeline constant-buffer window is invalid");
    if (stage.constant_buffer_support_index)
        require(*stage.constant_buffer_support_index < 31 &&
                    buffers.insert(*stage.constant_buffer_support_index).second,
                "Engine pipeline constant-buffer support index is invalid");
    if (stage.emulates_depth_bias)
        require(expected == shader_input::Stage::Fragment && !stage.writes_fragment_depth &&
                    buffers.insert(30).second,
                "shader depth bias ABI requires an exclusive fragment buffer 30");
    // The command-stream renderer binds vertex input through descriptors only.
    require(!stage.vertex_fetch_buffer_index,
            "Shader-side vertex fetch stages are supported by the NVN driver only");
    require(!stage.guest_memory, "Guest-memory window tables are supported by the NVN driver only");
    require(!stage.requires_rasterizer_discard, "Vertex stages without outputs are supported by the NVN driver only");
    if (stage.draw_support_buffer_index)
        require(expected == shader_input::Stage::Vertex &&
                    *stage.draw_support_buffer_index < 31 &&
                    buffers.insert(*stage.draw_support_buffer_index).second,
                "Draw-parameter support requires an exclusive vertex buffer index");
    if (stage.texture_handle_cbuf) {
        require(*stage.texture_handle_cbuf < 18,
                "Engine pipeline texture selector reflection is invalid");
        if (texture_selector)
            require(*texture_selector == *stage.texture_handle_cbuf,
                    "Engine pipeline live texture selector differs from reflection");
    }
    for (const auto& binding : stage.textures)
        require(binding.texture_index < 16 &&
                    (binding.sampler_index < 16 || binding.sampler_index == gm::TextureBinding::no_sampler) &&
                    !(binding.handle_byte_offset & 3u) &&
                    handles.insert(binding.handle_byte_offset).second &&
                    textures.insert(binding.texture_index).second &&
                    (binding.sampler_index == gm::TextureBinding::no_sampler ||
                     samplers.insert(binding.sampler_index).second),
                "Engine pipeline texture reflection is invalid");
    for (const auto& fact : stage.texture_facts)
        require(fact.cbuf_slot < 18 && !(fact.handle_byte_offset & 3u) &&
                    facts.insert(fact.handle_byte_offset).second &&
                    handles.contains(fact.handle_byte_offset) &&
                    stage.texture_handle_cbuf &&
                    fact.cbuf_slot == *stage.texture_handle_cbuf,
                "Engine pipeline texture fact is stale or invalid");
    require(stage.texture_facts.empty() ||
                stage.texture_facts.size() == stage.textures.size(),
            "Engine pipeline texture facts do not cover every binding");
}

// The draw topology decodes like validate_engine_stop: the override register
// (method 0x652) selects the hardware topology word (method 0x65c, where 1 is
// points), otherwise the begin-op topology (where 0 is points).
[[nodiscard]] bool draw_points(const RenderStop& stop) {
    const auto& r = stop.registers;
    const auto topology = r[0x652] ? r[0x65c] : stop.draw->topology;
    return r[0x652] ? topology == 1 : topology == 0;
}

void validate_stage_pair(const CompiledGraphicsPair& pair,
                         std::optional<std::uint32_t> texture_selector) {
    require(pair.vertex && pair.fragment,
            "Engine pipeline compiled stage pair is missing");
    const auto& vertex = *pair.vertex;
    const auto& fragment = *pair.fragment;
    validate_stage_resources(vertex, shader_input::Stage::Vertex, texture_selector);
    validate_stage_resources(fragment, shader_input::Stage::Fragment, texture_selector);
    require(vertex.vertex_depth == gm::VertexDepthConvention::NegativeOneToOne &&
                vertex.fragment_position == gm::FragmentPositionConvention::Unspecified &&
                fragment.vertex_depth == gm::VertexDepthConvention::Unspecified &&
                fragment.vertex_attributes.empty() &&
                (fragment.fragment_position ==
                     gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW ||
                 (fragment.fragment_position == gm::FragmentPositionConvention::Unspecified &&
                  !fragment.interface.position_input_mask)),
            "Engine pipeline coordinate variant is unsupported");
    // Fragment stages may export depth and sample mask. The MSL backend emits
    // the depth output as [[depth(any)]] with no [[early_fragment_tests]], so
    // Metal runs the depth test after the fragment stage and the exported
    // value replaces the rasterized depth for both the compare and the write.
    // A pass without a depth attachment discards the export, and Metal ANDs a
    // sample-mask export into the rasterizer coverage. Depth-bias emulation
    // stays exclusive with a guest depth export (validate_stage_resources).
    require(vertex.interface.position_input_mask == 0 &&
                vertex.interface.position_output_mask && !(vertex.interface.position_output_mask & ~15u) &&
                vertex.interface.color_output_mask == 0 &&
                !vertex.fragment_output_types_from_render_targets &&
                !vertex.writes_fragment_depth && !vertex.writes_sample_mask &&
                fragment.interface.position_output_mask == 0 &&
                !(fragment.interface.position_input_mask & ~15u) &&
                fragment.interface.generic_outputs.empty(),
            "Engine pipeline stage interface is unsupported");
    std::uint32_t color_summary = 0;
    std::set<unsigned> outputs;
    for (const auto& output : fragment.color_outputs) {
        require(output.location < 8 && output.component_mask &&
                    output.components >= 1 && output.components <= 4 &&
                    (!fragment.fragment_output_types_from_render_targets || output.components == 4) &&
                    !(output.component_mask >> output.components) &&
                    (output.scalar == ColorOutputScalar::f32 ||
                     output.scalar == ColorOutputScalar::u32 ||
                     output.scalar == ColorOutputScalar::s32) &&
                    outputs.insert(output.location).second,
                "Engine pipeline fragment output reflection is invalid");
        color_summary |= std::uint32_t(output.component_mask) << (4 * output.location);
    }
    require(color_summary == fragment.interface.color_output_mask,
            "Engine pipeline fragment outputs disagree with the summary mask");

    std::set<unsigned> locations, attributes;
    const bool class_inputs = !vertex.vertex_class_inputs.empty();
    require(!class_inputs || vertex.vertex_attributes.empty(),
            "Engine pipeline vertex reflection mixes input models");
    require(vertex.interface.generic_inputs.size() ==
                (class_inputs ? vertex.vertex_class_inputs.size()
                              : vertex.vertex_attributes.size()),
            "Engine pipeline vertex reflection is incomplete");
    for (const auto& input : vertex.vertex_attributes) {
        require(input.generic_slot < 32 && input.attribute_index < 31 &&
                    input.format == gm::VertexFormat::Float32x3 &&
                    locations.insert(input.generic_slot).second &&
                    attributes.insert(input.attribute_index).second,
                "Engine pipeline legacy vertex input is invalid");
        const auto found = std::find_if(vertex.interface.generic_inputs.begin(),
            vertex.interface.generic_inputs.end(), [&](const auto& item) {
                return item.slot == input.generic_slot;
            });
        require(found != vertex.interface.generic_inputs.end() &&
                    found->component_mask && !(found->component_mask & ~7u),
                "Engine pipeline legacy vertex input mask is invalid");
    }
    for (const auto& input : vertex.vertex_class_inputs) {
        require(input.location < 31 &&
                    input.input_class <= VertexFormatClass::sint_class &&
                    locations.insert(input.location).second,
                "Engine pipeline vertex class input is invalid");
        const auto found = std::find_if(vertex.interface.generic_inputs.begin(),
            vertex.interface.generic_inputs.end(), [&](const auto& item) {
                return item.slot == input.location && item.component_mask;
            });
        require(found != vertex.interface.generic_inputs.end(),
                "Engine pipeline vertex class input has no interface record");
    }
    std::set<unsigned> linked;
    for (const auto& output : vertex.interface.generic_outputs)
        require(output.slot < 32 && output.component_mask &&
                    !(output.component_mask & ~15u) &&
                    linked.insert(output.slot).second,
                "Engine pipeline vertex output is invalid");
    linked.clear();
    for (const auto& input : fragment.interface.generic_inputs) {
        require(input.slot < 32 && input.component_mask &&
                    !(input.component_mask & ~15u) &&
                    linked.insert(input.slot).second &&
                    (input.interpolation == shader::graphics::Interpolation::Constant ||
                     input.interpolation == shader::graphics::Interpolation::Perspective ||
                     input.interpolation == shader::graphics::Interpolation::ScreenLinear),
                "Engine pipeline fragment input is invalid");
        require(input.interpolation != shader::graphics::Interpolation::Perspective ||
                    fragment.fragment_position ==
                        gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW,
                "Engine pipeline perspective input lacks its position variant");
        const auto output = std::find_if(vertex.interface.generic_outputs.begin(),
            vertex.interface.generic_outputs.end(), [&](const auto& item) {
                return item.slot == input.slot;
            });
        require(output != vertex.interface.generic_outputs.end() &&
                    (output->component_mask & input.component_mask) == input.component_mask,
                "Engine pipeline stages do not link");
    }
}

void validate_engine_stop(const RenderStop& stop) {
    const auto& r = stop.registers;
    require(stop.engine == engine_class::threed && stop.kind == RenderKind::Draw &&
                stop.draw.has_value(),
            "Engine pipeline requires a stopped 3D draw");
    require(r[0x651] == 1 ||
                (r[0x651] == 0 && stop.effective_render_enable_mode == 1),
            "Engine pipeline conditional rendering is unsupported");
    // L7 does not yet implement sample-mask suppression or color-key tests.
    require((r[0x3ef] & 1u) && r[0x4d5] == 0,
            "Engine pipeline sample mask or color key is unsupported");
    require(stop.programs[1].enabled && stop.programs[5].enabled &&
                !stop.programs[0].enabled && !stop.programs[2].enabled &&
                !stop.programs[3].enabled && !stop.programs[4].enabled,
            "Engine pipeline requires only vertex and fragment programs");
    const auto topology = r[0x652] ? r[0x65c] : stop.draw->topology;
    const bool points = draw_points(stop);
    const bool lines = r[0x652] ? topology == 2 || topology == 3
                              : topology >= 1 && topology <= 3;
    // Point draws rasterize through the translated vertex stage's
    // [[point_size]] export: the program's own size (attribute 0x6c) or the
    // fixed 3D point size register (method 0x546), written at
    // vertex entry. The sprite origin is a fragment function constant. Metal has
    // no point fill mode for triangles (polygon mode POINT), which the draw
    // admission refuses separately.
    if (points)
        require(std::isfinite(std::bit_cast<float>(r[0x546])),
                "Engine pipeline point size register is not finite");
    // Metal has no line-width, smoothing or stipple state. Such lines are
    // drawn one pixel wide, and each distinct state
    // is reported once: a visible approximation, not an exact replay.
    if ((lines || r[0x36b] == 0x1b01 || r[0x36c] == 0x1b01) &&
        (r[0x4ec] != std::bit_cast<std::uint32_t>(1.0f) || r[0x4ed] != std::bit_cast<std::uint32_t>(1.0f) ||
         r[0x55c] != 0 || r[0x59b] != 0)) {
        static std::mutex reported_mutex;
        static std::set<std::array<std::uint32_t, 4>> reported;
        const std::array<std::uint32_t, 4> line_state{r[0x4ec], r[0x4ed], r[0x55c], r[0x59b]};
        const std::lock_guard lock(reported_mutex);
        if (reported.size() < 16 && reported.insert(line_state).second)
            armigo::diag::write(armigo::diag::Level::warning,  "Armigo Metal: lines drawn one pixel wide (width %g, aliased width %g, smooth %u, "
                         "stipple %u): Metal has no line-width, smoothing or stipple state\n",
                         std::bit_cast<float>(line_state[0]), std::bit_cast<float>(line_state[1]),
                         line_state[2], line_state[3]);
    }
    // The admitted shader variant has one established coordinate convention.
    // Signed mirroring, window-space vertices and swizzled coordinates require
    // matching shader variants; changing only winding cannot implement them.
    require(r[0x64b] == 1 && r[0x649] == 0 && r[0x4eb] == 0 && r[0x35f] == 0,
            "Engine pipeline viewport coordinate convention is unsupported");
    for (unsigned slot = 0; slot < 16; ++slot) {
        const auto base = 0x280 + slot * 8;
        const auto sx = std::bit_cast<float>(r[base]);
        const auto sy = std::bit_cast<float>(r[base + 1]);
        if (slot && sx == 0 && sy == 0) continue;
        require(sx > 0 && sy < 0 && r[base + 6] == 0x6420 &&
                    std::bit_cast<float>(r[base + 2]) == 0.5f &&
                    std::bit_cast<float>(r[base + 5]) == 0.5f,
                "Engine pipeline viewport sign, swizzle or depth mapping is unsupported");
    }
}

class StageProvider final : public GraphicsStageProvider {
public:
    StageProvider(platform::CompiledGraphicsLookup lookup, void* context,
                  platform::CompiledComputeLookup compute_lookup, bool immutable_stages)
        : lookup_(lookup), compute_lookup_(compute_lookup), context_(context),
          immutable_stages_(immutable_stages) {}

    RendererResult<AdmittedGraphicsStages> lookup(
        const RenderStop& stop, const GpuAddressSpace& memory) override {
        try {
            require(lookup_ != nullptr,
                    "Engine stage lookup is unavailable in clear-only mode");
            validate_engine_stop(stop);
            auto selected = lookup_(context_, stop, memory);
            if (const auto* command = std::get_if<CommandResult>(&selected)) {
                require(!command->ok(),
                        "Engine stage lookup returned success without stages");
                return RendererError(*command);
            }
            auto pair = std::get<CompiledGraphicsPair>(std::move(selected));
            // Immutable stages validate once per texture selector; digests_
            // retains them, keeping the addresses valid identities.
            const auto validated_key = std::tuple{pair.vertex.get(), pair.fragment.get(), stop.registers[0x982]};
            const bool validate = !immutable_stages_ || !validated_.contains(validated_key);
            if (validate) validate_stage_pair(pair, stop.registers[0x982]);
            AdmittedGraphicsStages result;
            const auto vertex = digests(pair.vertex);
            const auto fragment = digests(pair.fragment);
            const auto package = package_digest(vertex, fragment);
            if (validate && immutable_stages_) validated_.insert(validated_key);
            result.vertex_key = {0, package, vertex.record, vertex.metallib, pair.vertex->entry_point,
                                 fragment_output_specialization(*pair.vertex)};
            result.fragment_key = {0, package, fragment.record, fragment.metallib, pair.fragment->entry_point,
                                   fragment_output_specialization(*pair.fragment)};
            // Pipeline lookups then hash and reject on one word each.
            result.vertex_key.hash = function_key_hash(result.vertex_key);
            result.fragment_key.hash = function_key_hash(result.fragment_key);
            result.stages = std::move(pair);
            return result;
        } catch (const RendererError& error) {
            return error;
        } catch (const std::bad_alloc&) {
            return failure("Engine stage admission allocation failed",
                           CommandStatus::BudgetExceeded);
        } catch (...) {
            return failure("Engine stage lookup failed");
        }
    }

    // Stage selection reads, of the general state registers, only the clip
    // convention, which these checks pin; the rest of them only admit.
    RendererStatus validate(const RenderStop& stop) override {
        try {
            validate_engine_stop(stop);
            return renderer_success();
        } catch (const RendererError& error) {
            return error;
        } catch (...) {
            return failure("Engine stage admission failed");
        }
    }

    RendererResult<AdmittedComputeStage> lookup_compute(
        const RenderStop& stop, const GpuAddressSpace& memory) override {
        try {
            require(compute_lookup_ != nullptr,
                    "Engine compute stage lookup is unavailable");
            require(stop.engine == engine_class::compute && stop.kind == RenderKind::Compute &&
                        stop.compute.has_value() && stop.has_compute_descriptor,
                    "Engine compute lookup requires a stopped queue-meta-data launch");
            auto selected = compute_lookup_(context_, stop, memory);
            if (const auto* command = std::get_if<CommandResult>(&selected)) {
                require(!command->ok(),
                        "Engine compute lookup returned success without a stage");
                return RendererError(*command);
            }
            AdmittedComputeStage result;
            result.stage = std::get<std::shared_ptr<const CompiledGraphicsStage>>(std::move(selected));
            require(result.stage && result.stage->stage == shader_input::Stage::Compute &&
                        !result.stage->metallib.empty() &&
                        result.stage->metallib.size() <= stage_budget &&
                        result.stage->entry_point == "armigo_compute_main",
                    "Engine compute lookup returned an invalid compute stage");
            // Immutable package stages keep their key: hashing the metallib
            // on every dispatch dominated compute admission.
            if (!immutable_stages_) {
                result.key = function_key(*result.stage, stage_package_component(*result.stage));
                return result;
            }
            auto [entry, inserted] = compute_keys_.try_emplace(result.stage.get());
            if (inserted) {
                try {
                    entry->second.first = result.stage;
                    entry->second.second = function_key(*result.stage, stage_package_component(*result.stage));
                    entry->second.second.hash = function_key_hash(entry->second.second);
                } catch (...) {
                    compute_keys_.erase(entry);
                    throw;
                }
            }
            result.key = entry->second.second;
            return result;
        } catch (const RendererError& error) {
            return error;
        } catch (const std::bad_alloc&) {
            return failure("Engine compute stage admission allocation failed",
                           CommandStatus::BudgetExceeded);
        } catch (...) {
            return failure("Engine compute stage lookup failed");
        }
    }

private:
    // Package stages are immutable shared objects: their content digests are
    // computed once per stage instead of hashing each metallib on every draw.
    // The retained stage keeps its address a valid identity.
    struct StageDigests {
        std::shared_ptr<const CompiledGraphicsStage> stage;
        std::string metallib, record;
    };
    StageDigests digests(const std::shared_ptr<const CompiledGraphicsStage>& stage) {
        const auto compute = [&] {
            const auto record = reflection_record(*stage);
            return StageDigests{stage, digest(stage->metallib), digest(std::as_bytes(std::span(record)))};
        };
        if (!immutable_stages_) return compute();
        auto [entry, inserted] = digests_.try_emplace(stage.get());
        if (inserted) {
            try {
                entry->second = compute();
            } catch (...) {
                digests_.erase(entry);
                throw;
            }
        }
        return entry->second;
    }

    // The pair digest of immutable stages is computed once per pair; the
    // retained stages in digests_ keep both addresses valid identities.
    std::string package_digest(const StageDigests& vertex, const StageDigests& fragment) {
        const auto compute = [&] {
            const auto component = [](const StageDigests& stage) {
                return stage.metallib + ':' + stage.record;
            };
            const auto joined = component(vertex) + ':' + component(fragment);
            return digest(std::as_bytes(std::span(joined)));
        };
        if (!immutable_stages_) return compute();
        const auto key = std::pair{vertex.stage.get(), fragment.stage.get()};
        if (const auto found = packages_.find(key); found != packages_.end()) return found->second;
        return packages_.emplace(key, compute()).first->second;
    }

    platform::CompiledGraphicsLookup lookup_;
    platform::CompiledComputeLookup compute_lookup_;
    void* context_;
    bool immutable_stages_;
    std::unordered_map<const CompiledGraphicsStage*, StageDigests> digests_;
    std::set<std::tuple<const CompiledGraphicsStage*, const CompiledGraphicsStage*, std::uint32_t>> validated_;
    // Compute keys of immutable stages; the retained stage keeps the address
    // a valid identity.
    std::unordered_map<const CompiledGraphicsStage*,
                       std::pair<std::shared_ptr<const CompiledGraphicsStage>, ShaderFunctionKey>> compute_keys_;
    std::map<std::pair<const CompiledGraphicsStage*, const CompiledGraphicsStage*>, std::string> packages_;
};

} // namespace

RendererResult<std::shared_ptr<GraphicsStageProvider>>
create_engine_stage_provider(platform::CompiledGraphicsLookup lookup,
                             void* lookup_context,
                             platform::CompiledComputeLookup compute_lookup,
                             bool immutable_stages) {
    try {
        return std::shared_ptr<GraphicsStageProvider>(
            std::make_shared<StageProvider>(lookup, lookup_context, compute_lookup, immutable_stages));
    } catch (const std::bad_alloc&) {
        return failure("Engine stage provider allocation failed",
                       CommandStatus::BudgetExceeded);
    }
}

} // namespace armigo::platform::metal
