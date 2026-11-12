// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Sampler state and Metal texture bindings are decoded from guest descriptors.

#include "fragment_output.h"
#include <armigo/platform/metal_draw.h>
#include <armigo/graphics/color_clear.h>
#include <armigo/graphics/texture.h>
#include <armigo/graphics/vertex_input.h>
#include "texture_format.h"
#include "texture_fact.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <string>

namespace armigo::platform {
namespace {
using namespace graphics;
namespace gm = shader::graphics_msl;
constexpr std::size_t budget = 256u << 20;
void require(bool condition, const char* message,
             CommandStatus status = CommandStatus::Unsupported) {
    if (!condition) throw CommandResult{status, message};
}
template<class T> T value(std::variant<T, CommandResult> result) {
    if (auto* error = std::get_if<CommandResult>(&result)) throw *error;
    return std::get<T>(std::move(result));
}
void read(const GpuAddressSpace& memory, std::uint64_t address,
          std::span<std::byte> bytes) {
    require(bytes.size() <= UINT64_MAX - address, "Metal draw GPU range overflows");
    while (!bytes.empty()) {
        std::size_t size = 0;
        for (const auto& mapping : memory.mappings()) {
            if (address >= mapping.gpu_base && address - mapping.gpu_base < mapping.mapped_size) {
                size = std::min<std::uint64_t>(bytes.size(), mapping.mapped_size - (address - mapping.gpu_base));
                break;
            }
        }
        require(size && memory.read(address, bytes.first(size)).ok(),
                "Metal draw guest-memory read failed", CommandStatus::MemoryFault);
        address += size;
        bytes = bytes.subspan(size);
    }
}
ColorClearPlan layout(std::uint64_t address, std::uint32_t width,
                      std::uint32_t height, std::uint32_t block) {
    require(width && height && width <= 16384 && height <= 16384 && block <= 5,
            "Metal draw image dimensions or block height are unsupported");
    ColorClearPlan plan;
    plan.target_address = address; plan.width = width; plan.height = height;
    plan.layout = ColorTargetLayout::BlockLinear; plan.block_height_log2 = block;
    plan.row_bytes = (std::size_t(width) * 4 + 63) & ~std::size_t(63);
    const auto rows = (std::size_t(height) + (8u << block) - 1) & ~std::size_t((8u << block) - 1);
    plan.storage_size = plan.row_bytes * rows;
    require(plan.storage_size <= budget && address <= UINT64_MAX - plan.storage_size,
            "Metal draw image exceeds backing budget", CommandStatus::BudgetExceeded);
    return plan;
}
std::vector<std::byte> linearize(const ColorClearPlan& plan, const GpuAddressSpace& memory) {
    std::vector<std::byte> tiled(plan.storage_size), linear(std::size_t(plan.width) * plan.height * 4);
    read(memory, plan.target_address, tiled);
    for (std::uint32_t y = 0; y < plan.height; ++y)
        for (std::uint32_t x = 0; x < plan.width; ++x) {
            const auto offset = value(pixel_byte_offset(plan, x, y));
            std::memcpy(linear.data() + (std::size_t(y) * plan.width + x) * 4, tiled.data() + offset, 4);
        }
    return linear;
}
// Sampled formats map through the TIC format table. The mapping returns the
// Metal sampling format and channel swizzle, or a structured error for an
// unsupported aspect. Packed depth data is converted before GPU upload;
// sRGB selection remains part of the table format identity.
std::variant<MetalSampledFormat, graphics::TextureError> sampled_pixel_format(
    const TextureDescriptor& texture) {
    return metal_sampled_format(texture);
}
// Float-sampled classes per the derived descriptor contract: unorm, snorm,
// float and sRGB forms, including the compressed BCn/ETC2/ASTC families.
// Integer and depth/stencil formats are explicit mismatches.
bool float_sampled(TextureFormat format) noexcept {
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
    default: return true;
    }
}
const DerivedTextureFact* recorded_texture_fact(const CompiledGraphicsStage& stage,
                                                std::uint32_t handle_byte_offset) noexcept {
    for (const auto& fact:stage.texture_facts)
        if (fact.handle_byte_offset==handle_byte_offset) return &fact;
    return nullptr;
}
std::string fact_handle_text(const DerivedTextureFact& fact) {
    return "cbuf "+std::to_string(unsigned(fact.cbuf_slot))+" offset "+
           std::to_string(fact.handle_byte_offset);
}
// CommandResult carries a borrowed detail view; the fact text stays valid
// until the next fact diagnostic on this thread, and the draw caller reads
// the result before issuing another draw.
std::string& fact_error_storage() {
    static thread_local std::string storage;
    return storage;
}
CommandResult fact_error(std::string text) {
    fact_error_storage() = std::move(text);
    return CommandResult{CommandStatus::Unsupported, fact_error_storage()};
}
// A consumed texture_descriptor fact is a draw-time contract: the offline
// translation proved the handle's shape from the shader's own texture
// instructions, and everything the code cannot know is verified against the
// live descriptors here. A live TIC the decode profile cannot describe stops
// inside resolve_texture_binding and is reported as a fact mismatch; the
// checks below compare the decoded target, coordinates, comparison and format
// class with the fact. The draw stops
// before any upload or sampling; never sample with a mismatched type.
void verify_texture_descriptor_fact(const CompiledGraphicsStage& stage,
                                    std::uint32_t handle_byte_offset,
                                    const ResolvedTextureBinding& resolved) {
    const auto* fact=recorded_texture_fact(stage,handle_byte_offset);
    if (!fact) return;
    const auto mismatch = metal::texture_fact_mismatch(*fact, resolved);
    if (!mismatch.empty())
        throw fact_error("texture_descriptor fact mismatch: " + mismatch +
                         " for handle " + fact_handle_text(*fact));
}
// Vertex formats map one to one onto MTLVertexFormat: normalized and float
// forms convert to float, integer forms keep raw bits. The plan only carries
// formats this function supports, so any residual value is an internal fault
// rather than a substitute format.
MTLVertexFormat metal_vertex_format(graphics::VertexAttribFormat format) {
    using graphics::VertexAttribFormat;
    switch (format) {
    case VertexAttribFormat::R8Unorm: return MTLVertexFormatUCharNormalized;
    case VertexAttribFormat::R8Snorm: return MTLVertexFormatCharNormalized;
    case VertexAttribFormat::R8Uint: return MTLVertexFormatUChar;
    case VertexAttribFormat::R8Sint: return MTLVertexFormatChar;
    case VertexAttribFormat::R16Float: return MTLVertexFormatHalf;
    case VertexAttribFormat::R16Unorm: return MTLVertexFormatUShortNormalized;
    case VertexAttribFormat::R16Snorm: return MTLVertexFormatShortNormalized;
    case VertexAttribFormat::R16Uint: return MTLVertexFormatUShort;
    case VertexAttribFormat::R16Sint: return MTLVertexFormatShort;
    case VertexAttribFormat::R32Float: return MTLVertexFormatFloat;
    case VertexAttribFormat::R32Uint: return MTLVertexFormatUInt;
    case VertexAttribFormat::R32Sint: return MTLVertexFormatInt;
    case VertexAttribFormat::R8G8Unorm: return MTLVertexFormatUChar2Normalized;
    case VertexAttribFormat::R8G8Snorm: return MTLVertexFormatChar2Normalized;
    case VertexAttribFormat::R8G8Uint: return MTLVertexFormatUChar2;
    case VertexAttribFormat::R8G8Sint: return MTLVertexFormatChar2;
    case VertexAttribFormat::R16G16Float: return MTLVertexFormatHalf2;
    case VertexAttribFormat::R16G16Unorm: return MTLVertexFormatUShort2Normalized;
    case VertexAttribFormat::R16G16Snorm: return MTLVertexFormatShort2Normalized;
    case VertexAttribFormat::R16G16Uint: return MTLVertexFormatUShort2;
    case VertexAttribFormat::R16G16Sint: return MTLVertexFormatShort2;
    case VertexAttribFormat::R32G32Float: return MTLVertexFormatFloat2;
    case VertexAttribFormat::R32G32Uint: return MTLVertexFormatUInt2;
    case VertexAttribFormat::R32G32Sint: return MTLVertexFormatInt2;
    case VertexAttribFormat::R8G8B8Unorm: return MTLVertexFormatUChar3Normalized;
    case VertexAttribFormat::R8G8B8Snorm: return MTLVertexFormatChar3Normalized;
    case VertexAttribFormat::R8G8B8Uint: return MTLVertexFormatUChar3;
    case VertexAttribFormat::R8G8B8Sint: return MTLVertexFormatChar3;
    case VertexAttribFormat::R16G16B16Float: return MTLVertexFormatHalf3;
    case VertexAttribFormat::R16G16B16Unorm: return MTLVertexFormatUShort3Normalized;
    case VertexAttribFormat::R16G16B16Snorm: return MTLVertexFormatShort3Normalized;
    case VertexAttribFormat::R16G16B16Uint: return MTLVertexFormatUShort3;
    case VertexAttribFormat::R16G16B16Sint: return MTLVertexFormatShort3;
    case VertexAttribFormat::R32G32B32Float: return MTLVertexFormatFloat3;
    case VertexAttribFormat::R32G32B32Uint: return MTLVertexFormatUInt3;
    case VertexAttribFormat::R32G32B32Sint: return MTLVertexFormatInt3;
    case VertexAttribFormat::R8G8B8A8Unorm: return MTLVertexFormatUChar4Normalized;
    case VertexAttribFormat::R8G8B8A8Snorm: return MTLVertexFormatChar4Normalized;
    case VertexAttribFormat::R8G8B8A8Uint: return MTLVertexFormatUChar4;
    case VertexAttribFormat::R8G8B8A8Sint: return MTLVertexFormatChar4;
    case VertexAttribFormat::R16G16B16A16Float: return MTLVertexFormatHalf4;
    case VertexAttribFormat::R16G16B16A16Unorm: return MTLVertexFormatUShort4Normalized;
    case VertexAttribFormat::R16G16B16A16Snorm: return MTLVertexFormatShort4Normalized;
    case VertexAttribFormat::R16G16B16A16Uint: return MTLVertexFormatUShort4;
    case VertexAttribFormat::R16G16B16A16Sint: return MTLVertexFormatShort4;
    case VertexAttribFormat::R32G32B32A32Float: return MTLVertexFormatFloat4;
    case VertexAttribFormat::R32G32B32A32Uint: return MTLVertexFormatUInt4;
    case VertexAttribFormat::R32G32B32A32Sint: return MTLVertexFormatInt4;
    case VertexAttribFormat::A2B10G10R10Unorm: return MTLVertexFormatUInt1010102Normalized;
    case VertexAttribFormat::A2B10G10R10Snorm: return MTLVertexFormatInt1010102Normalized;
    case VertexAttribFormat::B10G11R11Float: return MTLVertexFormatFloatRG11B10;
    }
    throw CommandResult{CommandStatus::Unsupported, "Vertex attribute format is unsupported"};
}
struct Admission {
    ColorClearPlan target;
    MTLViewport viewport;
    MTLScissorRect scissor;
    bool blending;
    MTLPixelFormat pixel_format;
};
Admission admit(const RenderStop& stop) {
    const auto& r = stop.registers;
    require(stop.engine == engine_class::threed && stop.kind == RenderKind::Draw && stop.draw.has_value(),
            "Metal draw requires a stopped 3D draw");
    require(stop.inline_indices.empty(),
            "Legacy Metal draw does not support inline indices");
    const auto& draw = *stop.draw;
    require((r[0x651] == 1 || (r[0x651] == 0 && stop.effective_render_enable_mode == 1)) && r[0xdf] == 1,
            "Metal draw render permission or rasterization is unsupported");
    require(draw.indexed && draw.topology == 4 && draw.instance_count == 1 && draw.base_instance == 0 &&
                draw.instance_index == 0 &&
            draw.count && draw.count <= 1u << 20 && draw.count % 3 == 0 && r[0x5f6] == 1 &&
            r[0x591] == 0 && r[0x37a] == 0 && r[0x652] == 0,
            "Metal draw requires non-restarting indexed u16 triangles and one instance");
    require((r[0x487] & 0x7f) == 1 && r[0x36b] == 0x1b02 && r[0x36c] == 0x1b02,
            "Metal draw requires one color target and filled polygons");
    require(r[0x4b9] == 1 && r[0x4d8] <= 1 && r[0x3e4] == 0 && r[0x680] == 0x1111 &&
            !stop.depth.enabled && r[0x54e] == 0 && r[0x4b3] == 0 && r[0x4ba] == 0 &&
            r[0x4e0] == 0 && r[0x4bb] == 0 && r[0x671] == 0 && r[0x54d] == 0 &&
            r[0x574] == 0 && r[0x54f] == 0 && (r[0x3ef] & 1) != 0,
            "Metal draw blend, masks, depth, stencil, alpha, logic or sample state is unsupported");
    require(r[0x374] == 0 && r[0x4d5] == 0 && r[0x4d7] == 0,
            "Metal draw blend program, color key or single ROP state is unsupported");
    if (r[0x4d8] != 0) {
        // Independent target zero: add with source-alpha and its complement
        // for both RGB and alpha. Other equations and blend programs remain
        // explicit rejections rather than inheriting host pipeline defaults.
        require(r[0x780] == 0 && r[0x781] == 1 && r[0x782] == 5 && r[0x783] == 6 &&
                r[0x784] == 1 && r[0x785] == 5 && r[0x786] == 6,
                "Metal draw alpha blend profile is unsupported");
    }
    require(r[0x646] <= 1 && r[0x647] == 0x901 && r[0x648] == 0x405,
            "Metal draw supports counterclockwise front faces and optional back culling");
    require(r[0x64b] == 1 && r[0x649] == 0 && r[0x4eb] == 0 && r[0x286] == 0x6420 &&
            r[0x35f] == 0 && r[0x64f] == 0x181d,
            "Metal draw viewport, origin, swizzle or depth convention is unsupported");
    require(stop.programs[1].enabled && stop.programs[5].enabled && !stop.programs[0].enabled &&
            !stop.programs[2].enabled && !stop.programs[3].enabled && !stop.programs[4].enabled,
            "Metal draw requires only vertex and fragment programs");
    const auto& target = stop.color_targets[0];
    require((target.format == 0xd6 || target.format == 0xd1) &&
            target.depth == 1 && target.base_layer == 0 &&
            (target.tile_mode & ~0x70u) == 0,
            "Metal draw requires a single-layer block-linear RGB10A2 or RGBA8 sRGB target");
    auto plan = layout(target.address, target.width, target.height, (target.tile_mode >> 4) & 7);
    plan.format = static_cast<ColorTargetFormat>(target.format);
    require(std::uint64_t(target.array_pitch) * 4 >= plan.storage_size,
            "Metal draw target array pitch is smaller than its backing");
    const auto f = [&](std::size_t i) { return std::bit_cast<float>(r[i]); };
    const float sx=f(0x280), sy=f(0x281), tx=f(0x283), ty=f(0x284);
    const bool full_clip = r[0x300] == target.width << 16 &&
                           r[0x301] == target.height << 16;
    const auto clip_width = r[0x300] >> 16, clip_height = r[0x301] >> 16;
    const bool matching_subregion = (r[0x300] & 0xffff) == 0 &&
        (r[0x301] & 0xffff) == 0 && clip_width > 0 && clip_height > 0 &&
        clip_width <= target.width && clip_height <= target.height &&
        tx == sx && ty == -sy && sx * 2 == clip_width && -sy * 2 == clip_height;
    require(std::isfinite(sx) && std::isfinite(sy) && std::isfinite(tx) && std::isfinite(ty) &&
            sx > 0 && sy < 0 && f(0x282) == 0.5f && f(0x285) == 0.5f &&
            (full_clip || matching_subregion) &&
            f(0x302) == 0 && f(0x303) == 1 && tx-sx >= 0 && ty+sy >= 0 &&
            tx+sx <= target.width && ty-sy <= target.height,
            "Metal draw viewport extent or depth mapping is unsupported");
    require(r[0x380] == 1, "Metal draw requires an explicit scissor");
    const auto x = r[0x381] & 0xffff, xmax = r[0x381] >> 16;
    const auto y = r[0x382] & 0xffff, ymax = r[0x382] >> 16;
    require(x < xmax && y < ymax && xmax <= target.width && ymax <= target.height &&
            r[0x3fd] == target.width << 16 && r[0x3fe] == target.height << 16,
            "Metal draw scissor or surface clip is unsupported");
    return {plan, MTLViewport{tx-sx, ty+sy, sx*2, -sy*2, 0, 1},
            MTLScissorRect{x,y,xmax-x,ymax-y}, r[0x4d8] != 0,
            target.format == 0xd1 ? MTLPixelFormatRGB10A2Unorm : MTLPixelFormatRGBA8Unorm_sRGB};
}
void reflection(const CompiledGraphicsPair& pair,std::uint32_t texture_selector) {
    require(pair.vertex && pair.fragment, "Metal draw compiled stage pair is missing");
    const auto& v=*pair.vertex; const auto& f=*pair.fragment;
    require(v.stage == shader_input::Stage::Vertex && f.stage == shader_input::Stage::Fragment &&
            !v.metallib.empty() && !f.metallib.empty() && v.metallib.size() <= budget && f.metallib.size() <= budget &&
            !v.entry_point.empty() && !f.entry_point.empty() &&
            v.entry_point.find('\0') == std::string::npos && f.entry_point.find('\0') == std::string::npos,
            "Metal draw stage library or entry point is invalid");
    require(v.vertex_depth == gm::VertexDepthConvention::NegativeOneToOne &&
            v.fragment_position == gm::FragmentPositionConvention::Unspecified &&
            f.vertex_depth == gm::VertexDepthConvention::Unspecified && f.vertex_attributes.empty() &&
            (f.fragment_position == gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW ||
             (f.fragment_position == gm::FragmentPositionConvention::Unspecified && !f.interface.position_input_mask)),
            "Metal draw compiled coordinate variant does not match guest state");
    require(v.interface.position_input_mask == 0 && v.interface.position_output_mask && !(v.interface.position_output_mask & ~15u) &&
            v.interface.color_output_mask == 0 && !v.writes_fragment_depth &&
            !v.writes_sample_mask && f.interface.position_output_mask == 0 &&
            !(f.interface.position_input_mask & ~15u) &&
            f.interface.generic_outputs.empty() &&
            f.textures.size() <= 16 && v.vertex_attributes.size() <= 13,
            "Metal draw compiled stage interface is unsupported");
    // Single-target admission. The renderer binds one RGBA8/RGB10A2 target at
    // location 0, so the fragment stage must export exactly one float4 output
    // there; integer-typed, sparse or multi-target exports stop here with a
    // structured diagnostic, before any GPU work or guest writeback. Actual
    // multiple-render-target rendering is a later lane. The summary-mask
    // re-check is fail-closed: the loader validates the same invariant.
    std::uint32_t fragment_colour_summary = 0;
    for (const auto& output:f.color_outputs)
        fragment_colour_summary |=
            std::uint32_t(output.component_mask) << (4 * output.location);
    require(fragment_colour_summary == f.interface.color_output_mask,
            "Metal draw colour output records disagree with the interface summary mask");
    require(!f.writes_fragment_depth, "Metal draw fragment depth exports are unsupported");
    require(!f.writes_sample_mask, "Metal draw sample mask exports are unsupported");
    require(f.color_outputs.size() == 1 && f.color_outputs[0].location == 0 &&
                f.color_outputs[0].component_mask == 15 &&
                f.color_outputs[0].scalar == ColorOutputScalar::f32 &&
                f.color_outputs[0].components == 4 &&
                f.interface.color_output_mask == 15,
            "Metal draw admits one float4 colour export at render target 0 only");
    std::set<unsigned> slots, attributes;
    const bool class_inputs = !v.vertex_class_inputs.empty();
    require(!class_inputs || v.vertex_attributes.empty(),
            "Metal draw vertex input reflection mixes stream and class inputs");
    require(v.interface.generic_inputs.size() ==
                (class_inputs ? v.vertex_class_inputs.size() : v.vertex_attributes.size()),
            "Metal draw vertex input reflection does not match its format options");
    for (const auto& a:v.vertex_attributes) {
        require(a.generic_slot < 32 && a.attribute_index < 31 && a.format == gm::VertexFormat::Float32x3 &&
                slots.insert(a.generic_slot).second && attributes.insert(a.attribute_index).second,
                "Metal draw vertex attribute option is invalid or duplicated");
        const auto it=std::find_if(v.interface.generic_inputs.begin(),v.interface.generic_inputs.end(),
            [&](const auto& i){return i.slot==a.generic_slot;});
        require(it!=v.interface.generic_inputs.end() && it->component_mask && !(it->component_mask & ~7u),
                "Metal draw vertex format does not cover the consumed components");
    }
    for (const auto& binding:v.vertex_class_inputs) {
        require(binding.location < 31 && slots.insert(binding.location).second,
                "Metal draw vertex class input location is invalid or duplicated");
        const auto it=std::find_if(v.interface.generic_inputs.begin(),v.interface.generic_inputs.end(),
            [&](const auto& i){return i.slot==binding.location && i.component_mask;});
        require(it!=v.interface.generic_inputs.end(),
                "Metal draw vertex class input does not cover a reflected input");
    }
    slots.clear();
    // Interpolation is declared by the fragment inputs; vertex outputs only
    // carry the values, so every parsed form links through the mask check.
    for (const auto& output:v.interface.generic_outputs)
        require(output.slot < 32 && output.component_mask && !(output.component_mask & ~15u) &&
                slots.insert(output.slot).second, "Metal draw vertex output reflection is invalid");
    slots.clear();
    for (const auto& input:f.interface.generic_inputs) {
        require(input.slot < 32 && input.component_mask && !(input.component_mask & ~15u) &&
                slots.insert(input.slot).second &&
                (input.interpolation == shader::graphics::Interpolation::Constant ||
                 input.interpolation == shader::graphics::Interpolation::Perspective ||
                 input.interpolation == shader::graphics::Interpolation::ScreenLinear),
                "Metal draw fragment interface is invalid");
        require(input.interpolation != shader::graphics::Interpolation::Perspective ||
                f.fragment_position == gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW,
                "Metal draw perspective input requires a compiled position convention");
        const auto output=std::find_if(v.interface.generic_outputs.begin(),v.interface.generic_outputs.end(),
            [&](const auto& o){return o.slot==input.slot;});
        require(output!=v.interface.generic_outputs.end() &&
                (output->component_mask & input.component_mask)==input.component_mask,
                "Metal draw vertex and fragment interfaces do not link");
    }
    for (const auto* stage:{&v,&f}) {
        require(!stage->custom_border,
                "Metal draw legacy backend cannot bind coverage-v1 shader support");
        require(!stage->argument_buffer && stage->legacy_texture_bindings_compatible,
                "Metal draw legacy backend requires directly representable texture bindings");
        for (const auto& resource : stage->resources)
            require(resource.cbuf_slot == texture_selector,
                    "Metal draw legacy backend requires resource provenance at the live selector");
        std::set<unsigned> guest,metal, textures,samplers,handles;
        for (const auto& cb:stage->constant_buffers)
            require(cb.guest_binding<18 && cb.metal_buffer_index<18 && cb.minimum_size && (cb.minimum_size & 3u) == 0 && cb.minimum_size<=65536 &&
                    guest.insert(cb.guest_binding).second && metal.insert(cb.metal_buffer_index).second,
                    "Metal draw constant-buffer reflection is invalid");
        require(stage->constant_buffer_windows.size() <= 18 &&
                stage->constant_buffer_support_index.has_value() == !stage->constant_buffer_windows.empty() &&
                (stage->constant_buffer_windows.empty() || stage->constant_buffers.empty()),
                "Metal draw runtime constant-buffer policy is invalid or mixed with legacy bindings");
        for (const auto& cb:stage->constant_buffer_windows)
            require(cb.guest_binding < 18 && cb.metal_buffer_index < 31 &&
                    cb.window_bytes >= 4 && cb.window_bytes <= 65536 && !(cb.window_bytes & 3u) &&
                    (cb.zero_padding_bytes == 0 || cb.zero_padding_bytes == 16) &&
                    guest.insert(cb.guest_binding).second && metal.insert(cb.metal_buffer_index).second,
                    "Metal draw constant-buffer window reflection is invalid");
        if (stage->constant_buffer_support_index)
            require(*stage->constant_buffer_support_index < 31 &&
                    metal.insert(*stage->constant_buffer_support_index).second,
                    "Metal draw constant-buffer support index is invalid or duplicated");
        if (stage->emulates_depth_bias)
            require(stage->stage == shader_input::Stage::Fragment &&
                    !stage->writes_fragment_depth && metal.insert(30).second,
                    "Metal draw depth bias ABI requires an exclusive fragment buffer 30");
        // A consumed texture_handle_cbuf fact is a draw-time contract: bound
        // handles resolve through the live selector register, so selecting any
        // other constant buffer would silently rebind onto foreign contents.
        if (stage->texture_handle_cbuf)
            require(texture_selector==*stage->texture_handle_cbuf,
                    "texture_handle_cbuf fact mismatch: the guest texture buffer selector does not select the compiled handle constant buffer");
        for (const auto& t:stage->textures) {
            require(t.sampler_index != gm::TextureBinding::no_sampler,
                    "Metal draw legacy backend does not support samplerless texture types");
            require(t.texture_index<16 && t.sampler_index<16 && t.handle_byte_offset%4==0 &&
                    handles.insert(t.handle_byte_offset).second &&
                    textures.insert(t.texture_index).second &&
                    samplers.insert(t.sampler_index).second,
                    "Metal draw texture reflection is invalid");
        }
        // Buffer namespaces are stage-local; fragment indices may overlap
        // vertex streams, but a vertex CB or support buffer may not.
        for (std::size_t ordinal = 0; ordinal < stage->vertex_attributes.size(); ++ordinal)
            require(18 + ordinal < 31 && metal.insert(18 + ordinal).second,
                    "Metal draw buffer plan aliases or exceeds vertex-stream indices");
    }
}
CommandResult execute(MetalDrawContext& context,const RenderStop& stop,const GpuAddressSpace& memory) {
    require(context.lookup, "Metal draw has no compiled-stage lookup provider");
    const auto plan=admit(stop);
    auto selected=context.lookup(context.lookup_context,stop,memory);
    if (auto* error=std::get_if<CommandResult>(&selected)) {
        require(!error->ok(), "Metal draw lookup returned success without compiled stages");
        return *error;
    }
    const auto stages=std::get<CompiledGraphicsPair>(std::move(selected));
    const auto& r=stop.registers; const auto& draw=*stop.draw;
    reflection(stages,r[0x982]);
    const auto index_base=(std::uint64_t(r[0x5f2])<<32)|r[0x5f3];
    const auto index_limit=(std::uint64_t(r[0x5f4])<<32)|r[0x5f5];
    const auto first=std::uint64_t(draw.first)*2;
    const auto index_bytes=std::size_t(draw.count)*2;
    require(first<=UINT64_MAX-index_base && index_bytes-1<=UINT64_MAX-(index_base+first) &&
            index_base+first+index_bytes-1<=index_limit, "Metal draw index range exceeds its limit");
    std::vector<std::uint16_t> indices(draw.count);
    read(memory,index_base+first,std::as_writable_bytes(std::span(indices)));
    const auto max_index=*std::max_element(indices.begin(),indices.end());
    auto target_pixels=linearize(plan.target,memory);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    require(device && [device supportsFamily:MTLGPUFamilyMetal4] && device.hasUnifiedMemory,
            "Metal draw requires unified-memory Metal 4");
    NSError* error=nil;
    id<MTL4CommandQueue> queue=[device newMTL4CommandQueue];
    id<MTL4Compiler> compiler=[device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    id<MTL4CommandAllocator> allocator=[device newCommandAllocator];
    id<MTL4CommandBuffer> commands=[device newCommandBuffer];
    auto* rd=[MTLResidencySetDescriptor new]; rd.initialCapacity=64;
    id<MTLResidencySet> residency=[device newResidencySetWithDescriptor:rd error:&error];
    require(queue && compiler && allocator && commands && residency, "Metal draw command allocation failed");
    NSMutableArray* retained=[NSMutableArray arrayWithArray:@[device,queue,compiler,allocator,commands,residency]];
    std::size_t uploaded_bytes = 0;
    const auto retain_resource=[&](id<MTLResource> resource) {
        require(resource!=nil,"Metal draw resource allocation failed");
        [retained addObject:resource]; [residency addAllocation:resource];
    };
    const auto buffer=[&](std::span<const std::byte> data) -> id<MTLBuffer> {
        require(data.size() <= budget-uploaded_bytes, "Metal draw aggregate upload budget exceeded",
                CommandStatus::BudgetExceeded);
        uploaded_bytes += data.size();
        id<MTLBuffer> result=[device newBufferWithBytes:data.data() length:data.size() options:MTLResourceStorageModeShared];
        retain_resource(result); return result;
    };
    auto* td=[MTL4ArgumentTableDescriptor new];
    td.maxBufferBindCount=31; td.maxTextureBindCount=16; td.maxSamplerStateBindCount=16;
    id<MTL4ArgumentTable> vt=[device newArgumentTableWithDescriptor:td error:&error];
    id<MTL4ArgumentTable> ft=[device newArgumentTableWithDescriptor:td error:&error];
    require(vt && ft,"Metal draw argument table allocation failed");
    [retained addObjectsFromArray:@[vt,ft]];
    struct DepthUpload {
        id<MTLBuffer> buffer;
        id<MTLTexture> texture;
        std::size_t row_bytes;
    };
    std::vector<DepthUpload> depth_uploads;
    // Sampler states for this draw, keyed by mapped properties; the per-draw
    // device makes a per-draw cache the correct scope.
    std::vector<std::pair<MetalSamplerProperties, id<MTLSamplerState>>> samplers;
    id<MTLBuffer> index_buffer=buffer(std::as_bytes(std::span(indices)));
    MTLVertexDescriptor* vd=[MTLVertexDescriptor vertexDescriptor];
    id<MTLBuffer> draw_index_buffer=index_buffer;
    if (!stages.vertex->vertex_class_inputs.empty()) {
        // Class-typed inputs: the vertex descriptor is built from the guest
        // attribute registers at draw time and each stream binds as one Metal
        // buffer covering exactly the resolved vertex range the draw touches.
        std::int64_t minimum_vertex=0, maximum_vertex=0;
        for (std::size_t i=0;i<indices.size();++i) {
            const auto resolved=std::int64_t(indices[i])+draw.base_vertex;
            require(resolved>=0,"Metal draw base vertex produces a negative vertex index");
            minimum_vertex=i==0?resolved:std::min(minimum_vertex,resolved);
            maximum_vertex=i==0?resolved:std::max(maximum_vertex,resolved);
        }
        // Class-input streams are chosen by the guest attribute registers at
        // draw time, so their Metal indices are validated here against the
        // same namespace the reflection checked for legacy stream attributes.
        std::set<unsigned> reserved;
        for (const auto& cb:stages.vertex->constant_buffer_windows)
            reserved.insert(cb.metal_buffer_index);
        if (stages.vertex->constant_buffer_support_index)
            reserved.insert(*stages.vertex->constant_buffer_support_index);
        std::map<std::uint32_t,std::vector<graphics::VertexAttributePlan>> by_stream;
        for (const auto& binding:stages.vertex->vertex_class_inputs) {
            auto planned=value(plan_vertex_attribute(stop,binding.location));
            require(planned.format.format_class==binding.input_class,
                    "Metal draw guest vertex format class contradicts the translated input class");
            by_stream[planned.stream].push_back(std::move(planned));
        }
        for (auto& entry:by_stream) {
            const auto stream=value(plan_vertex_stream(stop,entry.first));
            require(18u+entry.first<=30 && !reserved.contains(18u+entry.first),
                    "Metal draw vertex stream aliases or exceeds the buffer binding range");
            const auto slot=18u+entry.first;
            const bool constant=entry.second.front().constant;
            require(std::all_of(entry.second.begin(),entry.second.end(),
                    [&](const auto& a){return a.constant==constant;}),
                    "Metal draw stream mixes constant and stream attributes");
            std::uint32_t reached=0;
            for (const auto& a:entry.second)
                reached=std::max(reached,a.offset+std::uint32_t(a.format.size_bytes));
            if (constant) {
                // Constant attributes bind from a zero buffer;
                // every fetched component reads zero.
                std::vector<std::byte> zeros(reached);
                id<MTLBuffer> zeros_buffer=buffer(zeros);
                [vt setAddress:zeros_buffer.gpuAddress atIndex:slot];
                vd.layouts[slot].stride=0;
            } else {
                constexpr auto maximum=std::numeric_limits<std::uint64_t>::max();
                const auto span=std::uint64_t(maximum_vertex-minimum_vertex);
                require(stream.stride==0||span<=maximum/stream.stride,
                        "Metal draw vertex stride multiplication overflows");
                const auto vertices_bytes=span*stream.stride;
                require(vertices_bytes<=maximum-reached,"Metal draw vertex range overflows");
                const auto start=stream.address+std::uint64_t(minimum_vertex)*stream.stride;
                require(start<=stream.inclusive_limit &&
                        stream.inclusive_limit-start>=vertices_bytes+reached-1,
                        "Metal draw vertex stream range exceeds its inclusive limit",
                        CommandStatus::MemoryFault);
                std::vector<std::byte> bytes(std::size_t(vertices_bytes+reached));
                read(memory,start,bytes);
                id<MTLBuffer> stream_buffer=buffer(bytes);
                [vt setAddress:stream_buffer.gpuAddress atIndex:slot];
                vd.layouts[slot].stride=stream.stride;
            }
            for (const auto& a:entry.second) {
                vd.attributes[a.location].format=metal_vertex_format(a.format.format);
                vd.attributes[a.location].bufferIndex=slot;
                vd.attributes[a.location].offset=a.offset;
            }
        }
        std::vector<std::uint16_t> rebased;
        rebased.reserve(indices.size());
        for (auto index:indices)
            rebased.push_back(std::uint16_t(std::int64_t(index)+draw.base_vertex-minimum_vertex));
        draw_index_buffer=buffer(std::as_bytes(std::span(rebased)));
    }
    for (std::size_t a=0;a<stages.vertex->vertex_attributes.size();++a) {
        const auto& attr=stages.vertex->vertex_attributes[a];
        const auto vp=value(plan_vertex_float3(stop,attr.generic_slot));
        std::vector<std::array<std::uint32_t,3>> vertices(std::size_t(max_index)+1);
        std::vector<bool> fetched(vertices.size());
        for (auto index:indices) if (!fetched[index]) {
            const auto resolved=std::int64_t(index)+draw.base_vertex;
            require(resolved>=0,"Metal draw base vertex produces a negative vertex index");
            vertices[index]=value(fetch_vertex_float3(vp,std::uint64_t(resolved),memory)); fetched[index]=true;
        }
        id<MTLBuffer> vb=buffer(std::as_bytes(std::span(vertices)));
        const auto slot=18+a;
        [vt setAddress:vb.gpuAddress atIndex:slot];
        vd.attributes[attr.attribute_index].format=MTLVertexFormatFloat3;
        vd.attributes[attr.attribute_index].bufferIndex=slot;
        vd.attributes[attr.attribute_index].offset=0; vd.layouts[slot].stride=12;
    }
    for (unsigned stage_index:{0u,4u}) {
        const auto& stage=stage_index==0?*stages.vertex:*stages.fragment;
        id<MTL4ArgumentTable> table=stage_index==0?vt:ft;
        for (const auto& cb:stage.constant_buffers) {
            const auto& binding=stop.constant_buffers[stage_index][cb.guest_binding];
            require(binding.enabled && binding.size>=cb.minimum_size,"Metal draw constant buffer is disabled or too small");
            std::vector<std::byte> bytes(cb.minimum_size); read(memory,binding.address,bytes);
            id<MTLBuffer> b=buffer(bytes); [table setAddress:b.gpuAddress atIndex:cb.metal_buffer_index];
        }
        for (const auto& cb:stage.constant_buffer_windows) {
            const auto& binding=stop.constant_buffers[stage_index][cb.guest_binding];
            std::vector<std::byte> bytes(cb.backing_bytes());
            // Whole words below the bound size; the rest of the window reads zero.
            const auto copied=binding.enabled ? std::min(binding.size,cb.window_bytes)&~3u : 0u;
            if (copied) read(memory,binding.address,std::span(bytes).first(copied));
            id<MTLBuffer> b=buffer(bytes); [table setAddress:b.gpuAddress atIndex:cb.metal_buffer_index];
        }
        if (stage.constant_buffer_support_index) {
            std::array<std::uint32_t,18> sizes{};
            for (std::size_t slot=0;slot<sizes.size();++slot) {
                const auto& binding=stop.constant_buffers[stage_index][slot];
                sizes[slot]=binding.enabled ? binding.size : 0u;
            }
            id<MTLBuffer> b=buffer(std::as_bytes(std::span(sizes)));
            [table setAddress:b.gpuAddress atIndex:*stage.constant_buffer_support_index];
        }
        if (stage.emulates_depth_bias) {
            // This legacy profile has no active depth attachment. New packages
            // still require a bound disabled dynamic state block.
            const std::array<float, 4> disabled{};
            id<MTLBuffer> b=buffer(std::as_bytes(std::span(disabled)));
            [table setAddress:b.gpuAddress atIndex:30];
        }
        for (const auto& binding:stage.textures) {
            const auto* fact=recorded_texture_fact(stage,binding.handle_byte_offset);
            auto resolved=resolve_texture_binding(stop,stage_index,binding.handle_byte_offset,memory);
            if (auto* failure=std::get_if<TextureError>(&resolved)) {
                // A recorded fact makes every live-descriptor rejection a fact
                // mismatch: the offline shape claim did not survive the draw.
                if (fact)
                    throw fact_error("texture_descriptor fact mismatch: the live "
                        "descriptors for handle "+fact_handle_text(*fact)+
                        " do not admit the derived shape: "+failure->message());
                throw CommandResult{CommandStatus::Unsupported,failure->message()};
            }
            const auto& texture=std::get<ResolvedTextureBinding>(resolved);
            verify_texture_descriptor_fact(stage,binding.handle_byte_offset,texture);
            // The descriptor decode describes every target, but this backend
            // binds plain 2D textures only; never upload another target as 2D.
            require(texture.texture.type==TextureType::Texture2D && texture.texture.samples<=1,
                    "Metal draw samples plain 2D textures only");
            require(texture.is_null || texture_format_info(texture.texture.format).depth ||
                        float_sampled(texture.texture.format),
                    "Metal draw legacy backend does not support integer texture types");
            const auto format_info=graphics::texture_format_info(texture.texture.format);
            require(texture.is_null || !format_info.depth || texture.texture.levels==1,
                    "Metal draw depth sampling supports single-level views only");
            id<MTLTexture> tex;
            if (texture.is_null) {
                require(fact || !texture.sampler.depth_compare,
                        "Metal draw null comparison texture requires reflected depth type");
                auto created=metal_null_texture(device,queue,texture.texture,
                    metal::null_texture_scalar(fact),metal::null_texture_depth(fact));
                if (const auto* failure=std::get_if<CommandResult>(&created)) throw *failure;
                tex=std::get<id<MTLTexture>>(created);
                retain_resource(tex);
            } else {
                auto sampled=read_linear_texture(texture.texture,memory);
                if (auto* failure=std::get_if<TextureError>(&sampled))
                    throw CommandResult{failure->code==TextureErrorCode::MemoryRead?CommandStatus::MemoryFault:CommandStatus::Unsupported,failure->message()};
                auto data=std::get<LinearTextureData>(std::move(sampled));
                const bool compressed=graphics::is_block_compressed(format_info);
                if (compressed) {
                    require(!graphics::is_bc_family(texture.texture.format) ||
                            [device supportsBCTextureCompression],
                            "Metal draw BC texture compression is unsupported on this device");
                    require(data.row_bytes==std::size_t((texture.texture.width+format_info.block_width-1)/format_info.block_width)*format_info.bytes_per_block,
                            "Metal draw compressed upload row is not a natural block row");
                }
                if (format_info.depth) {
                    require(texture.sampler.depth_compare,
                            "Metal draw depth sampling requires a comparison sampler");
                    auto converted=metal_depth_upload(texture.texture.format,data.bytes);
                    if (const auto* failure=std::get_if<graphics::TextureError>(&converted))
                        throw CommandResult{CommandStatus::Unsupported,failure->message()};
                    data.bytes=std::get<std::vector<std::byte>>(std::move(converted));
                    data.row_bytes=std::size_t(texture.texture.width)*
                        (texture.texture.format==TextureFormat::D16Unorm?2:4);
                } else {
                    require(!texture.sampler.depth_compare,
                            "Metal draw comparison sampling requires a depth format");
                }
                const auto mapped_format=sampled_pixel_format(texture.texture);
                if (auto* failure=std::get_if<graphics::TextureError>(&mapped_format))
                    throw CommandResult{CommandStatus::Unsupported,failure->message()};
                const auto sampled_format=std::get<MetalSampledFormat>(mapped_format);
                auto* desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:sampled_format.pixel_format
                           width:texture.texture.width height:texture.texture.height mipmapped:NO];
                desc.storageMode=format_info.depth?MTLStorageModePrivate:MTLStorageModeShared;
                desc.usage=MTLTextureUsageShaderRead;
                desc.swizzle=sampled_format.swizzle;
                tex=[device newTextureWithDescriptor:desc]; retain_resource(tex);
                if (format_info.depth) {
                    depth_uploads.push_back({buffer(data.bytes),tex,data.row_bytes});
                } else {
                    require(data.bytes.size() <= budget-uploaded_bytes, "Metal draw aggregate upload budget exceeded",
                            CommandStatus::BudgetExceeded);
                    uploaded_bytes += data.bytes.size();
                    [tex replaceRegion:MTLRegionMake2D(0,0,texture.texture.width,texture.texture.height) mipmapLevel:0
                             withBytes:data.bytes.data() bytesPerRow:data.row_bytes];
                }
            }
            // Samplers map from the live decoded TSC through the platform
            // mapping; states cache by their mapped properties so repeated
            // descriptors inside one draw allocate once. The device here is
            // per-draw, so the cache lives inside this draw execution.
            const auto mapped_sampler=metal_sampler_properties(texture.sampler, texture.texture.srgb);
            if (auto* failure=std::get_if<graphics::TextureError>(&mapped_sampler))
                throw CommandResult{CommandStatus::Unsupported,failure->message()};
            auto properties=std::get<MetalSamplerProperties>(mapped_sampler);
            require(!properties.custom_border,
                    "legacy Metal draw has no custom-border support binding");
            // The guest request table tops out at 16, Metal's documented
            // sampler-anisotropy maximum, so no device clamp applies.
            id<MTLSamplerState> sampler;
            if (const auto cached=std::find_if(samplers.begin(),samplers.end(),
                    [&](const auto& entry){return entry.first==properties;});
                cached!=samplers.end()) {
                sampler=cached->second;
            } else {
                auto* sd=metal_sampler_descriptor(properties);
                sampler=[device newSamplerStateWithDescriptor:sd];
                require(sampler!=nil,"Metal draw sampler allocation failed");
                [retained addObject:sampler];
                samplers.emplace_back(properties,sampler);
            }
            [table setTexture:tex.gpuResourceID atIndex:binding.texture_index];
            [table setSamplerState:sampler.gpuResourceID atIndex:binding.sampler_index];
        }
    }
    const auto library=[&](const CompiledGraphicsStage& stage) -> id<MTLLibrary> {
        dispatch_data_t data=dispatch_data_create(stage.metallib.data(),stage.metallib.size(),nullptr,DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> lib=[device newLibraryWithData:data error:&error];
        require(lib!=nil,"Metal draw compiled library could not be loaded"); [retained addObject:lib];

        require([[lib functionNames] containsObject:
                    [NSString stringWithUTF8String:stage.entry_point.c_str()]],
                "Metal draw compiled library lacks its entry point");
        return lib;
    };
    auto* vf=[MTL4LibraryFunctionDescriptor new]; vf.library=library(*stages.vertex);
    vf.name=[NSString stringWithUTF8String:stages.vertex->entry_point.c_str()];
    auto* ff=[MTL4LibraryFunctionDescriptor new]; ff.library=library(*stages.fragment);
    ff.name=[NSString stringWithUTF8String:stages.fragment->entry_point.c_str()];
    auto* pd=[MTL4RenderPipelineDescriptor new]; pd.vertexFunctionDescriptor=vf; pd.fragmentFunctionDescriptor=metal::specialize_fragment_output(ff,*stages.fragment);
    pd.vertexDescriptor=vd; pd.inputPrimitiveTopology=MTLPrimitiveTopologyClassTriangle;
    auto* attachment=pd.colorAttachments[0];
    attachment.pixelFormat=plan.pixel_format;
    attachment.blendingState=plan.blending ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
    if (plan.blending) {
        attachment.rgbBlendOperation=attachment.alphaBlendOperation=MTLBlendOperationAdd;
        attachment.sourceRGBBlendFactor=attachment.sourceAlphaBlendFactor=MTLBlendFactorSourceAlpha;
        attachment.destinationRGBBlendFactor=attachment.destinationAlphaBlendFactor=MTLBlendFactorOneMinusSourceAlpha;
    }
    // Pipelines are cached by the compiled-stage libraries, the full vertex
    // layout, the render-target format and the blend state — the same key
    // shape required by the guest contract. Draws are serialized by the runtime contract,
    // so an ordinary map is sufficient.
    const auto fingerprint=[](const std::vector<std::byte>& bytes) {
        std::uint64_t hash=1469598103934665603ull;
        for (const auto byte:bytes) {
            hash^=std::to_integer<unsigned>(byte);
            hash*=1099511628211ull;
        }
        return hash;
    };
    std::string pipeline_key=std::to_string(fingerprint(stages.vertex->metallib))+':'+
        std::to_string(fingerprint(stages.fragment->metallib))+':'+
        std::to_string(unsigned(plan.pixel_format))+':'+(plan.blending?"b":"n")+':';
    for (unsigned i=0;i<31;++i) {
        const auto attribute=vd.attributes[i];
        pipeline_key+=std::to_string(unsigned(attribute.format)) + ',' +
            std::to_string(attribute.bufferIndex) + ',' + std::to_string(attribute.offset) + ';';
    }
    for (unsigned i=0;i<31;++i) pipeline_key+=std::to_string(vd.layouts[i].stride)+';';
    static std::map<std::string, id<MTLRenderPipelineState>> pipelines;
    id<MTLRenderPipelineState> pipeline;
    if (const auto cached=pipelines.find(pipeline_key); cached!=pipelines.end()) pipeline=cached->second;
    if (!pipeline) {
        id<MTLRenderPipelineState> created=[compiler newRenderPipelineStateWithDescriptor:pd compilerTaskOptions:nil error:&error];
        require(created!=nil,"Metal draw pipeline creation failed");
        [retained addObject:created];
        pipelines.emplace(pipeline_key,created);
        pipeline=created;
    }
    auto* target_desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:plan.pixel_format
                      width:plan.target.width height:plan.target.height mipmapped:NO];
    target_desc.storageMode=MTLStorageModePrivate; target_desc.usage=MTLTextureUsageRenderTarget;
    id<MTLTexture> target=[device newTextureWithDescriptor:target_desc]; retain_resource(target);
    id<MTLBuffer> upload=buffer(target_pixels);
    id<MTLBuffer> output=[device newBufferWithLength:target_pixels.size() options:MTLResourceStorageModeShared]; retain_resource(output);
    [residency commit]; [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> copy=[commands computeCommandEncoder];
    require(copy!=nil,"Metal draw upload encoder creation failed");
    [copy copyFromBuffer:upload sourceOffset:0 sourceBytesPerRow:std::size_t(plan.target.width)*4
        sourceBytesPerImage:target_pixels.size() sourceSize:MTLSizeMake(plan.target.width,plan.target.height,1)
        toTexture:target destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    for (const auto& depth : depth_uploads) {
        [copy copyFromBuffer:depth.buffer sourceOffset:0 sourceBytesPerRow:depth.row_bytes
            sourceBytesPerImage:depth.row_bytes*depth.texture.height
            sourceSize:MTLSizeMake(depth.texture.width,depth.texture.height,1)
            toTexture:depth.texture destinationSlice:0 destinationLevel:0
            destinationOrigin:MTLOriginMake(0,0,0)];
    }
    [copy endEncoding];
    auto* pass=[MTL4RenderPassDescriptor new]; pass.colorAttachments[0].texture=target;
    pass.colorAttachments[0].loadAction=MTLLoadActionLoad; pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    id<MTL4RenderCommandEncoder> render=[commands renderCommandEncoderWithDescriptor:pass];
    require(render!=nil,"Metal draw render encoder creation failed");
    [render barrierAfterQueueStages:MTLStageBlit beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [render setRenderPipelineState:pipeline]; [render setViewport:plan.viewport]; [render setScissorRect:plan.scissor];
    [render setFrontFacingWinding:MTLWindingCounterClockwise];
    [render setCullMode:r[0x646]?MTLCullModeBack:MTLCullModeNone];
    [render setArgumentTable:vt atStages:MTLRenderStageVertex]; [render setArgumentTable:ft atStages:MTLRenderStageFragment];
    [render drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:draw.count indexType:MTLIndexTypeUInt16
        indexBuffer:draw_index_buffer.gpuAddress indexBufferLength:index_bytes];
    [render endEncoding];
    copy=[commands computeCommandEncoder]; require(copy!=nil,"Metal draw readback encoder creation failed");
    [copy barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
        sourceSize:MTLSizeMake(plan.target.width,plan.target.height,1) toBuffer:output destinationOffset:0
        destinationBytesPerRow:std::size_t(plan.target.width)*4 destinationBytesPerImage:target_pixels.size()];
    [copy endEncoding]; [commands endCommandBuffer];
    dispatch_semaphore_t finished=dispatch_semaphore_create(0); __block NSError* execution_error=nil;
    auto* options=[MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count); static_cast<void>(stages.vertex->metallib.size());
        static_cast<void>(stages.fragment->metallib.size());
        execution_error=feedback.error; dispatch_semaphore_signal(finished);
    }];
    id<MTL4CommandBuffer> submitted[]={commands}; [queue commit:submitted count:1 options:options];
    require(!dispatch_semaphore_wait(finished,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC)),
            "Metal draw GPU completion timed out",CommandStatus::Stopped);
    require(!execution_error,"Metal draw GPU execution failed",CommandStatus::Stopped);
    const auto* pixels=static_cast<const std::byte*>(output.contents);
    for (std::uint32_t y=0;y<plan.target.height;++y)
        for (std::uint32_t x=0;x<plan.target.width;x+=4) {
            const auto offset=value(pixel_byte_offset(plan.target,x,y));
            const auto count=std::min(4u,plan.target.width-x);
            require(memory.write(plan.target.target_address+offset,
                {pixels+(std::size_t(y)*plan.target.width+x)*4,count*4}).ok(),
                "Metal draw guest-memory writeback failed",CommandStatus::MemoryFault);
        }
    if (context.statistics) { ++context.statistics->completed_draws;
        context.statistics->written_pixels+=std::uint64_t(plan.target.width)*plan.target.height; }
    return {};
}
}  // namespace

graphics::CommandResult validate_metal_draw_state(const graphics::RenderStop& snapshot) noexcept {
    try {
        static_cast<void>(admit(snapshot));
        return {};
    } catch (const graphics::CommandResult& error) { return error;
    } catch (...) {
        return {graphics::CommandStatus::Unsupported, "Metal draw state validation failed"};
    }
}

graphics::CommandResult execute_metal_draw(void* raw,const graphics::RenderStop& snapshot,
                                          const graphics::GpuAddressSpace& memory) noexcept {
    @autoreleasepool {
        try {
            require(raw!=nullptr,"Metal draw context is missing");
            return execute(*static_cast<MetalDrawContext*>(raw),snapshot,memory);
        } catch (const graphics::CommandResult& error) { return error;
        } catch (const std::bad_alloc&) {
            return {graphics::CommandStatus::BudgetExceeded,"Metal draw staging allocation failed"};
        } catch (...) { return {graphics::CommandStatus::Unsupported,"Metal draw provider or backend failed"}; }
    }
}
}  // namespace armigo::platform
