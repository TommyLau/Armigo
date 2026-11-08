// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "stage_internal.h"
#include "../resources/detail.h"
#include "../handle_program.h"
#include <array>
#include <tuple>

namespace armigo::shader::pipeline::msl {
namespace {
bool same_io(const IoUse& a, const IoUse& b) {
    return a.semantic == b.semantic && a.location == b.location && a.mask == b.mask && a.type == b.type &&
        a.interpolation == b.interpolation && a.access == b.access && a.vertex_format == b.vertex_format &&
        a.vertex_stream == b.vertex_stream && a.byte_offset == b.byte_offset && a.byte_stride == b.byte_stride &&
        a.vertex_class == b.vertex_class;
}
bool same_use(const ResourceUse& a, const ResourceUse& b) {
    if (a.index() != b.index()) return false;
    if (auto* x = std::get_if<BufferUse>(&a)) {
        const auto& y = std::get<BufferUse>(b);
        if (x->source.index() != y.source.index() || x->access != y.access || x->size.kind != y.size.kind ||
            x->size.bytes != y.size.bytes || x->size.alignment != y.size.alignment ||
            x->size.constant_buffer_policy != y.size.constant_buffer_policy) return false;
        if (auto* s = std::get_if<ConstantBufferSource>(&x->source)) return s->slot == std::get<ConstantBufferSource>(y.source).slot;
        if (auto* s = std::get_if<StorageBufferSource>(&x->source)) return s->descriptor == std::get<StorageBufferSource>(y.source).descriptor;
        if (auto* s = std::get_if<VertexStreamSource>(&x->source)) return s->slot == std::get<VertexStreamSource>(y.source).slot;
        const auto& s = std::get<SupportBufferSource>(x->source);
        const auto& t = std::get<SupportBufferSource>(y.source);
        return s.schema == t.schema && s.version == t.version;
    }
    if (auto* x = std::get_if<TextureUse>(&a)) {
        const auto& y = std::get<TextureUse>(b);
        return x->kind == y.kind && x->coherent == y.coherent && x->atomic == y.atomic && x->key == y.key &&
            x->descriptor == y.descriptor && x->access == y.access && x->array_length == y.array_length &&
            x->dynamically_indexed == y.dynamically_indexed && x->modes == y.modes && x->pool == y.pool;
    }
    const auto& x = std::get<SamplerUse>(a); const auto& y = std::get<SamplerUse>(b);
    return x.key == y.key && x.comparison == y.comparison && x.normalized_coordinates == y.normalized_coordinates &&
        x.array_length == y.array_length && x.pool == y.pool;
}
std::string interpolation(Interpolation i, ScalarType scalar_type) {
    if (i.mode == InterpolationMode::flat) {
        require(i.location == InterpolationLocation::center, "Flat input cannot request an interpolation location", DiagnosticCode::unsupported_modifier);
        return ", flat";
    }
    require(scalar_type == ScalarType::f32 || scalar_type == ScalarType::f16,
            "Integer varyings require flat interpolation", DiagnosticCode::unsupported_modifier);
    std::string location;
    switch (i.location) {
    case InterpolationLocation::center: location = "center"; break;
    case InterpolationLocation::centroid: location = "centroid"; break;
    case InterpolationLocation::sample: location = "sample"; break;
    default: fail("Offset interpolation needs a pull-model attribute", DiagnosticCode::unsupported_capability);
    }
    require(i.mode == InterpolationMode::perspective || i.mode == InterpolationMode::linear, "Unknown interpolation mode");
    return ", " + location + (i.mode == InterpolationMode::linear ? "_no_perspective" : "_perspective");
}
}
void validate_plan(const StructuredProgram& p, const ResourcePlan& plan, const EmitOptions& options) {
    require(p.ir.stage <= Stage::compute, "Unsupported shader stage", DiagnosticCode::unsupported_capability);
    if (p.ir.stage == Stage::compute)
        take(resources::check_requirements(p.ir, p.requirements, p.profile, Phase::emit_msl));
    require(plan.compute == p.requirements.compute, "Binding plan compute launch differs from requirements");
    require(plan.stage == p.ir.stage && plan.profile == p.profile, "Binding plan stage/profile disagrees with structured program");
    require(!plan.profile.identity.empty() && plan.profile.version && plan.profile.buffer_slots && plan.profile.texture_slots && plan.profile.sampler_slots,
            "Metal capability profile is not explicit", DiagnosticCode::missing_fact);
    require(plan.reflection_version == 3, "Unsupported reflection version", DiagnosticCode::unsupported_capability);
    require(plan.dependencies == p.ir.dependencies, "Binding plan dependencies differ from structured program");
    require(plan.fragment_position == FragmentPositionConvention::unspecified || plan.fragment_position == options.fragment_position, "Emission position convention differs from binding plan", DiagnosticCode::conflicting_fact);
    require(options.fragment_position <= FragmentPositionConvention::metal_position_w_matches_guest_position_w, "Unknown position convention");
    require(p.ir.stage == Stage::fragment || options.fragment_position == FragmentPositionConvention::unspecified, "Fragment convention used by vertex stage");
    require(p.ir.stage == Stage::fragment || !options.fragment_output_types_from_render_targets,
            "Render-target fragment output types used by a non-fragment stage");
    require(!options.emulate_depth_bias || (p.ir.stage == Stage::fragment && plan.profile.buffer_slots > 30),
            "Shader depth bias requires a fragment stage and Metal buffer 30",
            DiagnosticCode::unsupported_capability);
    require(plan.resources.size() <= p.ir.limits.resources && plan.io.size() <= p.ir.limits.resources &&
            plan.dependencies.size() <= p.ir.limits.dependencies, "Resource plan exceeds limits", DiagnosticCode::limit_exceeded);
    require(plan.resources.size() == p.requirements.uses.size(), "Resource plan does not cover requirements");
    require(plan.io.size() == p.requirements.io.size(), "IO plan does not cover requirements");
    for (std::size_t i = 0; i < plan.io.size(); ++i) require(same_io(plan.io[i], p.requirements.io[i]), "IO plan changes a requirement");
    std::set<std::uint32_t> ids;
    std::set<std::pair<MetalNamespace, unsigned>> occupied;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> member_ranges;
    std::uint64_t texture_count = 0, sampler_count = 0;
    const auto native_stream_count = p.ir.stage == Stage::vertex
        ? std::count_if(plan.io.begin(), plan.io.end(), [](const auto& io) {
              return io.semantic == IoSemantic::generic && io.access == Access::read &&
                   io.vertex_format && !io.vertex_stream;
          }) : 0;
    require(native_stream_count <= std::max(0, int(plan.profile.buffer_slots) - int(constant_buffer_slot_count)),
            "Native vertex streams exceed the Metal buffer namespace");
    auto native_stream_index = [&](std::uint32_t index) {
        return index >= constant_buffer_slot_count &&
               index - constant_buffer_slot_count < native_stream_count;
    };
    if (plan.argument_buffer) {
        const auto& layout = *plan.argument_buffer;
        require(layout.abi_version == 1 && layout.alignment == 8 && layout.byte_extent &&
                layout.byte_extent % 8 == 0 && layout.buffer_index < plan.profile.buffer_slots,
                "Invalid stage argument-buffer layout");
        require(!native_stream_index(layout.buffer_index),
                "Argument buffer overlaps a native vertex stream");
        occupied.insert({MetalNamespace::buffer, layout.buffer_index});
    }
    unsigned constant_buffers = 0, support_buffers = 0;
    std::set<unsigned> constant_slots;
    for (const auto& r : plan.resources) {
        require(r.resource.value < p.requirements.uses.size() && ids.insert(r.resource.value).second, "Invalid or repeated planned resource");
        require(same_use(r.use, p.requirements.uses[r.resource.value]), "Binding plan changes a resource requirement");
        MetalNamespace space; unsigned count = 1, capacity;
        if (auto* t = std::get_if<TextureUse>(&r.use)) {
            require(t->kind <= ResourceTextureKind::storage_image && t->access <= Access::read_write &&
                    std::all_of(t->modes.begin(), t->modes.end(), [](auto mode) { return mode <= TextureMode::query_samples; }), "Invalid texture requirement enum");
            require(t->descriptor.scalar != ScalarType::f16 || plan.profile.supports_fp16, "FP16 texture is not supported by profile", DiagnosticCode::unsupported_capability);
            if (t->pool) {
                require(plan.profile.texture_pools && t->kind == ResourceTextureKind::sampled && t->access == Access::read &&
                        !t->coherent && !t->atomic && t->array_length == 1 && !t->dynamically_indexed &&
                        t->key == TextureKey{texture_pool_slot, 0} && t->descriptor.shape.dimension != TextureDimension::buffer,
                        "Invalid texture pool requirement");
                space = MetalNamespace::buffer; capacity = plan.profile.buffer_slots;
            } else { space = MetalNamespace::texture; count = t->array_length; capacity = plan.profile.texture_slots; texture_count += count; } }
        else if (auto* s = std::get_if<SamplerUse>(&r.use)) {
            if (s->pool) {
                require(plan.profile.texture_pools && s->array_length == 1 && s->key == TextureKey{texture_pool_slot, 0},
                        "Invalid sampler pool requirement");
                space = MetalNamespace::buffer; capacity = plan.profile.buffer_slots;
            } else { space = MetalNamespace::sampler; count = s->array_length; capacity = plan.profile.sampler_slots; sampler_count += count; } }
        else {
            const auto& buffer = std::get<BufferUse>(r.use);
            require(buffer.access <= Access::read_write && buffer.size.kind <= SizeKind::storage_descriptor, "Invalid buffer requirement enum");
            require(buffer.size.kind != SizeKind::storage_descriptor ||
                    std::holds_alternative<StorageBufferSource>(buffer.source),
                    "Runtime storage extent requires a storage descriptor source");
            if (const auto* cb = std::get_if<ConstantBufferSource>(&buffer.source)) {
                ++constant_buffers;
                require(cb->slot < constant_buffer_slot_count && constant_slots.insert(cb->slot).second &&
                        buffer.access == Access::read && buffer.size.kind == SizeKind::constant_buffer_window &&
                        buffer.size.constant_buffer_policy == ConstantBufferPolicy::runtime_zero_words_v1 &&
                        buffer.size.bytes >= 4 && buffer.size.bytes <= constant_buffer_window_bytes &&
                        buffer.size.bytes % 4 == 0 && buffer.size.alignment == 4,
                        "Invalid runtime constant-buffer read window");
            } else {
                require(buffer.size.kind != SizeKind::constant_buffer_window &&
                        buffer.size.constant_buffer_policy == ConstantBufferPolicy::none,
                        "Constant-buffer policy on a non-constant buffer");
                if (const auto* support = std::get_if<SupportBufferSource>(&buffer.source)) {
                    const bool guest = support->schema == guest_memory_schema;
                    if (!guest) ++support_buffers;
                    require(guest ? support->version == guest_memory_version && buffer.size.bytes == guest_memory_bytes
                                  : support->schema == constant_buffer_support_schema &&
                                    support->version == constant_buffer_support_version &&
                                    buffer.size.bytes == constant_buffer_support_bytes,
                            "Unsupported support-buffer ABI", DiagnosticCode::unsupported_capability);
                    require(buffer.access == (guest ? Access::read_write : Access::read) &&
                            buffer.size.kind == SizeKind::accessed_extent &&
                            buffer.size.alignment == (guest ? guest_memory_alignment : 4u),
                            "Unsupported support-buffer ABI", DiagnosticCode::unsupported_capability);
                }
            }
            space = MetalNamespace::buffer; capacity = plan.profile.buffer_slots;
        }
        if (const auto* direct = std::get_if<MetalRange>(&r.location)) {
            require(space == MetalNamespace::buffer || !plan.argument_buffer,
                    "Argument-buffer stage cannot mix direct textures or samplers");
            require(count && count == direct->count && space == direct->space &&
                    count <= capacity && direct->first <= capacity - count,
                    "Resource binding range is invalid");
            if (const auto* buffer = std::get_if<BufferUse>(&r.use);
                buffer && !std::holds_alternative<VertexStreamSource>(buffer->source))
                require(!native_stream_index(direct->first),
                        "Ordinary buffer overlaps a native vertex stream");
            for (unsigned i = 0; i < count; ++i)
                require(occupied.insert({space, direct->first + i}).second, "Resource binding ranges overlap");
        } else {
            const auto& member = std::get<ArgumentBufferMember>(r.location);
            require(plan.argument_buffer && space != MetalNamespace::buffer && count && member.count == count &&
                    member.stride == 8 && member.byte_offset % 8 == 0,
                    "Invalid argument-buffer member tag, count, or alignment");
            const auto extent = std::uint64_t(count) * member.stride;
            require(extent <= plan.argument_buffer->byte_extent &&
                    member.byte_offset <= plan.argument_buffer->byte_extent - extent,
                    "Argument-buffer member exceeds its container");
            member_ranges.emplace_back(member.byte_offset, static_cast<std::uint32_t>(extent));
        }
    }
    require(plan.argument_buffer.has_value() ==
            (texture_count > plan.profile.texture_slots || sampler_count > plan.profile.sampler_slots),
            "Stage argument-buffer selection disagrees with resource totals");
    if (plan.argument_buffer) {
        std::sort(member_ranges.begin(), member_ranges.end());
        std::uint64_t end = 0;
        for (const auto& [offset, extent] : member_ranges) {
            require(offset == end, "Argument-buffer members overlap or leave an ABI gap");
            end += extent;
        }
        require(end == plan.argument_buffer->byte_extent, "Argument-buffer extent differs from member layout");
    }
    require(support_buffers == (constant_buffers ? 1u : 0u),
            "Constant-buffer windows require exactly one size-table support buffer");
    const bool instance_id = p.ir.stage == Stage::vertex &&
        std::any_of(plan.io.begin(), plan.io.end(), [](const auto& io) {
            return io.access == Access::read && io.semantic == IoSemantic::instance_id;
        });
    const bool height = p.ir.stage == Stage::fragment &&
        std::any_of(plan.io.begin(), plan.io.end(), [](const auto& io) {
            return io.access == Access::read && io.semantic == IoSemantic::render_target_height;
        });
    require(plan.draw_support_buffer_index.has_value() == (instance_id || height),
            "Vertex InstanceId and the fragment render-target height require the draw-parameter support ABI");
    if (plan.draw_support_buffer_index) {
        const auto index = *plan.draw_support_buffer_index;
        require(index < plan.profile.buffer_slots && !native_stream_index(index) &&
                occupied.insert({MetalNamespace::buffer, index}).second,
                "Draw support buffer overlaps a planned resource or exceeds the Metal namespace",
                DiagnosticCode::conflicting_fact);
    }
    if (plan.vertex_fetch_buffer_index) {
        const auto index = *plan.vertex_fetch_buffer_index;
        require(p.ir.stage == Stage::vertex && native_stream_count == 0, "Vertex fetch requires a vertex stage "
                "whose inputs carry derived classes", DiagnosticCode::conflicting_fact);
        require(index < plan.profile.buffer_slots && occupied.insert({MetalNamespace::buffer, index}).second,
                "Vertex fetch table overlaps a planned resource or exceeds the Metal namespace",
                DiagnosticCode::conflicting_fact);
    }
    require(!options.emulate_depth_bias || !occupied.contains({MetalNamespace::buffer, 30}),
            "Shader depth bias buffer 30 overlaps a planned resource",
            DiagnosticCode::conflicting_fact);
    if (plan.custom_border) {
        require((plan.stage == Stage::fragment || plan.stage == Stage::vertex) &&
                !plan.custom_border->pairs.empty() &&
                plan.custom_border->buffer_index < plan.profile.buffer_slots &&
                !native_stream_index(plan.custom_border->buffer_index) &&
                !(options.emulate_depth_bias && plan.custom_border->buffer_index == 30) &&
                occupied.insert({MetalNamespace::buffer, plan.custom_border->buffer_index}).second,
                "Custom-border support overlaps a planned buffer");
    }
    std::set<std::tuple<bool, IoSemantic, unsigned, unsigned>> io_components;
    for (const auto& io : plan.io) {
        require(!options.emulate_depth_bias || io.semantic != IoSemantic::fragment_depth,
                "Shader depth bias cannot replace a guest fragment-depth export",
                DiagnosticCode::unsupported_capability);
        require(io.semantic == IoSemantic::generic || io.semantic == IoSemantic::color || io.location == 0, "Builtin IO location must be zero");
        require(io.access != Access::read_write && io.access <= Access::read_write, "Stage IO needs one direction");
        require(io.interpolation.mode <= InterpolationMode::linear && io.interpolation.location <= InterpolationLocation::offset, "Invalid interpolation mode/location");
        require(io.type.components >= 1 && io.type.components <= 4 && io.mask && !(io.mask >> io.type.components), "Invalid IO mask/type");
        require(io.type.scalar != ScalarType::f16 || plan.profile.supports_fp16, "FP16 IO is not supported by profile", DiagnosticCode::unsupported_capability);
        if (io.vertex_class)
            require(plan.stage == Stage::vertex && io.access == Access::read &&
                    io.semantic == IoSemantic::generic && io.type == IoType{vertex_class_scalar(*io.vertex_class), 4},
                    "Vertex input class disagrees with its IO type");
        for (unsigned c = 0; c < 4; ++c) if (io.mask & (1u << c))
            require(io_components.insert({io.access == Access::write, io.semantic, io.location, c}).second, "Overlapping IO requirements");
        if (plan.stage == Stage::fragment && io.access == Access::read &&
            ((io.semantic == IoSemantic::generic && io.interpolation.mode == InterpolationMode::perspective) ||
             ((io.semantic == IoSemantic::position || io.semantic == IoSemantic::fragment_coord) && (io.mask & 8))))
            require(options.fragment_position == FragmentPositionConvention::metal_position_w_matches_guest_position_w,
                    "PositionW and perspective inputs require an explicit host position convention", DiagnosticCode::missing_fact);
        if (options.fragment_output_types_from_render_targets && io.access == Access::write &&
            io.semantic == IoSemantic::color)
            require(io.type == IoType{ScalarType::u32, 4},
                    "Adaptive fragment colors require uint4 raw-register storage",
                    DiagnosticCode::conflicting_fact);
    }
}
std::string resource_argument(const PlannedResource& r) {
    const auto name = "resource" + number(r.resource.value);
    const auto* direct = std::get_if<MetalRange>(&r.location);
    const auto index = direct ? number(direct->first) : std::string{};
    if (const auto* b = std::get_if<BufferUse>(&r.use)) {
        require(direct, "Ordinary buffers must use direct bindings");
        require(b->size.bytes && b->size.alignment >= 4 && std::has_single_bit(b->size.alignment), "Buffer extent/alignment is not proven", DiagnosticCode::missing_fact);
        if (const auto* cb = std::get_if<ConstantBufferSource>(&b->source)) {
            require(cb->slot < 18 && b->access == Access::read, "Invalid constant buffer requirement");
            return "constant uint* " + name + " [[buffer(" + index + ")]]";
        }
        if (std::holds_alternative<SupportBufferSource>(b->source))
            return std::string(b->access == Access::read ? "constant uint* " : "device uint* ") + name + " [[buffer(" + index + ")]]";
        require(std::holds_alternative<StorageBufferSource>(b->source),
                "Vertex-stream resources require an explicit ABI adapter", DiagnosticCode::unsupported_capability);
        return std::string(b->access == Access::read ? "const device uint* " : "device uint* ") + name + " [[buffer(" + index + ")]]";
    }
    if (const auto* s = std::get_if<SamplerUse>(&r.use)) {
        if (s->pool) {
            require(direct && direct->space == MetalNamespace::buffer, "A sampler pool must use a direct buffer binding");
            return "constant ArmigoPool" + number(r.resource.value) + "& " + name + " [[buffer(" + index + ")]]";
        }
        std::string type = s->array_length == 1 ? "sampler" : "array<sampler, " + number(s->array_length) + ">";
        return type + " " + name + (direct ? " [[sampler(" + index + ")]]" : "");
    }
    const auto& t = std::get<TextureUse>(r.use);
    if (t.pool) {
        require(direct && direct->space == MetalNamespace::buffer, "A texture pool must use a direct buffer binding");
        return "constant ArmigoPool" + number(r.resource.value) + "& " + name + " [[buffer(" + index + ")]]";
    }
    auto type = texture_type(t);
    if (t.array_length > 1) type = "array<" + type + ", " + number(t.array_length) + ">";
    return type + " " + name + (direct ? " [[texture(" + index + ")]]" : "");
}
// A pool table (the project documentation): n, a reserved word, the read
// bitmap's address and one declared element; reads index past it through
// the element's address, so Metal never encodes a whole ID range.
std::string pool_declarations(const ResourcePlan& plan) {
    std::string text;
    bool marks = false;
    for (const auto& r : plan.resources) {
        const auto* t = std::get_if<TextureUse>(&r.use);
        const auto* s = std::get_if<SamplerUse>(&r.use);
        if (!(t && t->pool) && !(s && s->pool)) continue;
        marks |= t != nullptr;
        text += "struct ArmigoPool" + number(r.resource.value) + " {\n    uint n;\n    uint reserved;\n    ulong seen;\n    array<" +
            (t ? texture_type(*t) : std::string("sampler")) + ", 1> e;\n};\n";
    }
    // Marks texture ID id read in the bitmap at seen, when there is one and
    // the ID is inside the table. The word is loaded first: a bit is set
    // once, so nearly every read costs a load rather than an atomic.
    if (marks) text +=
        "static inline __attribute__((always_inline)) uint armigo_pool_read(ulong seen, uint id, uint n) {\n"
        "    if (seen != 0 && id < n) {\n"
        "        device atomic_uint* word = reinterpret_cast<device atomic_uint*>(seen) + (id >> 5);\n"
        "        const uint bit = 1u << (id & 31u);\n"
        "        if ((atomic_load_explicit(word, memory_order_relaxed) & bit) == 0u)\n"
        "            atomic_fetch_or_explicit(word, bit, memory_order_relaxed);\n"
        "    }\n"
        "    return min(id, n);\n"
        "}\n";
    return text;
}
std::string texture_type(const TextureUse& t) {
    const auto& d = t.descriptor; const auto shape = d.shape;
    require((d.format == ComponentFormat::unsigned_integer && d.scalar == ScalarType::u32) ||
            (d.format == ComponentFormat::signed_integer && d.scalar == ScalarType::s32) ||
            ((d.format == ComponentFormat::unorm || d.format == ComponentFormat::snorm || d.format == ComponentFormat::floating || d.format == ComponentFormat::depth) &&
             (d.scalar == ScalarType::f32 || d.scalar == ScalarType::f16)), "Texture format and component type disagree");
    require(!t.coherent, "Coherent image ABI is not supported", DiagnosticCode::unsupported_capability);
    // An atomic image is a 2D integer storage texture with read-write access.
    // The atomic methods are members of that access, and there is no separate
    // atomic access qualifier. A buffer or a float image has no atomic access
    // of this form, and a coherent image stays refused.
    if (t.atomic)
        require(t.kind == ResourceTextureKind::storage_image && !shape.array && !shape.multisample &&
                shape.dimension == TextureDimension::d2 &&
                (d.scalar == ScalarType::u32 || d.scalar == ScalarType::s32),
                "Atomic image ABI is only a 2D integer storage image", DiagnosticCode::unsupported_capability);
    require(d.scalar == ScalarType::f32 || d.scalar == ScalarType::s32 || d.scalar == ScalarType::u32 || d.scalar == ScalarType::f16,
            "Texture component type is not supported", DiagnosticCode::unsupported_capability);
    require(!d.comparison || (d.format == ComponentFormat::depth && d.scalar == ScalarType::f32), "Comparison texture must use depth float components");
    std::string type = d.format == ComponentFormat::depth ? "depth" : "texture";
    switch (shape.dimension) {
    case TextureDimension::d1: type += "1d"; require(!shape.multisample, "Multisample 1D texture is invalid"); break;
    case TextureDimension::d2: type += "2d"; break;
    case TextureDimension::d3: type += "3d"; require(!shape.array && !shape.multisample, "Invalid 3D texture shape"); break;
    case TextureDimension::cube: type += "cube"; require(!shape.multisample, "Multisample cube is invalid"); break;
    case TextureDimension::buffer: type += "_buffer"; require(!shape.array && !shape.multisample, "Invalid texture buffer shape"); break;
    default: fail("Invalid texture dimension");
    }
    require(d.format != ComponentFormat::depth || shape.dimension == TextureDimension::d2 || shape.dimension == TextureDimension::cube,
            "Depth texture shape is unsupported", DiagnosticCode::unsupported_capability);
    if (shape.multisample) type += "_ms";
    if (shape.array) type += "_array";
    type += "<" + scalar(d.scalar);
    if (t.kind == ResourceTextureKind::storage_image) {
        require(d.format != ComponentFormat::depth && shape.dimension != TextureDimension::cube && !shape.multisample,
                "Storage image shape is unsupported", DiagnosticCode::unsupported_capability);
        if (t.atomic) type += ", access::read_write";
        else type += t.access == Access::read ? ", access::read" :
            t.access == Access::write ? ", access::write" : ", access::read_write";
    } else {
        require(t.access == Access::read, "Sampled texture cannot be writable");
        if (shape.multisample || shape.dimension == TextureDimension::buffer) type += ", access::read";
    }
    type += ">";
    return type;
}
std::string io_declarations(const ResourcePlan& plan, bool output,
                            bool fragment_output_types_from_render_targets,
                            bool emulate_depth_bias, bool fixed_point_size, bool invariant_position) {
    std::string text;
    if (output && fragment_output_types_from_render_targets) {
        std::set<std::uint16_t> locations;
        for (const auto& io : plan.io) {
            if (io.access != Access::write || io.semantic != IoSemantic::color ||
                !locations.insert(io.location).second) continue;
            const auto location = number(io.location);
            const auto name = io_name(io.semantic, io.location);
            text += "constant uint armigo_" + name + "_type [[function_constant(" + location + ")]];\n";
            text += "constant bool armigo_" + name + "_is_f32 = armigo_" + name + "_type == 0u;\n";
            text += "constant bool armigo_" + name + "_is_u32 = armigo_" + name + "_type == 1u;\n";
            text += "constant bool armigo_" + name + "_is_s32 = armigo_" + name + "_type == 2u;\n";
        }
    }
    if (output && emulate_depth_bias)
        text += "constant bool armigo_has_depth [[function_constant(8)]];\n";
    text += "struct Armigo" + std::string(output ? "Output" : "Input") + " {\n";
    std::set<std::string> names;
    for (const auto& io : plan.io) {
        if ((io.access == Access::write) != output) continue;
        const auto name = io_name(io.semantic, io.location);
        auto field = [&](std::string type, std::string member, std::string attribute) {
            require(names.insert(member).second, "IO declaration requires incompatible duplicate fields");
            text += "    " + type + " " + member + (attribute.empty() ? "" : " [[" + attribute + "]]") + ";\n";
        };
        const auto type = scalar(io.type.scalar) + (io.type.components == 1 ? "" : number(io.type.components));
        switch (io.semantic) {
        case IoSemantic::generic:
            if (plan.stage == Stage::vertex && !output) {
                // A fact-less package declares the code-derived class instead
                // of the pipeline-state format; legalize aligned io.type with
                // either, so float4/uint4/int4 comes from io.type directly.
                require(io.vertex_format.has_value() || io.vertex_class.has_value(),
                        "Vertex input has neither an attribute format fact nor a derived class",
                        DiagnosticCode::missing_fact);
                // A fetching stage fills its inputs from the fetch table.
                field(type, name, plan.vertex_fetch_buffer_index ? "" : "attribute(" + number(io.location) + ")");
            } else {
                require((plan.stage == Stage::vertex && output) || (plan.stage == Stage::fragment && !output), "Invalid generic IO direction");
                // Scalar user fields keep component-specific interpolation independent
                // of which neighbouring components the other stage uses.
                for (unsigned c = 0; c < io.type.components; ++c) if (io.mask & (1u << c)) {
                    auto suffix = "_" + number(c);
                    field(scalar(io.type.scalar), name + suffix, "user(locn" + number(io.location) + suffix + ")" +
                          (output ? "" : interpolation(io.interpolation, io.type.scalar)));
                }
            }
            break;
        case IoSemantic::position: case IoSemantic::fragment_coord:
            require(io.type == IoType{ScalarType::f32, 4} && ((plan.stage == Stage::vertex && output && io.semantic == IoSemantic::position) ||
                    (plan.stage == Stage::fragment && !output)), "Invalid position IO");
            field("float4", name, output && invariant_position ? "position, invariant" : "position"); break;
        case IoSemantic::color:
            require(plan.stage == Stage::fragment && output && io.location < 8 &&
                    (io.type.scalar == ScalarType::f32 || io.type.scalar == ScalarType::u32 || io.type.scalar == ScalarType::s32 || io.type.scalar == ScalarType::f16), "Invalid color output");
            if (fragment_output_types_from_render_targets) {
                require(io.type == IoType{ScalarType::u32, 4},
                        "Adaptive fragment colors require uint4 raw-register storage");
                const auto location = number(io.location);
                field("float4", name + "_f32", "color(" + location + "), function_constant(armigo_" + name + "_is_f32)");
                field("uint4", name + "_u32", "color(" + location + "), function_constant(armigo_" + name + "_is_u32)");
                field("int4", name + "_s32", "color(" + location + "), function_constant(armigo_" + name + "_is_s32)");
            } else {
                field(type, name, "color(" + number(io.location) + ")");
            }
            break;
        case IoSemantic::fragment_depth:
            require(plan.stage == Stage::fragment && output && io.type == IoType{ScalarType::f32, 1}, "Invalid fragment depth");
            field(type, name, "depth(any)"); break;
        case IoSemantic::thread_id: case IoSemantic::workgroup_id: case IoSemantic::subgroup_lane_id:
        case IoSemantic::quad_lane_id: {
            const bool scalar_lane = io.semantic == IoSemantic::subgroup_lane_id ||
                                     io.semantic == IoSemantic::quad_lane_id;
            const bool stage_ok = io.semantic == IoSemantic::quad_lane_id ? plan.stage == Stage::fragment :
                io.semantic == IoSemantic::subgroup_lane_id ? plan.stage != Stage::vertex :
                plan.stage == Stage::compute;
            require(stage_ok && !output && io.location == 0 &&
                    io.type == IoType{ScalarType::u32, static_cast<std::uint8_t>(scalar_lane ? 1 : 3)},
                    "Invalid compute builtin");
            require(names.insert(name).second, "Repeated compute builtin");
            break;
        }
        case IoSemantic::y_direction:
            // The S2R y-direction word is a namespace-level function-constant
            // selection emitted by the stage declaration section, never a
            // stage-in field.
            require(!output && (plan.stage == Stage::vertex || plan.stage == Stage::fragment) &&
                    io.location == 0 && io.mask == 1 && io.type == IoType{ScalarType::u32, 1},
                    "Invalid y-direction input");
            require(names.insert(name).second, "Repeated y-direction input");
            break;
        case IoSemantic::render_target_height:
            // A local the entry reads from the draw-parameter record.
            require(!output && plan.stage == Stage::fragment && io.location == 0 && io.mask == 1 &&
                    io.type == IoType{ScalarType::u32, 1}, "Invalid render-target height input");
            require(names.insert(name).second, "Repeated render-target height input");
            break;
        case IoSemantic::vertex_id: case IoSemantic::instance_id:
            require(plan.stage == Stage::vertex && !output && io.type == IoType{ScalarType::u32, 1} &&
                    io.location == 0 && io.mask == 1, "Invalid vertex builtin");
            // Vertex IDs are direct entry parameters, never stage-in fields.
            require(names.insert(name).second, "Repeated vertex builtin");
            break;
        case IoSemantic::front_facing:
            require(plan.stage == Stage::fragment && !output && io.type == IoType{ScalarType::boolean, 1}, "Invalid front-facing input");
            field(type, name, "front_facing"); break;
        case IoSemantic::point_coord:
            require(plan.stage == Stage::fragment && !output && io.type == IoType{ScalarType::f32, 2}, "Invalid point coordinates");
            field(type, name, "point_coord"); break;
        case IoSemantic::sample_id:
            require(plan.stage == Stage::fragment && !output && io.type == IoType{ScalarType::u32, 1}, "Invalid sample ID");
            field(type, name, "sample_id"); break;
        case IoSemantic::sample_mask:
            require(plan.stage == Stage::fragment && io.type == IoType{ScalarType::u32, 1}, "Invalid sample mask");
            field(type, name, "sample_mask"); break;
        case IoSemantic::point_size:
            require(plan.stage == Stage::vertex && output && io.type == IoType{ScalarType::f32, 1}, "Invalid point size");
            field(type, name, "point_size"); break;
        case IoSemantic::layer: case IoSemantic::viewport_index:
            require(plan.stage == Stage::vertex && output && io.type == IoType{ScalarType::u32, 1}, "Invalid vertex routing output");
            field(type, name, io.semantic == IoSemantic::layer ? "render_target_array_index" : "viewport_array_index"); break;
        default: fail("IO builtin requires an unsupported host adapter", DiagnosticCode::unsupported_capability);
        }
    }
    if (output && emulate_depth_bias)
        text += "    float armigo_depth [[depth(any), function_constant(armigo_has_depth)]];\n";
    // Synthesized outside plan.io (the plan mirrors the program requirements),
    // so a program without the 0x6c export still declares the point-size
    // output the entry prologue writes.
    if (output && fixed_point_size)
        text += "    float " + io_name(IoSemantic::point_size, 0) + " [[point_size]];\n";
    return text + "};\n";
}
} // namespace armigo::shader::pipeline::msl
