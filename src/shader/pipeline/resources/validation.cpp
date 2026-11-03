// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "detail.h"
#include <algorithm>
#include <limits>
#include <set>

namespace armigo::shader::pipeline::resources {
namespace {
class BoundProof {
public:
    BoundProof(const Program& program, std::uint64_t& work) : p(program), work(work), defs(p.values.size()), phis(p.values.size()) {
        for (const auto& op : p.operations) {
            for (const auto& output : op.outputs) {
                if (++work > p.limits.analysis_steps) return;
                if (const auto* value = std::get_if<ValueId>(&output); value && value->value < defs.size()) defs[value->value].push_back(&op);
            }
        }
        for (const auto& block : p.blocks) for (const auto& phi : block.phis)
            if (phi.output.value < phis.size()) phis[phi.output.value] = &phi;
    }
    std::optional<std::uint32_t> bound(const Operand& operand, unsigned depth = 0) {
        if (++work > p.limits.analysis_steps || depth > std::min(p.limits.nesting_depth, 256U)) return {};
        if (const auto* literal = std::get_if<LiteralBits>(&operand)) {
            if (literal->type == ValueType::word32 && literal->bits <= UINT32_MAX) return static_cast<std::uint32_t>(literal->bits);
            return {};
        }
        const auto* value = std::get_if<ValueId>(&operand);
        if (!value || value->value >= defs.size() || p.values[value->value].type != ValueType::word32) return {};
        std::optional<std::uint32_t> maximum;
        auto merge = [&](std::optional<std::uint32_t> next) {
            if (!next) return false;
            maximum = std::max(maximum.value_or(0), *next); return true;
        };
        for (const auto* op : defs[value->value]) if (!merge(operation(*op, depth + 1))) return {};
        if (const auto* phi = phis[value->value]) for (const auto& input : phi->inputs) {
            if (input.value != phi->output && !merge(bound(input.value, depth + 1))) return {};
        }
        return maximum;
    }
private:
    const Program& p;
    std::uint64_t& work;
    std::vector<std::vector<const Operation*>> defs;
    std::vector<const Phi*> phis;
    std::optional<std::uint32_t> operation(const Operation& op, unsigned depth) {
        if (++work > p.limits.analysis_steps || depth > std::min(p.limits.nesting_depth, 256U)) return {};
        if (op.outputs.size() != 1 || op.numeric.scalar != ScalarType::u32 ||
            (op.numeric.domain != Domain::bits && op.numeric.domain != Domain::unsigned_integer)) return {};
        if (op.opcode == Opcode::copy && op.inputs.size() == 1) return bound(op.inputs[0], depth + 1);
        if (op.opcode == Opcode::load) {
            const auto* m = std::get_if<MemoryInfo>(&op.info);
            if (!m || m->space != AddressSpace::constant_buffer || !m->constant_buffer_slot || m->element_bytes != 4 ||
                m->elements != 1 || m->dynamic_byte_offset || m->array_index || m->is_volatile) return {};
            const auto byte_offset = m->byte_offset & ~std::uint32_t{3};
            if (byte_offset > constant_buffer_window_bytes - 4) return {};
            const auto key = fact_query::constant_word(*m->constant_buffer_slot, byte_offset).key();
            for (const auto& d : p.dependencies) {
                if (++work > p.limits.analysis_steps) return {};
                if (d.key == key) if (const auto* bits = std::get_if<std::uint32_t>(&d.value)) return *bits;
            }
            return {};
        }
        if (op.opcode == Opcode::load_io) {
            const auto& io = std::get<IoInfo>(op.info);
            if (p.stage != Stage::compute) return {};
            if (io.semantic == IoSemantic::subgroup_lane_id) return 31;
            if (io.semantic == IoSemantic::thread_id && std::has_single_bit(io.component_mask))
                for (const auto& d : p.dependencies) if (d.key == FactKey{FactKind::compute_local_size})
                    if (const auto* size = std::get_if<ComputeLocalSize>(&d.value)) {
                        const auto c = std::countr_zero(unsigned(io.component_mask));
                        if (c < 3 && size->dimensions[c]) return size->dimensions[c] - 1;
                    }
        }
        if (op.inputs.size() != 2) return {};
        if (op.opcode == Opcode::bit_and || (op.opcode == Opcode::minimum && op.numeric.domain == Domain::unsigned_integer)) {
            for (unsigned side = 0; side < 2; ++side) if (const auto* literal = std::get_if<LiteralBits>(&op.inputs[side]);
                literal && literal->type == ValueType::word32 && literal->bits <= UINT32_MAX) {
                const auto other = bound(op.inputs[1 - side], depth + 1);
                return std::min(static_cast<std::uint32_t>(literal->bits), other.value_or(UINT32_MAX));
            }
        }
        if ((op.opcode == Opcode::shift_left || op.opcode == Opcode::shift_right) &&
            std::holds_alternative<LiteralBits>(op.inputs[1])) {
            const auto a = bound(op.inputs[0], depth + 1);
            if (!a) return {};
            const auto count = std::get<LiteralBits>(op.inputs[1]).bits & 31;
            const auto result = op.opcode == Opcode::shift_left ? std::uint64_t(*a) << count : *a >> count;
            if (result <= UINT32_MAX) return static_cast<std::uint32_t>(result);
            return {};
        }
        auto a = bound(op.inputs[0], depth + 1), b = bound(op.inputs[1], depth + 1);
        if (!a || !b) return {};
        if (op.opcode == Opcode::bit_and || op.opcode == Opcode::minimum) return std::min(*a, *b);
        std::uint64_t result;
        if (op.opcode == Opcode::add) result = std::uint64_t(*a) + *b;
        else if (op.opcode == Opcode::multiply) result = std::uint64_t(*a) * *b;
        else return {};
        if (result <= UINT32_MAX) return static_cast<std::uint32_t>(result);
        return {};
    }
};
}
Status check_profile(const MetalProfile& profile, Phase phase) {
    if (profile.identity.empty() || profile.version == 0 || profile.buffer_slots == 0 ||
        profile.texture_slots == 0 || profile.sampler_slots == 0)
        return diagnose(phase, DiagnosticCode::unsupported_capability, {}, "An explicit nonempty Metal profile is required");
    return std::monostate{};
}
bool valid_descriptor(const TextureDescriptor& d) {
    if (d.shape.dimension > TextureDimension::buffer || d.format > ComponentFormat::depth ||
        d.format_namespace.empty()) return false;
    if (d.scalar != ScalarType::f32 && d.scalar != ScalarType::u32 && d.scalar != ScalarType::s32) return false;
    if ((d.format == ComponentFormat::unsigned_integer && d.scalar != ScalarType::u32) ||
        (d.format == ComponentFormat::signed_integer && d.scalar != ScalarType::s32) ||
        (d.format != ComponentFormat::unsigned_integer && d.format != ComponentFormat::signed_integer &&
         d.scalar != ScalarType::f32)) return false;
    if (d.comparison && (d.format != ComponentFormat::depth || d.scalar != ScalarType::f32)) return false;
    if (d.shape.multisample && d.shape.dimension != TextureDimension::d2) return false;
    if (d.shape.array && (d.shape.dimension == TextureDimension::d3 || d.shape.dimension == TextureDimension::buffer)) return false;
    return true;
}
bool is_image(const Operation& op) {
    return op.opcode == Opcode::image_load || op.opcode == Opcode::image_store || op.opcode == Opcode::image_atomic;
}
bool needs_sampler(const Operation& op) {
    const auto& t = std::get<TextureInfo>(op.info);
    return (op.opcode == Opcode::texture_sample && t.mode != TextureMode::fetch) ||
           (op.opcode == Opcode::texture_query && t.mode == TextureMode::query_lod);
}
Status check_texture(const Operation& op, const TextureDescriptor& d, const MetalProfile& profile,
                     Stage stage, Phase phase) {
    auto fail = [&](std::string message, DiagnosticCode code = DiagnosticCode::unsupported_modifier) -> Status {
        return diagnose(phase, code, op.source, std::move(message));
    };
    if (!valid_descriptor(d)) return fail("Texture descriptor is semantically invalid", DiagnosticCode::invalid_fact);
    const auto& t = std::get<TextureInfo>(op.info);
    const bool sampling = needs_sampler(op);
    const bool query = op.opcode == Opcode::texture_query;
    const bool dimension_query = query && t.mode != TextureMode::query_lod;
    const bool image = is_image(op);
    if (t.mode > TextureMode::query_samples || (image && t.mode != TextureMode::fetch) ||
        (op.opcode == Opcode::texture_sample && t.mode > TextureMode::fetch) ||
        (query && t.mode < TextureMode::query_lod)) return fail("Texture mode disagrees with operation", DiagnosticCode::invalid_ir);
    if (t.component_mask == 0 || t.component_mask > 15 || t.gather_component > 3 || t.query_component > 3 ||
        (op.opcode != Opcode::image_store && op.outputs.size() != static_cast<std::size_t>(std::popcount(t.component_mask))) ||
        (op.opcode == Opcode::image_store && t.data.size() != static_cast<std::size_t>(std::popcount(t.component_mask))) ||
        (!image && !t.data.empty()) || (op.opcode == Opcode::image_load && !t.data.empty()))
        return fail("Texture components or data count are invalid", DiagnosticCode::invalid_ir);
    if ((op.opcode == Opcode::image_atomic && (!t.atomic || t.data.size() != (*t.atomic == AtomicKind::compare_exchange ? 2U : 1U))) ||
        (op.opcode != Opcode::image_atomic && t.atomic) || (t.atomic && *t.atomic > AtomicKind::compare_exchange))
        return fail("Texture atomic metadata is invalid", DiagnosticCode::invalid_ir);
    const unsigned dimensions = d.shape.dimension == TextureDimension::d1 || d.shape.dimension == TextureDimension::buffer ? 1 :
                                d.shape.dimension == TextureDimension::d2 ? 2 : 3;
    if (t.coordinates.size() != (dimension_query ? 0 : dimensions)) return fail("Texture coordinate count disagrees with descriptor");
    if (t.array_layer.has_value() != (d.shape.array && !dimension_query)) return fail("Texture array-layer role disagrees with descriptor");
    if (t.compare.has_value() != (sampling && d.comparison)) return fail("Texture comparison role disagrees with descriptor");
    if (t.sample_index.has_value() != (d.shape.multisample && !dimension_query)) return fail("Texture sample-index role disagrees with descriptor");
    if (sampling && (d.shape.multisample || d.shape.dimension == TextureDimension::buffer || d.scalar != ScalarType::f32))
        return fail("This texture shape or component type cannot be sampled");
    if (sampling && !d.normalized_coordinates)
        return fail("Unnormalized sampling requires a coordinate legalization pass");
    if (image && (d.shape.dimension == TextureDimension::cube || d.comparison)) return fail("Unsupported storage-image descriptor");
    if ((!query && op.numeric.scalar != d.scalar) || (dimension_query && op.numeric.scalar != ScalarType::u32) ||
        (t.mode == TextureMode::query_lod && op.numeric.scalar != ScalarType::f32))
        return fail("Texture component/result conversion must be explicit");
    if (t.mode == TextureMode::query_lod && !profile.supports_query_lod)
        return fail("Metal profile does not support query LOD", DiagnosticCode::unsupported_capability);
    if (t.mode == TextureMode::query_samples && !d.shape.multisample) return fail("Sample-count query requires a multisample texture");
    if (t.mode == TextureMode::query_levels && (d.shape.multisample || d.shape.dimension == TextureDimension::buffer))
        return fail("Level-count query requires a mipmapped texture shape");
    const bool level = t.mode == TextureMode::sample_level || t.mode == TextureMode::sample_bias ||
        (t.mode == TextureMode::fetch && !image && !d.shape.multisample && d.shape.dimension != TextureDimension::buffer) ||
        (t.mode == TextureMode::query_size && !d.shape.multisample && d.shape.dimension != TextureDimension::buffer);
    if (t.mode != TextureMode::gather && t.level_or_bias.has_value() != level) return fail("Texture level/bias role disagrees with mode");
    if (t.mode == TextureMode::gather && t.level_or_bias) return fail("Explicit-level gather has no proven Metal lowering");
    const auto gradients = t.mode == TextureMode::sample_gradients ? dimensions : 0U;
    if (t.gradients_x.size() != gradients || t.gradients_y.size() != gradients) return fail("Texture gradient roles disagree with mode");
    if (t.lod_clamp) return fail("Texture LOD clamp has no proven lowering");
    // Outside the fragment stage the resolver has already rewritten implicit
    // and bias sampling into explicit level sampling, and a Metal gather always
    // reads the base level without derivatives, so gather is the only
    // derivative-sensitive mode still admissible here.
    if (stage != Stage::fragment && includes(required_effects(op), Effects::derivative) &&
        t.mode != TextureMode::gather)
        return fail("Derivative-sensitive texture operation requires fragment stage", DiagnosticCode::unsupported_capability);
    if (!t.offsets.empty()) {
        // One offset per axis is the common literal-offset form. A gather
        // without depth comparison may instead carry a per-texel offset set
        // (textureGatherOffsets), four offsets per
        // axis; Metal lowers it as one gather per offset.
        const bool per_texel = t.mode == TextureMode::gather && !t.compare &&
            t.offsets.size() == 4 * dimensions;
        if ((t.offsets.size() != dimensions && !per_texel) || dimension_query ||
            d.shape.dimension == TextureDimension::cube || image)
            return fail("Texture offset set has no proven lowering");
        for (auto index : t.offsets) {
            if (index.value >= op.inputs.size()) return fail("Texture offset input is out of range", DiagnosticCode::invalid_ir);
            const auto* bits = std::get_if<LiteralBits>(&op.inputs[index.value]);
            if (!bits || bits->type != ValueType::word32 ||
                (bits->bits > 7 && bits->bits < UINT32_MAX - 7))
                return fail("Texture offsets require a literal in the supported signed range");
        }
    }
    if (op.opcode == Opcode::image_atomic) {
        if (stage == Stage::fragment && !profile.supports_fragment_atomics)
            return fail("Metal profile does not support fragment image atomics", DiagnosticCode::unsupported_capability);
        if (d.scalar != ScalarType::u32 && d.scalar != ScalarType::s32) return fail("Image atomics require integer components");
    }
    return std::monostate{};
}

Status check_requirements(const Program& p, const ResourceRequirements& requirements,
                          const MetalProfile& profile, Phase phase) {
    auto fail = [&](std::string message, SourceLocation source = {}, DiagnosticCode code = DiagnosticCode::invalid_ir) -> Status {
        return diagnose(phase, code, source, std::move(message));
    };
    if (p.stage > Stage::compute) return fail("Invalid resource stage");
    if ((p.stage == Stage::compute) != requirements.compute.has_value())
        return fail("Compute stage and launch requirements disagree");
    if (requirements.compute) {
        if (!valid_compute_configuration(*requirements.compute)) return fail("Invalid compute launch requirements");
        bool local = false, shared = false;
        for (const auto& d : p.dependencies) {
            if (d.key == FactKey{FactKind::compute_local_size}) {
                const auto* size = std::get_if<ComputeLocalSize>(&d.value);
                if (!size || *size != requirements.compute->local_size) return fail("Compute local-size dependency mismatch");
                local = true;
            }
            if (d.key == FactKey{FactKind::compute_shared_memory_bytes}) {
                const auto* size = std::get_if<std::uint32_t>(&d.value);
                if (!size || *size != requirements.compute->shared_memory_bytes) return fail("Compute shared-memory dependency mismatch");
                shared = true;
            }
        }
        if (!local || !shared) return fail("Compute launch dependencies are missing", {}, DiagnosticCode::missing_fact);
    }
    if (p.operations.size() > p.limits.operations || p.values.size() > p.limits.values || p.blocks.size() > p.limits.blocks)
        return fail("Resource input exceeds IR limits", {}, DiagnosticCode::limit_exceeded);
    if (requirements.uses.size() > p.limits.resources || requirements.io.size() > p.limits.resources)
        return fail("Resource requirement count exceeds limit", {}, DiagnosticCode::limit_exceeded);
    std::uint64_t work = 0;
    auto charge = [&] { return ++work <= p.limits.analysis_steps; };
    BoundProof bounds(p, work);
    std::set<std::uint16_t> streams;
    std::set<std::uint8_t> constant_buffers;
    std::uint32_t constant_buffer_supports = 0;
    for (const auto& use : requirements.uses) {
        if (!charge()) return fail("Resource validation work exceeds limit", {}, DiagnosticCode::limit_exceeded);
        if (const auto* b = std::get_if<BufferUse>(&use)) {
            if (b->access > Access::read_write || b->size.kind > SizeKind::storage_descriptor ||
                b->size.constant_buffer_policy > ConstantBufferPolicy::runtime_zero_words_v1 ||
                b->size.bytes == 0 || !std::has_single_bit(b->size.alignment))
                return fail("Invalid buffer requirement");
            if (const auto* cb = std::get_if<ConstantBufferSource>(&b->source)) {
                if (cb->slot >= constant_buffer_slot_count || b->access != Access::read ||
                    b->size.kind != SizeKind::constant_buffer_window || b->size.alignment != 4 ||
                    b->size.bytes < 4 || b->size.bytes > constant_buffer_window_bytes || b->size.bytes % 4 != 0 ||
                    b->size.constant_buffer_policy != ConstantBufferPolicy::runtime_zero_words_v1 ||
                    !constant_buffers.insert(cb->slot).second)
                    return fail("Invalid or duplicate constant-buffer requirement");
            } else if (const auto* storage = std::get_if<StorageBufferSource>(&b->source)) {
                if (!valid_fact_key(fact_query::texture(storage->descriptor).key()) ||
                    b->size.kind == SizeKind::constant_buffer_window ||
                    b->size.constant_buffer_policy != ConstantBufferPolicy::none)
                    return fail("Invalid storage-buffer requirement");
                if (b->size.kind == SizeKind::storage_descriptor &&
                    (storage->descriptor.handle_byte_offset % 8 != 0 ||
                     storage->descriptor.handle_byte_offset > constant_buffer_window_bytes - 12 ||
                     b->size.bytes != 4 || b->size.alignment != 4))
                    return fail("Invalid runtime storage descriptor requirement");
            } else if (const auto* support = std::get_if<SupportBufferSource>(&b->source)) {
                ++constant_buffer_supports;
                if (support->schema != constant_buffer_support_schema ||
                    support->version != constant_buffer_support_version || b->access != Access::read ||
                    b->size.kind != SizeKind::accessed_extent || b->size.bytes != constant_buffer_support_bytes ||
                    b->size.alignment != 4 || b->size.constant_buffer_policy != ConstantBufferPolicy::none)
                    return fail("Constant-buffer support table has an invalid ABI");
            } else {
                const auto slot = std::get<VertexStreamSource>(b->source).slot;
                if (p.stage != Stage::vertex || b->access != Access::read ||
                    b->size.kind >= SizeKind::constant_buffer_window ||
                    b->size.constant_buffer_policy != ConstantBufferPolicy::none || !streams.insert(slot).second)
                    return fail("Invalid or duplicate vertex stream requirement");
            }
        } else if (const auto* t = std::get_if<TextureUse>(&use)) {
            if (!valid_fact_key(fact_query::texture(t->key).key()) || !valid_descriptor(t->descriptor) ||
                t->kind > ResourceTextureKind::storage_image || t->access > Access::read_write || t->array_length == 0 ||
                (t->kind == ResourceTextureKind::sampled && (t->access != Access::read || t->atomic || t->coherent)) ||
                (t->atomic && t->access != Access::read_write) || t->modes.empty() ||
                !std::is_sorted(t->modes.begin(), t->modes.end()) ||
                std::adjacent_find(t->modes.begin(), t->modes.end()) != t->modes.end() || t->modes.back() > TextureMode::query_samples)
                return fail("Invalid texture requirement");
        } else {
            const auto& s = std::get<SamplerUse>(use);
            if (!valid_fact_key(fact_query::texture(s.key).key()) || s.array_length == 0)
                return fail("Invalid sampler requirement");
        }
    }
    if ((!constant_buffers.empty() && constant_buffer_supports != 1) ||
        (constant_buffers.empty() && constant_buffer_supports != 0))
        return fail("Exactly one constant-buffer support table is required with constant-buffer resources");
    for (const auto& io : requirements.io) {
        if (!charge()) return fail("IO validation work exceeds limit", {}, DiagnosticCode::limit_exceeded);
        if (io.semantic > IoSemantic::quad_lane_id || io.access > Access::read_write || io.mask == 0 || io.mask > 15 ||
            io.type.scalar > ScalarType::boolean || io.type.components == 0 || io.type.components > 4 ||
            io.interpolation.mode > InterpolationMode::linear || io.interpolation.location > InterpolationLocation::offset)
            return fail("Invalid IO reflection");
        if (io.vertex_stream) {
            if (!streams.contains(*io.vertex_stream) || !io.vertex_format || io.byte_stride == 0)
                return fail("Vertex stream reflection lacks a matching buffer or layout");
            const auto format = io.vertex_format->format;
            const unsigned width = format <= VertexFormat::snorm8 ? 1 : format <= VertexFormat::f16 ? 2 : 4;
            if (format > VertexFormat::f32 || io.vertex_format->components == 0 || io.vertex_format->components > 4 ||
                std::uint64_t(io.byte_offset) + width * io.vertex_format->components > io.byte_stride)
                return fail("Vertex attribute exceeds its stream stride");
        } else if (io.byte_offset || io.byte_stride) return fail("Vertex layout has no stream identity");
    }
    for (const auto& op : p.operations) {
        if (!charge()) return fail("Resource validation work exceeds limit", {}, DiagnosticCode::limit_exceeded);
        const auto* schema = signature(op.opcode);
        if (!schema || !schema->inputs.accepts(op.inputs.size()) || !schema->outputs.accepts(op.outputs.size()) ||
            static_cast<std::size_t>(schema->info) != op.info.index() || !includes(op.effects, required_effects(op)) ||
            !valid_numeric_mode(op.numeric)) return fail("Resource operation has an invalid schema or effects", op.source);
        auto resource = [&](std::optional<ResourceId> id) -> const ResourceUse* {
            return id && id->value < requirements.uses.size() ? &requirements.uses[id->value] : nullptr;
        };
        if (const auto* m = std::get_if<MemoryInfo>(&op.info)) {
            if (!std::has_single_bit(m->element_bytes) || m->element_bytes > 8 || m->elements == 0 || m->elements > 4 ||
                !std::has_single_bit(m->alignment) || m->space > AddressSpace::shared || m->access > Access::read_write ||
                m->order > MemoryOrder::sequential || m->scope > MemoryScope::device)
                return fail("Memory width, alignment or access metadata is invalid", op.source);
            std::vector<bool> referenced(op.inputs.size());
            auto role = [&](InputIndex index) {
                if (index.value >= op.inputs.size()) return false;
                referenced[index.value] = true; return true;
            };
            auto type = [&](const Operand& operand) -> std::optional<ValueType> {
                if (const auto* value = std::get_if<ValueId>(&operand); value && value->value < p.values.size()) return p.values[value->value].type;
                if (const auto* literal = std::get_if<LiteralBits>(&operand)) return literal->type;
                return {};
            };
            for (auto index : {m->dynamic_byte_offset, m->array_index, m->global_address_high}) if (index &&
                (!role(*index) || type(op.inputs[index->value]) != ValueType::word32)) return fail("Memory address role is invalid", op.source);
            if (m->storage_address) for (auto index : {m->storage_address->base_low,
                    m->storage_address->base_high, m->storage_address->byte_size})
                if (!role(index) || type(op.inputs[index.value]) != ValueType::word32)
                    return fail("Storage descriptor role is invalid", op.source);
            for (const auto& target : m->storage_alternatives)
                for (auto index : {target.address.base_low, target.address.base_high, target.address.byte_size})
                    if (!m->storage_address || !role(index) || type(op.inputs[index.value]) != ValueType::word32)
                        return fail("Storage alternative descriptor role is invalid", op.source);
            if ((m->global_address_high || m->storage_address) && m->space != AddressSpace::storage)
                return fail("Global address metadata is attached to nonstorage memory", op.source);
            const auto data_type = m->element_bytes == 8 ? ValueType::word64 : ValueType::word32;
            for (auto index : m->data) if (!role(index) || type(op.inputs[index.value]) != data_type)
                return fail("Memory data role is invalid", op.source);
            for (const auto& output : op.outputs) if (!std::holds_alternative<ValueId>(output) || type(output) != data_type)
                return fail("Memory result storage is invalid", op.source);
            if (std::find(referenced.begin(), referenced.end(), false) != referenced.end()) return fail("Memory input has no role", op.source);
            if ((op.opcode == Opcode::load && (m->access != Access::read || op.outputs.size() != m->elements || !m->data.empty() || m->atomic)) ||
                (op.opcode == Opcode::store && (m->access == Access::read || m->data.size() != m->elements || m->atomic)) ||
                (op.opcode == Opcode::atomic && (!m->atomic || m->access != Access::read_write ||
                    m->data.size() != (*m->atomic == AtomicKind::compare_exchange ? 2U : 1U))))
                return fail("Memory operation disagrees with its access/data roles", op.source);
            if (m->atomic && *m->atomic > AtomicKind::compare_exchange) return fail("Invalid memory atomic kind", op.source);
            if ((m->space == AddressSpace::constant_buffer && (!m->constant_buffer_slot ||
                    *m->constant_buffer_slot >= constant_buffer_slot_count || m->access != Access::read)) ||
                (m->space != AddressSpace::constant_buffer && m->constant_buffer_slot)) return fail("Memory source slot is invalid", op.source);
            if ((m->space == AddressSpace::local || m->space == AddressSpace::shared) && m->resource)
                return fail("Local/shared memory has a buffer resource", op.source);
            if (m->space == AddressSpace::shared && op.opcode != Opcode::barrier && op.opcode != Opcode::memory_barrier) {
                if (!requirements.compute) return fail("Shared access has no compute allocation", op.source);
                // armigo-shared-guard-v1, re-derived independently: a nonliteral
                // offset without a legalized bound stays runtime-checked, and
                // emission lowers it byte-wise beside the in-range guard, so
                // neither an alignment proof nor a static bound is demanded.
                // Atomics keep strict proofs: a dropped RMW is not a defined
                // guest outcome.
                const bool guarded = m->dynamic_byte_offset && !m->dynamic_byte_offset_upper_bound && !m->atomic;
                if (!guarded && (m->byte_offset % m->element_bytes || (m->dynamic_byte_offset &&
                    !shared_address_aligned(p, op.inputs[m->dynamic_byte_offset->value], m->element_bytes, work))))
                    return fail("Shared memory alignment proof is missing", op.source, DiagnosticCode::unsupported_capability);
                std::optional<std::uint32_t> maximum = 0;
                if (m->dynamic_byte_offset) maximum = bounds.bound(op.inputs[m->dynamic_byte_offset->value]);
                if (!guarded && (!maximum || (m->dynamic_byte_offset_upper_bound && *m->dynamic_byte_offset_upper_bound < *maximum)))
                    return fail("Shared offset has no sufficient re-established proof", op.source, DiagnosticCode::missing_fact);
                if (guarded) {
                    const auto extent = std::uint64_t(m->byte_offset) +
                        std::uint64_t(m->element_bytes) * m->elements;
                    if (extent > requirements.compute->shared_memory_bytes)
                        return fail("Shared access exceeds allocation", op.source);
                    continue;
                }
                const auto extent = std::uint64_t(m->byte_offset) + m->dynamic_byte_offset_upper_bound.value_or(*maximum) +
                    std::uint64_t(m->element_bytes) * m->elements;
                if (extent > requirements.compute->shared_memory_bytes) return fail("Shared access exceeds allocation", op.source);
            }
            if (m->space != AddressSpace::constant_buffer && m->space != AddressSpace::storage) continue;
            if (op.opcode == Opcode::barrier || op.opcode == Opcode::memory_barrier) continue;
            const auto* use = resource(m->resource);
            const auto* b = use ? std::get_if<BufferUse>(use) : nullptr;
            if (!b || !covers(b->access, m->access)) return fail("Memory resource is missing or has incompatible access", op.source);
            if (m->space == AddressSpace::constant_buffer) {
                const auto* cb = std::get_if<ConstantBufferSource>(&b->source);
                if (!cb || m->constant_buffer_slot != cb->slot) return fail("Constant-buffer resource does not match its slot", op.source);
            } else if (!std::holds_alternative<StorageBufferSource>(b->source)) return fail("Storage resource has the wrong source kind", op.source);
            if (b->size.kind == SizeKind::storage_descriptor) {
                if (!storage_descriptor_inputs(p, op, std::get<StorageBufferSource>(b->source), work))
                    return fail("Runtime storage descriptor inputs do not match the binding", op.source,
                        work > p.limits.analysis_steps ? DiagnosticCode::limit_exceeded : DiagnosticCode::invalid_ir);
                for (const auto& target : m->storage_alternatives) {
                    const auto* alternative = resource(target.resource);
                    const auto* ab = alternative ? std::get_if<BufferUse>(alternative) : nullptr;
                    const auto* source = ab ? std::get_if<StorageBufferSource>(&ab->source) : nullptr;
                    if (!source || ab->size.kind != SizeKind::storage_descriptor || !covers(ab->access, m->access) ||
                        !storage_address_inputs(p, op, target.address, *source, work))
                        return fail("Storage alternative does not match its runtime descriptor binding", op.source,
                            work > p.limits.analysis_steps ? DiagnosticCode::limit_exceeded : DiagnosticCode::invalid_ir);
                }
                continue;
            }
            if (!m->storage_alternatives.empty())
                return fail("Storage alternatives require runtime descriptor bindings", op.source);
            if (m->storage_address || m->global_address_high)
                return fail("Global address requires a runtime descriptor binding", op.source);
            std::optional<std::uint32_t> maximum;
            if (m->array_index) return fail("Memory arrays lack a source/stride contract", op.source);
            if (m->dynamic_byte_offset) {
                if (m->dynamic_byte_offset->value >= op.inputs.size()) return fail("Memory offset role is out of range", op.source);
                maximum = bounds.bound(op.inputs[m->dynamic_byte_offset->value]);
                if (work > p.limits.analysis_steps)
                    return fail("Memory offset analysis exceeds limit", op.source, DiagnosticCode::limit_exceeded);
            }
            if (m->space == AddressSpace::constant_buffer) {
                if (m->dynamic_byte_offset_upper_bound &&
                    (!maximum || *m->dynamic_byte_offset_upper_bound < *maximum))
                    return fail("Constant-buffer offset metadata has no sufficient re-established proof", op.source);
                if (m->dynamic_byte_offset && !maximum && b->size.bytes != constant_buffer_window_bytes)
                    return fail("Unbounded constant-buffer offset requires the full runtime window", op.source);
                std::uint64_t extent = constant_buffer_window_bytes;
                if (!m->dynamic_byte_offset || maximum) {
                    const std::uint64_t highest_base = std::uint64_t(m->byte_offset) +
                        m->dynamic_byte_offset_upper_bound.value_or(maximum.value_or(0));
                    const std::uint64_t last_component = highest_base + std::uint64_t(m->element_bytes) * (m->elements - 1);
                    const std::uint64_t last_word_address = m->element_bytes == 8 ?
                        ((last_component + 4) & ~std::uint64_t{3}) : (last_component & ~std::uint64_t{3});
                    extent = std::min<std::uint64_t>(last_word_address + 4, constant_buffer_window_bytes);
                }
                if (extent > b->size.bytes) return fail("Memory extent exceeds reflected buffer size", op.source);
            } else {
                if (m->dynamic_byte_offset) {
                    if (!maximum) return fail("Memory offset bound cannot be re-established", op.source,
                        DiagnosticCode::unresolved_resource);
                    if (!m->dynamic_byte_offset_upper_bound || *m->dynamic_byte_offset_upper_bound < *maximum)
                        return fail("Memory offset bound is missing or understates the proven range", op.source);
                }
                const std::uint64_t extent = std::uint64_t(m->byte_offset) +
                    std::uint64_t(m->element_bytes) * m->elements + maximum.value_or(0);
                if (extent > b->size.bytes) return fail("Memory extent exceeds reflected buffer size", op.source);
            }
        }
        if (const auto* t = std::get_if<TextureInfo>(&op.info)) {
            std::vector<bool> referenced(op.inputs.size());
            bool roles_valid = true;
            auto role = [&](InputIndex index) {
                if (index.value >= referenced.size()) roles_valid = false;
                else referenced[index.value] = true;
            };
            for (const auto* list : {&t->coordinates, &t->gradients_x, &t->gradients_y, &t->offsets, &t->data})
                for (auto index : *list) role(index);
            for (auto index : {t->array_layer, t->array_index, t->level_or_bias, t->compare, t->sample_index, t->lod_clamp}) if (index) role(*index);
            if (!roles_valid || std::find(referenced.begin(), referenced.end(), false) != referenced.end())
                return fail("Texture input roles are invalid or incomplete", op.source);
            for (const auto& input : op.inputs) {
                if (!charge()) return fail("Operand validation work exceeds limit", op.source, DiagnosticCode::limit_exceeded);
                bool word32 = false;
                if (const auto* value = std::get_if<ValueId>(&input)) word32 = value->value < p.values.size() && p.values[value->value].type == ValueType::word32;
                if (const auto* literal = std::get_if<LiteralBits>(&input)) word32 = literal->type == ValueType::word32 && literal->bits <= UINT32_MAX;
                if (!word32) return fail("Texture roles require word32 storage", op.source);
            }
            const auto* use = resource(t->resource);
            const auto* texture = use ? std::get_if<TextureUse>(use) : nullptr;
            const auto* key = std::get_if<TextureKey>(&t->texture);
            const auto access = op.opcode == Opcode::image_atomic ? Access::read_write :
                                op.opcode == Opcode::image_store ? Access::write : Access::read;
            if (!texture || !key || *key != texture->key || !t->descriptor || *t->descriptor != texture->descriptor ||
                texture->kind != (is_image(op) ? ResourceTextureKind::storage_image : ResourceTextureKind::sampled) ||
                !covers(texture->access, access) || (t->coherent && !texture->coherent) || (t->atomic && !texture->atomic) ||
                !std::binary_search(texture->modes.begin(), texture->modes.end(), t->mode))
                return fail("Texture resource does not cover operation requirements", op.source);
            if (auto status = check_texture(op, *t->descriptor, profile, p.stage, phase); std::holds_alternative<Diagnostic>(status)) return status;
            if (t->array_index) {
                if (!texture->dynamically_indexed || t->array_index->value >= op.inputs.size()) return fail("Texture array use lacks a valid index role", op.source);
                auto maximum = bounds.bound(op.inputs[t->array_index->value]);
                if (!maximum || *maximum >= texture->array_length) return fail("Texture array index bound cannot be re-established", op.source,
                    work > p.limits.analysis_steps ? DiagnosticCode::limit_exceeded : DiagnosticCode::unresolved_resource);
            } else if (texture->array_length != 1) return fail("Texture array has no index role", op.source);
            if (needs_sampler(op)) {
                const auto* sampler_use = resource(t->sampler_resource);
                const auto* sampler = sampler_use ? std::get_if<SamplerUse>(sampler_use) : nullptr;
                const auto* sampler_key = t->sampler ? std::get_if<TextureKey>(&*t->sampler) : key;
                if (!sampler || !sampler_key || sampler->key != *sampler_key ||
                    sampler->comparison != t->descriptor->comparison ||
                    sampler->normalized_coordinates != t->descriptor->normalized_coordinates ||
                    sampler->array_length != texture->array_length)
                    return fail("Sampler resource does not match the texture operation", op.source);
            } else if (t->sampler_resource || t->sampler) return fail("Sampler is present on an operation that does not use one", op.source);
        }
        if (const auto* io = std::get_if<IoInfo>(&op.info)) {
            if (io->component_mask == 0 || io->component_mask > 15 || io->type.components == 0 || io->type.components > 4 ||
                (op.opcode != Opcode::store_io && (!io->data.empty() || op.outputs.size() != static_cast<std::size_t>(std::popcount(io->component_mask)))) ||
                (op.opcode == Opcode::store_io && io->data.size() != static_cast<std::size_t>(std::popcount(io->component_mask))))
                return fail("IO component/data metadata is invalid", op.source);
            std::vector<bool> referenced(op.inputs.size());
            auto role = [&](InputIndex index) {
                if (index.value >= referenced.size()) return false;
                referenced[index.value] = true; return true;
            };
            for (auto index : io->data) if (!role(index)) return fail("IO data role is out of range", op.source);
            for (auto index : {io->dynamic_byte_offset, io->vertex_index, io->sample_index, io->reciprocal_w, io->offset_x, io->offset_y})
                if (index && !role(*index)) return fail("IO input role is out of range", op.source);
            if (std::find(referenced.begin(), referenced.end(), false) != referenced.end()) return fail("IO input has no role", op.source);
            std::uint8_t covered = 0;
            for (const auto& reflected : requirements.io) {
                if (!charge()) return fail("IO matching work exceeds limit", op.source, DiagnosticCode::limit_exceeded);
                if (reflected.semantic == io->semantic && reflected.location == io->location && reflected.type == io->type &&
                    reflected.interpolation == io->interpolation && covers(reflected.access,
                        (op.opcode == Opcode::store_io || io->output_storage) ? Access::write : Access::read))
                    covered |= reflected.mask;
            }
            if ((covered & io->component_mask) != io->component_mask) return fail("IO reflection omits operation components", op.source);
        }
    }
    return std::monostate{};
}
} // namespace armigo::shader::pipeline::resources
