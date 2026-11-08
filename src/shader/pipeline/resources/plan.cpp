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
#include <numeric>
#include <tuple>

namespace armigo::shader::pipeline {
namespace {
// Sorting changes allocation order only. Reflected ResourceIds remain arena indices.
auto buffer_key(const BufferUse& use) {
    std::uint32_t index = 0, offset = 0;
    std::string schema;
    if (const auto* cb = std::get_if<ConstantBufferSource>(&use.source)) index = cb->slot;
    else if (const auto* storage = std::get_if<StorageBufferSource>(&use.source)) {
        index = storage->descriptor.cbuf_slot; offset = storage->descriptor.handle_byte_offset;
    } else if (const auto* support = std::get_if<SupportBufferSource>(&use.source)) {
        schema = support->schema; index = support->version;
    } else index = std::get<VertexStreamSource>(use.source).slot;
    return std::tuple{use.source.index(), index, offset, schema};
}
bool allocation_less(const ResourceUse& a, const ResourceUse& b) {
    if (a.index() != b.index()) return a.index() < b.index();
    if (const auto* buffer = std::get_if<BufferUse>(&a)) return buffer_key(*buffer) < buffer_key(std::get<BufferUse>(b));
    if (const auto* texture = std::get_if<TextureUse>(&a)) {
        const auto& other = std::get<TextureUse>(b);
        return std::tuple{texture->key, texture->kind, texture->array_length} <
               std::tuple{other.key, other.kind, other.array_length};
    }
    const auto& x = std::get<SamplerUse>(a);
    const auto& y = std::get<SamplerUse>(b);
    return std::tuple{x.key, x.array_length, x.comparison, x.normalized_coordinates} <
           std::tuple{y.key, y.array_length, y.comparison, y.normalized_coordinates};
}
}
Result<ResourcePlan> plan_resources(const StructuredProgram& program, const MetalProfile& profile) {
    constexpr auto phase = Phase::plan_resources;
    auto fail = [&](std::string message, DiagnosticCode code = DiagnosticCode::invalid_ir) -> Result<ResourcePlan> {
        return diagnose(phase, code, {}, std::move(message));
    };
    if (auto status = resources::check_profile(profile, phase); std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    if (profile != program.profile) return fail("Resource planner profile differs from legalization profile", DiagnosticCode::unsupported_capability);
    if (auto status = resources::check_requirements(program.ir, program.requirements, profile, phase); std::holds_alternative<Diagnostic>(status))
        return std::get<Diagnostic>(status);
    auto dependencies = merge_dependencies(program.ir.dependencies, {}, program.ir.limits, phase);
    if (auto* error = std::get_if<Diagnostic>(&dependencies)) return *error;
    if (std::get<std::vector<Dependency>>(dependencies) != program.ir.dependencies) return fail("Resource planning requires canonical dependencies");
    ResourcePlan plan{3, program.ir.stage, profile, {}, program.requirements.io, program.ir.dependencies,
        FragmentPositionConvention::unspecified, program.requirements.compute, false, {}, {}};
    const auto& uses = program.requirements.uses;
    std::uint64_t texture_count = 0, sampler_count = 0;
    for (const auto& use : uses) {
        if (const auto* texture = std::get_if<TextureUse>(&use); texture && !texture->pool) texture_count += texture->array_length;
        if (const auto* sampler = std::get_if<SamplerUse>(&use); sampler && !sampler->pool) sampler_count += sampler->array_length;
    }
    const bool indirect = texture_count > profile.texture_slots || sampler_count > profile.sampler_slots;
    if (indirect && (texture_count > UINT32_MAX / 8u || sampler_count > UINT32_MAX / 8u - texture_count))
        return fail("Stage argument buffer extent exceeds uint32: textures=" + std::to_string(texture_count) +
                    ", samplers=" + std::to_string(sampler_count), DiagnosticCode::limit_exceeded);
    plan.resources.resize(uses.size());
    std::vector<bool> occupied(profile.buffer_slots);
    std::vector<std::uint32_t> streams;
    std::uint64_t steps = 0;
    // Explicit stream mappings are ordered by their first appearance in IO reflection.
    // Unreferenced stream requirements follow in requirements order, never guest slot order.
    auto add_stream = [&](std::uint16_t slot) {
        for (std::uint32_t id = 0; id < uses.size(); ++id) {
            if (++steps > program.ir.limits.analysis_steps) return false;
            const auto* buffer = std::get_if<BufferUse>(&uses[id]);
            const auto* stream = buffer ? std::get_if<VertexStreamSource>(&buffer->source) : nullptr;
            if (stream && stream->slot == slot && std::find(streams.begin(), streams.end(), id) == streams.end()) streams.push_back(id);
        }
        return true;
    };
    for (const auto& io : plan.io) if (io.vertex_stream && !add_stream(*io.vertex_stream))
        return fail("Stream allocation work exceeds limit", DiagnosticCode::limit_exceeded);
    for (const auto& use : uses) if (const auto* b = std::get_if<BufferUse>(&use))
        if (const auto* s = std::get_if<VertexStreamSource>(&b->source); s && !add_stream(s->slot))
            return fail("Stream allocation work exceeds limit", DiagnosticCode::limit_exceeded);
    for (std::size_t ordinal = 0; ordinal < streams.size(); ++ordinal) {
        const auto slot = constant_buffer_slot_count + ordinal;
        if (slot >= occupied.size()) return fail("Vertex streams exceed Metal buffer capacity", DiagnosticCode::limit_exceeded);
        occupied[slot] = true;
        const auto id = streams[ordinal];
        plan.resources[id] = {ResourceId{id}, uses[id], MetalRange{MetalNamespace::buffer, static_cast<std::uint32_t>(slot), 1}};
    }
    // Native vertex-format inputs are serialized as streams at 18 + ordinal
    // even when no explicit VertexStreamSource requirement was needed by MSL.
    if (program.ir.stage == Stage::vertex) {
        const auto native_streams = std::count_if(plan.io.begin(), plan.io.end(), [](const auto& io) {
            return io.semantic == IoSemantic::generic && io.access == Access::read &&
                   io.vertex_format && !io.vertex_stream;
        });
        for (std::uint32_t ordinal = 0; ordinal < native_streams; ++ordinal) {
            const auto slot = constant_buffer_slot_count + ordinal;
            if (slot >= occupied.size()) return fail("Vertex streams exceed Metal buffer capacity", DiagnosticCode::limit_exceeded);
            occupied[slot] = true;
        }
    }
    std::vector<std::uint32_t> order(uses.size());
    std::iota(order.begin(), order.end(), 0);
    const auto sorting_bound = std::uint64_t(uses.size()) * (std::bit_width(uses.size()) + 1);
    if (sorting_bound > program.ir.limits.analysis_steps - std::min(steps, program.ir.limits.analysis_steps))
        return fail("Resource sorting work exceeds limit", DiagnosticCode::limit_exceeded);
    steps += sorting_bound;
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return allocation_less(uses[a], uses[b]); });
    std::uint32_t next_texture = 0, next_sampler = 0, next_buffer = 0;
    std::uint32_t member_offset = 0;
    for (const auto id : order) {
        if (++steps > program.ir.limits.analysis_steps) return fail("Allocation work exceeds limit", DiagnosticCode::limit_exceeded);
        const auto& use = uses[id];
        MetalRange range;
        const auto* texture_use = std::get_if<TextureUse>(&use);
        const auto* sampler_use = std::get_if<SamplerUse>(&use);
        // A pool is a table in the buffer namespace (the project documentation).
        const bool pool = (texture_use && texture_use->pool) || (sampler_use && sampler_use->pool);
        if (const auto* buffer = std::get_if<BufferUse>(&use); buffer || pool) {
            if (buffer && std::holds_alternative<VertexStreamSource>(buffer->source)) continue;
            while (next_buffer < occupied.size() && occupied[next_buffer]) ++next_buffer;
            if (next_buffer >= occupied.size()) return fail("Metal buffer namespace is exhausted", DiagnosticCode::limit_exceeded);
            range = {MetalNamespace::buffer, next_buffer, 1};
            occupied[next_buffer++] = true;
        } else {
            const auto* texture = std::get_if<TextureUse>(&use);
            const auto count = texture ? texture->array_length : std::get<SamplerUse>(use).array_length;
            if (indirect) {
                plan.resources[id] = {ResourceId{id}, use, ArgumentBufferMember{member_offset, count, 8}};
                member_offset += count * 8;
                continue;
            }
            auto& next = texture ? next_texture : next_sampler;
            const auto capacity = texture ? profile.texture_slots : profile.sampler_slots;
            if (count == 0 || count > capacity || next > capacity - count)
                return fail("Metal texture/sampler namespace is exhausted", DiagnosticCode::limit_exceeded);
            range = {texture ? MetalNamespace::texture : MetalNamespace::sampler,
                     next, count};
            next += count;
        }
        plan.resources[id] = {ResourceId{id}, use, range};
    }
    if (indirect) {
        while (next_buffer < occupied.size() && occupied[next_buffer]) ++next_buffer;
        if (next_buffer >= occupied.size())
            return fail("Stage argument buffer has no free Metal buffer slot: textures=" + std::to_string(texture_count) +
                        ", samplers=" + std::to_string(sampler_count), DiagnosticCode::limit_exceeded);
        plan.argument_buffer = StageArgumentBufferLayout{1, next_buffer, member_offset, 8};
        occupied[next_buffer++] = true;
    }
    const auto reads = [&](IoSemantic semantic) {
        return std::any_of(plan.io.begin(), plan.io.end(), [&](const auto& io) {
            return io.access == Access::read && io.semantic == semantic;
        });
    };
    if ((program.ir.stage == Stage::vertex && reads(IoSemantic::instance_id)) ||
        (program.ir.stage == Stage::fragment && reads(IoSemantic::render_target_height))) {
        while (next_buffer < occupied.size() && occupied[next_buffer]) ++next_buffer;
        if (next_buffer >= occupied.size())
            return fail("Draw support has no free Metal buffer slot", DiagnosticCode::limit_exceeded);
        plan.draw_support_buffer_index = next_buffer;
    }
    return plan;
}
} // namespace armigo::shader::pipeline
