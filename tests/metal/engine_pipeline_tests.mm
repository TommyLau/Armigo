// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "platform/metal/engine_pipeline.h"
#include "platform/metal/native_pipeline_state.h"
#include "platform/metal/native_resources.h"

#include <bit>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <span>

namespace g = armigo::graphics;
namespace p = armigo::platform;
namespace m = armigo::platform::metal;
namespace gm = armigo::shader::graphics_msl;

namespace {

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                      \
            std::fprintf(stderr, "Engine pipeline check failed at %s:%d: %s\n", \
                         __FILE__, __LINE__, #condition);                        \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

template <class T>
T take(g::RendererResult<T> result) {
    if (auto* value = std::get_if<T>(&result)) return std::move(*value);
    const auto command = std::get<g::RendererError>(result).command();
    std::fprintf(stderr, "Unexpected renderer error: %.*s\n",
                 int(command.detail.size()), command.detail.data());
    std::exit(1);
}

bool rejected(const auto& result) {
    return std::holds_alternative<g::RendererError>(result);
}

struct LookupContext {
    std::shared_ptr<p::CompiledGraphicsStage> vertex;
    std::shared_ptr<p::CompiledGraphicsStage> fragment;
    unsigned calls = 0;
};

p::CompiledGraphicsLookupResult lookup(void* raw, const g::RenderStop&,
                                       const g::GpuAddressSpace&) {
    auto& context = *static_cast<LookupContext*>(raw);
    ++context.calls;
    return p::CompiledGraphicsPair{context.vertex, context.fragment};
}

LookupContext stages() {
    LookupContext context;
    context.vertex = std::make_shared<p::CompiledGraphicsStage>();
    context.fragment = std::make_shared<p::CompiledGraphicsStage>();
    context.vertex->metallib = {std::byte{1}, std::byte{2}};
    context.vertex->stage = armigo::shader_input::Stage::Vertex;
    context.vertex->entry_point = "vertex_main";
    context.vertex->vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    context.vertex->interface.position_output_mask = 15;
    context.fragment->metallib = {std::byte{3}, std::byte{4}};
    context.fragment->stage = armigo::shader_input::Stage::Fragment;
    context.fragment->entry_point = "fragment_main";
    context.fragment->interface.color_output_mask = 15;
    context.fragment->color_outputs.push_back(
        {0, 15, p::ColorOutputScalar::f32, 4});
    return context;
}

g::RenderStop stop() {
    g::RenderStop value;
    value.engine = g::engine_class::threed;
    value.kind = g::RenderKind::Draw;
    value.draw = g::DrawDescription{true, 4, 0, 3, 0, 0, 1};
    auto& r = value.registers;
    r[0x0df] = 1;
    r[0x651] = 1;
    r[0x487] = 1;
    r[0x36b] = r[0x36c] = 0x1b02;
    r[0x4b9] = 1;
    r[0x680] = 0x1111;
    r[0x3ef] = 1;
    r[0x646] = 1;
    r[0x647] = 0x901;
    r[0x648] = 0x405;
    r[0x64b] = 1;
    r[0x286] = 0x6420;
    r[0x64f] = 0x181d;
    r[0x5f6] = 1;
    for (const auto [index, number] :
         {std::pair{0x280, 4.0f}, std::pair{0x281, -4.0f},
          std::pair{0x282, 0.5f}, std::pair{0x283, 4.0f},
          std::pair{0x284, 4.0f}, std::pair{0x285, 0.5f},
          std::pair{0x303, 1.0f}})
        r[index] = std::bit_cast<std::uint32_t>(number);
    r[0x300] = r[0x301] = 8u << 16;
    r[0x380] = 1;
    r[0x381] = r[0x382] = 8u << 16;
    r[0x3fd] = r[0x3fe] = 8u << 16;
    value.programs[1].enabled = value.programs[5].enabled = true;
    value.color_targets[0] = {0x1000, 8, 8, 0xd6, 0, 1, 128, 0};
    return value;
}

struct AddressSpace {
    std::array<std::byte, 4096> bytes{};
    static bool read(void* raw, std::uint64_t address,
                     std::span<std::byte> output) noexcept {
        auto& self = *static_cast<AddressSpace*>(raw);
        if (address < 0x1000 || address - 0x1000 > self.bytes.size() ||
            output.size() > self.bytes.size() - (address - 0x1000))
            return false;
        std::memcpy(output.data(), self.bytes.data() + address - 0x1000,
                    output.size());
        return true;
    }
    static bool write(void*, std::uint64_t,
                      std::span<const std::byte>) noexcept {
        return false;
    }
    g::GpuAddressSpace make() {
        const g::Mapping mapping{0x1000, bytes.size(), 0, 0, bytes.size(), 4096};
        return std::get<g::GpuAddressSpace>(g::make_address_space(
            std::span{&mapping, 1}, {read, write, this}));
    }
};

// A provider told its stages are immutable derives the same identities, once
// per stage, as the provider that rehashes on every request.
void immutable_provider_memoizes_identities() {
    auto context = stages();
    auto live = take(m::create_engine_stage_provider(lookup, &context));
    auto memoized = take(m::create_engine_stage_provider(lookup, &context, nullptr, true));
    AddressSpace backing;
    const auto memory = backing.make();
    const auto expected = take(live->lookup(stop(), memory));
    const auto first = take(memoized->lookup(stop(), memory));
    const auto second = take(memoized->lookup(stop(), memory));
    CHECK(first.vertex_key.artifact == expected.vertex_key.artifact);
    CHECK(first.vertex_key.package == expected.vertex_key.package);
    CHECK(first.fragment_key.variant == expected.fragment_key.variant);
    CHECK(second.vertex_key.package == first.vertex_key.package);
    CHECK(second.fragment_key.artifact == first.fragment_key.artifact);
}

void provider_identity_and_live_lookup() {
    auto context = stages();
    auto provider = take(m::create_engine_stage_provider(lookup, &context));
    AddressSpace backing;
    const auto memory = backing.make();
    auto first = take(provider->lookup(stop(), memory));
    context.vertex->metallib.push_back(std::byte{5});
    auto artifact_changed = take(provider->lookup(stop(), memory));
    CHECK(first.vertex_key.artifact != artifact_changed.vertex_key.artifact);
    CHECK(first.vertex_key.package != artifact_changed.vertex_key.package);
    context.vertex->entry_point = "vertex_main_variant";
    auto entry_changed = take(provider->lookup(stop(), memory));
    CHECK(artifact_changed.vertex_key.entry_point !=
          entry_changed.vertex_key.entry_point);
    CHECK(artifact_changed.vertex_key.variant != entry_changed.vertex_key.variant);
    CHECK(artifact_changed.vertex_key.package != entry_changed.vertex_key.package);
    context.fragment->constant_buffers.push_back({0, 4, 0});
    auto reflection_changed = take(provider->lookup(stop(), memory));
    CHECK(entry_changed.fragment_key.artifact ==
          reflection_changed.fragment_key.artifact);
    CHECK(entry_changed.fragment_key.variant !=
          reflection_changed.fragment_key.variant);
    CHECK(entry_changed.fragment_key.package !=
          reflection_changed.fragment_key.package);
    CHECK(context.calls == 4);
    context.vertex->draw_support_buffer_index = 4;
    const auto with_draw_support = take(provider->lookup(stop(), memory));
    CHECK(reflection_changed.vertex_key.artifact == with_draw_support.vertex_key.artifact);
    CHECK(reflection_changed.vertex_key.variant != with_draw_support.vertex_key.variant);
    context.vertex->draw_support_buffer_index = 5;
    const auto relocated = take(provider->lookup(stop(), memory));
    CHECK(with_draw_support.vertex_key.variant != relocated.vertex_key.variant);
    context.vertex->draw_support_buffer_index = 31;
    CHECK(rejected(provider->lookup(stop(), memory)));
    context.vertex->draw_support_buffer_index.reset();
    context.fragment->draw_support_buffer_index = 4;
    CHECK(rejected(provider->lookup(stop(), memory)));

    auto clear_only = take(m::create_engine_stage_provider(nullptr, nullptr));
    CHECK(rejected(clear_only->lookup(stop(), memory)));
}

void fragment_exports_admitted_and_keyed() {
    auto context = stages();
    auto provider = take(m::create_engine_stage_provider(lookup, &context));
    AddressSpace backing;
    const auto memory = backing.make();
    const auto baseline = take(provider->lookup(stop(), memory));
    // A fragment stage may export depth, sample mask, or both; each flag
    // combination keeps its own package identity through the reflection
    // digest, so exporting and non-exporting variants never share a key.
    context.fragment->writes_fragment_depth = true;
    const auto depth_export = take(provider->lookup(stop(), memory));
    CHECK(depth_export.fragment_key.artifact == baseline.fragment_key.artifact);
    CHECK(depth_export.fragment_key.variant != baseline.fragment_key.variant);
    CHECK(depth_export.fragment_key.package != baseline.fragment_key.package);
    context.fragment->writes_sample_mask = true;
    const auto both_exports = take(provider->lookup(stop(), memory));
    CHECK(both_exports.fragment_key.variant != depth_export.fragment_key.variant);
    context.fragment->writes_fragment_depth = false;
    const auto sample_mask = take(provider->lookup(stop(), memory));
    CHECK(sample_mask.fragment_key.variant != baseline.fragment_key.variant);
    CHECK(sample_mask.fragment_key.variant != both_exports.fragment_key.variant);
    // Vertex stages can never carry fragment-stage exports.
    context.vertex->writes_fragment_depth = true;
    CHECK(rejected(provider->lookup(stop(), memory)));
    context.vertex->writes_fragment_depth = false;
    context.vertex->writes_sample_mask = true;
    CHECK(rejected(provider->lookup(stop(), memory)));
}

void texture_contract_changes_stage_identity() {
    auto context = stages();
    context.fragment->fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    context.fragment->textures = {{32, 0, gm::TextureBinding::no_sampler}};
    context.fragment->texture_handle_cbuf = 0;
    context.fragment->texture_facts = {{0, 32, true, false}};
    auto provider = take(m::create_engine_stage_provider(lookup, &context));
    AddressSpace backing;
    const auto memory = backing.make();
    const auto first = take(provider->lookup(stop(), memory));
    const auto original = context.fragment->texture_facts.front();
    for (unsigned mutation = 0; mutation != 7; ++mutation) {
        auto& fact = context.fragment->texture_facts.front();
        fact = original;
        switch (mutation) {
        case 0: fact.dimension = p::TextureFactDimension::cube; break;
        case 1: fact.array = true; break;
        case 2: fact.multisample = true; break;
        case 3: fact.scalar = p::TextureSampledScalar::u32; break;
        case 4: fact.component_format = p::TextureComponentFormat::unorm; break;
        case 5: fact.normalized_coordinates = false; break;
        case 6: fact.comparison = true; break;
        }
        const auto changed = take(provider->lookup(stop(), memory));
        CHECK(first.fragment_key.artifact == changed.fragment_key.artifact);
        CHECK(first.fragment_key.variant != changed.fragment_key.variant);
        CHECK(first.fragment_key.package != changed.fragment_key.package);
        CHECK(first.vertex_key.package != changed.vertex_key.package);
    }
}

m::PipelineKey base_key(const std::shared_ptr<m::PipelineStateCache>& cache,
                        const m::AdmittedGraphicsStages& stages) {
    m::PipelineKey key;
    key.context = cache->context();
    key.vertex = stages.vertex_key;
    key.fragment = stages.fragment_key;
    key.colors[0].format = g::TextureFormat::R8G8B8A8Srgb;
    return key;
}

void cache_rejects_invalid_keys() {
    auto device = take(m::create_device_context());
    auto cache = take(m::create_pipeline_state_cache(device, 1024 * 1024));
    auto context = stages();
    auto provider = take(m::create_engine_stage_provider(lookup, &context));
    AddressSpace backing;
    auto admitted = take(provider->lookup(stop(), backing.make()));
    auto key = base_key(cache, admitted);

    auto mismatch = key;
    mismatch.vertex.artifact.push_back('x');
    CHECK(rejected(cache->acquire(mismatch, admitted)));
    auto invalid_context = key;
    ++invalid_context.context.device;
    CHECK(rejected(cache->acquire(invalid_context, admitted)));
    auto specialized = key;
    specialized.vertex.specialization.push_back(
        {0, "u32", {std::byte{1}, std::byte{0}, std::byte{0}, std::byte{0}}});
    CHECK(rejected(cache->acquire(specialized, admitted)));
}

void native_state_reuse_and_trim() {
    auto device = take(m::create_device_context());
    auto cache = take(m::create_pipeline_state_cache(device, 1024 * 1024));
    g::SamplerDescriptor descriptor;
    auto first = take(cache->sampler(descriptor));
    descriptor.raw_words[0] = 0xffffffff;
    auto second = take(cache->sampler(descriptor));
    auto sampler1 = std::dynamic_pointer_cast<const m::MetalSamplerResource>(first);
    auto sampler2 = std::dynamic_pointer_cast<const m::MetalSamplerResource>(second);
    CHECK(sampler1 && sampler2 && sampler1->sampler() == sampler2->sampler());

    // The border conversion flag is inert for default non-border addressing.
    // The shared sampler mapper, rather than the removed compatibility gate,
    // decides whether the live combination has a representation.
    {
        auto border_flag = descriptor;
        border_flag.srgb_border_conversion = true;
        CHECK(!rejected(cache->sampler(border_flag)));
    }

    auto depth1 = take(cache->depth_stencil({}));
    auto depth2 = take(cache->depth_stencil({}));
    auto native_depth1 =
        std::dynamic_pointer_cast<const m::MetalDepthStencilResource>(depth1);
    auto native_depth2 =
        std::dynamic_pointer_cast<const m::MetalDepthStencilResource>(depth2);
    CHECK(native_depth1 && native_depth2 &&
          native_depth1->depth_stencil() == native_depth2->depth_stencil());
    CHECK(rejected(cache->trim(0)));
    CHECK(native_depth1->depth_stencil() != nil && sampler1->sampler() != nil);
    first.reset();
    second.reset();
    sampler1.reset();
    sampler2.reset();
    depth1.reset();
    depth2.reset();
    native_depth1.reset();
    native_depth2.reset();
    CHECK(std::holds_alternative<std::monostate>(cache->trim(0)));

    auto tiny = take(m::create_pipeline_state_cache(device, 1));
    CHECK(rejected(tiny->sampler({})));
}

} // namespace

int main() {
    @autoreleasepool {
        provider_identity_and_live_lookup();
    immutable_provider_memoizes_identities();
        fragment_exports_admitted_and_keyed();
        texture_contract_changes_stage_identity();
        cache_rejects_invalid_keys();
        native_state_reuse_and_trim();
    }
    std::printf("Engine pipeline failures: %u\n", failures);
    return failures ? 1 : 0;
}
