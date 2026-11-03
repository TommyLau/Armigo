// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "platform/metal/device_context.h"
#include "platform/metal/metal_renderer.h"
#include "platform/metal/texture_cache.h"
#include "platform/metal/texture_fact.h"
#include "platform/metal/buffer_cache.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>

namespace g = armigo::graphics;
namespace m = armigo::platform::metal;
namespace p = armigo::platform;
namespace {
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "renderer contract check failed: " << message << '\n';
        std::exit(1);
    }
}

// These compile-only implementations deliberately reject execution. No device,
// queue, shader package or successful GPU work is fabricated by this test.
struct MockDevice final : m::DeviceContext {
    m::PipelineContextKey key{1, "synthetic-chain", "synthetic-profile"};
    id<MTLDevice> device() const noexcept override { return nil; }
    id<MTL4CommandQueue> queue() const noexcept override { return nil; }
    const m::PipelineContextKey& identity() const noexcept override { return key; }
};
struct MockPipelines final : m::PipelineStateCache {
    m::PipelineContextKey key{1, "synthetic-chain", "synthetic-profile"};
    const m::PipelineContextKey& context() const noexcept override { return key; }
    g::RendererResult<m::ResidentPipeline> acquire(
        const m::PipelineKey&, const m::AdmittedGraphicsStages&) override { return {}; }
    g::RendererResult<g::RetainedResource> depth_stencil(
        const g::DepthStencilState&) override { return {}; }
    g::RendererResult<g::RetainedResource> sampler(
        const g::SamplerDescriptor&) override { return {}; }
    g::RendererStatus trim(std::uint64_t) override { return {}; }
};
struct MockTextures final : g::TextureCache {
    g::ResidencyDomain domain() const noexcept override { return {}; }
    g::RendererResult<g::ResidentTexture> acquire(const g::TextureRequest&) override { return {}; }
    g::RendererStatus invalidate(g::BackingRange) override { return {}; }
    g::RendererStatus resolve(const g::TextureViewKey&, g::ResolveReason) override { return {}; }
    g::RendererStatus trim(std::uint64_t) override { return {}; }
};
struct MockBuffers final : g::BufferCache {
    g::ResidencyDomain domain() const noexcept override { return {}; }
    g::RendererResult<g::ResidentBuffer> acquire(const g::BufferRequest&) override { return {}; }
    g::RendererResult<g::ResidentBuffer> upload(std::span<const std::byte>, std::uint64_t) override {
        return {};
    }
    g::RendererStatus invalidate(g::BackingRange) override { return {}; }
    g::RendererStatus resolve(g::BufferKey, g::ResolveReason) override { return {}; }
    g::RendererStatus trim(std::uint64_t) override { return {}; }
};
struct MockBindings final : m::BindingPlanner {
    g::RendererResult<m::Bindings> prepare(const g::RenderStop&,
        const m::AdmittedGraphicsStages&, const g::DrawPlan&, g::ResourceMemory&,
        g::BufferCache&, g::TextureCache&, const m::BindingReuse&) override { return {}; }
};
struct MockQueue final : m::SubmissionQueue {
    g::ResidencyDomain domain() const noexcept override { return {}; }
    g::RendererResult<g::SubmissionReceipt> append(m::PreparedRender) override { return {}; }
    g::RendererStatus flush(g::SubmissionBoundary) override { return {}; }
    g::RendererStatus poll() override { return {}; }
    g::RendererStatus drain(std::uint64_t) override { return {}; }
};
struct MockProvider final : m::GraphicsStageProvider {
    g::RendererResult<m::AdmittedGraphicsStages> lookup(
        const g::RenderStop&, const g::GpuAddressSpace&) override { return {}; }
};
struct MockRenderer final : m::MetalRenderer {
    g::RendererResult<g::SubmissionReceipt> enqueue(g::RenderRequest) override { return {}; }
    g::RendererStatus flush(g::SubmissionBoundary) override { return {}; }
    g::RendererStatus poll() override { return {}; }
    g::RendererStatus drain(std::uint64_t) override { return {}; }
};

void pipeline_key_contract() {
    m::PipelineKey key;
    key.context = {7, "synthetic-chain", "profile"};
    key.vertex = {0, "package", "vertex", "artifact-v", "vertex_main", {}};
    key.fragment = {0, "package", "fragment", "artifact-f", "fragment_main", {}};
    key.attributes.push_back({0, 1, 0, g::VertexAttribFormat::R32G32Float});
    key.buffers.push_back({1, 8, g::VertexStep::PerVertex, 1});
    key.colors[0].format = g::TextureFormat::R8G8B8A8Unorm;
    const auto differs = [&](auto mutate) {
        auto changed = key;
        mutate(changed);
        check(!(changed == key), "baked pipeline field must change equality");
    };
    differs([](auto& k) { ++k.context.device; });
    differs([](auto& k) { k.context.compiler_chain += "x"; });
    differs([](auto& k) { k.context.capability_profile += "x"; });
    differs([](auto& k) { k.vertex.package += "x"; });
    differs([](auto& k) { k.vertex.variant += "x"; });
    differs([](auto& k) { k.vertex.artifact += "x"; });
    differs([](auto& k) { k.vertex.entry_point += "x"; });
    differs([](auto& k) { k.fragment.variant += "x"; });
    differs([](auto& k) { k.fragment.specialization.push_back({0, "u32", {std::byte{1}}}); });
    differs([](auto& k) { ++k.attributes[0].location; });
    differs([](auto& k) { ++k.attributes[0].buffer_index; });
    differs([](auto& k) { ++k.attributes[0].offset; });
    differs([](auto& k) { k.attributes[0].format = g::VertexAttribFormat::R32Float; });
    differs([](auto& k) { ++k.buffers[0].buffer_index; });
    differs([](auto& k) { ++k.buffers[0].stride; });
    differs([](auto& k) { k.buffers[0].step = g::VertexStep::PerInstance; });
    differs([](auto& k) { ++k.buffers[0].rate; });
    differs([](auto& k) { k.colors[7].format = g::TextureFormat::R32Uint; });
    differs([](auto& k) { k.colors[0].blend.enabled = true; });
    differs([](auto& k) { k.colors[0].blend.rgb.operation = g::BlendOperation::Max; });
    differs([](auto& k) { k.colors[0].blend.rgb.source = g::BlendFactor::SourceAlpha; });
    differs([](auto& k) { k.colors[0].blend.alpha.destination = g::BlendFactor::One; });
    differs([](auto& k) { k.colors[0].blend.write_mask = 1; });
    differs([](auto& k) { k.depth_format = g::TextureFormat::D32Float; });
    differs([](auto& k) { k.stencil_format = g::TextureFormat::D32FloatS8Uint; });
    differs([](auto& k) { k.coverage.samples = 4; });
    differs([](auto& k) { k.coverage.sample_mask = 1; });
    differs([](auto& k) { k.coverage.multisample = true; });
    differs([](auto& k) { k.coverage.alpha_to_coverage = true; });
    differs([](auto& k) { k.coverage.alpha_to_one = true; });
    differs([](auto& k) { k.primitive = g::PrimitiveClass::Line; });
    differs([](auto& k) { k.rasterization_enabled = false; });
    differs([](auto& k) { k.shader_emulation = 1; });
    // Deliberately identical hashes must still compare complete descriptors.
    auto changed = key;
    changed.colors[7].format = g::TextureFormat::R32Sint;
    const auto colliding_hash = [](const auto&) { return 0u; };
    check(colliding_hash(key) == colliding_hash(changed) && !(key == changed),
          "hash equality is not pipeline identity");

}
// The fact target check admits exact matches, serves same-dimensions
// arrayness differences from a layer-zero view target, and rejects every
// other target difference with both targets named.
void texture_target_view_contract() {
    g::ResolvedTextureBinding live;
    live.texture.format = g::TextureFormat::R8G8B8A8Unorm;
    live.texture.samples = 1;
    live.texture.normalized_coordinates = true;
    const auto fact_for = [](p::TextureFactDimension dimension, bool array,
                             bool multisample = false) {
        p::DerivedTextureFact fact{};
        fact.dimension = dimension;
        fact.array = array;
        fact.multisample = multisample;
        return fact;
    };
    const auto plan_for = [&](p::TextureFactDimension dimension, bool array,
                              g::TextureType type, std::uint32_t samples = 1,
                              bool multisample = false) {
        live.texture.type = type;
        live.texture.samples = samples;
        return m::texture_target_plan(fact_for(dimension, array, multisample), live);
    };
    const auto expect_exact = [&](const char* what, p::TextureFactDimension dimension,
                                  bool array, g::TextureType type) {
        const auto plan = plan_for(dimension, array, type);
        const auto* admitted = std::get_if<m::TextureTargetPlan>(&plan);
        check(admitted != nullptr && admitted->exact && admitted->layer_count == 1 &&
                  admitted->first_layer == 0,
              what);
    };
    const auto expect_view = [&](const char* what, p::TextureFactDimension dimension,
                                 bool array, g::TextureType type,
                                 g::TextureType view, std::uint32_t layer_count) {
        const auto plan = plan_for(dimension, array, type);
        const auto* admitted = std::get_if<m::TextureTargetPlan>(&plan);
        check(admitted != nullptr && !admitted->exact &&
                  admitted->view_target == view && admitted->first_layer == 0 &&
                  admitted->layer_count == layer_count,
              what);
    };
    const auto expect_reject = [&](const char* what, p::TextureFactDimension dimension,
                                   bool array, g::TextureType type,
                                   std::uint32_t samples, bool multisample,
                                   std::initializer_list<std::string_view> names) {
        const auto plan = plan_for(dimension, array, type, samples, multisample);
        const auto* mismatch = std::get_if<std::string>(&plan);
        check(mismatch != nullptr, what);
        if (!mismatch) return;
        for (const auto name : names)
            check(mismatch->find(name) != std::string::npos, what);
    };
    expect_exact("matching 2D facts must bind exactly",
                 p::TextureFactDimension::d2, false, g::TextureType::Texture2D);
    // Rectangle storage shares the plain 2D shape, so the fact passes exactly.
    expect_exact("rectangle storage must satisfy a plain 2D fact",
                 p::TextureFactDimension::d2, false, g::TextureType::Texture2DRect);
    expect_exact("matching cube facts must bind exactly",
                 p::TextureFactDimension::cube, false, g::TextureType::Cubemap);
    expect_exact("matching 2D-array facts must bind exactly",
                 p::TextureFactDimension::d2, true, g::TextureType::Texture2DArray);
    expect_view("2D facts must view 2D-array storage at layer zero",
                p::TextureFactDimension::d2, false, g::TextureType::Texture2DArray,
                g::TextureType::Texture2D, 1);
    expect_view("2D-array facts must view plain 2D storage",
                p::TextureFactDimension::d2, true, g::TextureType::Texture2D,
                g::TextureType::Texture2DArray, 1);
    expect_view("1D facts must view 1D-array storage",
                p::TextureFactDimension::d1, false, g::TextureType::Texture1DArray,
                g::TextureType::Texture1D, 1);
    expect_view("1D-array facts must view plain 1D storage",
                p::TextureFactDimension::d1, true, g::TextureType::Texture1D,
                g::TextureType::Texture1DArray, 1);
    expect_view("cube facts must view cube-array storage at its first face group",
                p::TextureFactDimension::cube, false, g::TextureType::CubemapArray,
                g::TextureType::Cubemap, 6);
    expect_view("cube-array facts must view cube storage as one face group",
                p::TextureFactDimension::cube, true, g::TextureType::Cubemap,
                g::TextureType::CubemapArray, 6);
    expect_reject("2D and 3D targets must reject naming both shapes",
                  p::TextureFactDimension::d2, false, g::TextureType::Texture3D, 1,
                  false, {"shader target (2D", "live texture target (3D"});
    expect_reject("cube and 2D targets must reject naming both shapes",
                  p::TextureFactDimension::cube, false, g::TextureType::Texture2D, 1,
                  false, {"shader target (cube", "live texture target (2D"});
    expect_reject("multisample live textures must reject naming the sample count",
                  p::TextureFactDimension::d2, false, g::TextureType::Texture2D, 4,
                  false, {"shader target (2D", "live texture target (2D", "4 samples"});
    expect_reject("multisample facts must reject single-sample storage",
                  p::TextureFactDimension::d2, false, g::TextureType::Texture2D, 1,
                  true, {"multisample"});
    expect_reject("the non-existent 3D-array target must reject",
                  p::TextureFactDimension::d3, true, g::TextureType::Texture3D, 1,
                  false, {"3D array", "not a bindable"});
    // View-key identity must distinguish admitted view targets.
    const g::TextureViewKey plain, viewed = [] {
        g::TextureViewKey key;
        key.view_target = g::TextureType::Texture2DArray;
        return key;
    }();
    check(!(plain == viewed) && plain == g::TextureViewKey{},
          "view-key equality must cover the optional view target");
}

} // namespace

int main() {
    static_assert(!std::is_abstract_v<MockDevice> && !std::is_abstract_v<MockPipelines>);
    static_assert(!std::is_abstract_v<MockBindings> && !std::is_abstract_v<MockProvider>);
    static_assert(!std::is_abstract_v<MockTextures> && !std::is_abstract_v<MockBuffers>);
    MockRenderer renderer;
    check(std::holds_alternative<g::RendererError>(renderer.enqueue({})),
          "mock renderer cannot claim unsupported work was accepted");
    MockQueue queue;
    check(std::holds_alternative<g::RendererError>(queue.append({})),
          "mock queue must reject unimplemented work");
    pipeline_key_contract();
    texture_target_view_contract();
    std::cout << "Metal renderer contracts compiled and checked\n";
}
