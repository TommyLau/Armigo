// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "bindings.h"
#include "pipeline_state.h"
#include "submission.h"
#include <armigo/graphics/renderer.h>

namespace armigo::platform::metal {

// The provider owns the package and lookup context. Lookups reuse the existing
// CompiledGraphicsPair and checked GpuAddressSpace. This call is synchronous;
// only admitted shared stages are retained, never the borrowed memory view.
class GraphicsStageProvider {
public:
    virtual ~GraphicsStageProvider() = default;
    virtual graphics::RendererResult<AdmittedGraphicsStages> lookup(
        const graphics::RenderStop& stop, const graphics::GpuAddressSpace& memory) = 0;
    // Admission checks of a stop whose stage-selection inputs are unchanged:
    // success means a previous lookup's stages remain valid for it. Providers
    // without separate checks report Unsupported, forcing a lookup.
    virtual graphics::RendererStatus validate(const graphics::RenderStop&) {
        return graphics::RendererError{{graphics::CommandStatus::Unsupported,
                                        "stage provider has no separate admission check"}};
    }
    virtual graphics::RendererResult<AdmittedComputeStage> lookup_compute(
        const graphics::RenderStop&, const graphics::GpuAddressSpace&) {
        return graphics::RendererError{{graphics::CommandStatus::Unsupported,
                                        "stage provider has no compute lookup"}};
    }
};
struct RendererLimits {
    std::uint64_t resident_bytes = 0, staging_bytes = 0;
    std::uint32_t inflight_submissions = 0, commands_per_submission = 0;
    // When set, submission boundaries leave GPU-dirty storage unresolved; the
    // owner must call the registry's submission_boundary() before any
    // guest-visible effect (see ResourceMemoryRegistry::internal_boundary()).
    bool defer_gpu_writeback = false;
};
class DeviceContext;
struct RendererServices {
    std::shared_ptr<DeviceContext> device;
    std::shared_ptr<graphics::ResourceMemory> memory;
    std::shared_ptr<GraphicsStageProvider> stages;
    std::shared_ptr<graphics::TextureCache> textures;
    std::shared_ptr<graphics::BufferCache> buffers;
    std::shared_ptr<PipelineStateCache> pipelines;
    std::shared_ptr<BindingPlanner> bindings;
    std::shared_ptr<SubmissionQueue> submissions;
};

// L10's persistent device/queue/cache owner. Portable callers use Renderer;
// native implementation is renderer.mm. Shared services must refer to one
// device/registry/queue domain (check cache/queue domain(), pipeline context
// and device identity; every request.memory must equal services.memory). Null
// services, zero budgets, unsupported Metal capabilities or an unverified
// requested native-tracking mode fail before accepting work.
// Construction does not install engine callbacks or replace execute_metal_draw's
// synchronous behavior. L11 supplies the live stage provider and effect owners.
class MetalRenderer : public graphics::Renderer {
public:
    ~MetalRenderer() override = default;
    virtual graphics::RendererStatus set_commit_hold(bool held) {
        if (!held) return graphics::renderer_success();
        return graphics::RendererError({graphics::CommandStatus::Unsupported,
                                        "Metal renderer does not support commit hold"});
    }
    virtual graphics::RendererStatus abort_commit_hold(graphics::CommandResult error) {
        static_cast<void>(error);
        return graphics::RendererError({graphics::CommandStatus::Unsupported,
                                        "Metal renderer does not support commit hold abort"});
    }
    virtual bool commit_held() const noexcept { return false; }
    // Engine stores reported by GPU range (see BindingPlanner::note_engine_store).
    virtual void note_engine_store(std::uint64_t gpu_address, std::uint64_t size) {
        static_cast<void>(gpu_address);
        static_cast<void>(size);
    }
    virtual void set_engine_stores_reported(bool reported) { static_cast<void>(reported); }
    virtual bool can_accept_while_held() const noexcept { return true; }
};
[[nodiscard]] graphics::RendererResult<std::shared_ptr<MetalRenderer>>
create_renderer(RendererLimits limits, RendererServices services);

} // namespace armigo::platform::metal
