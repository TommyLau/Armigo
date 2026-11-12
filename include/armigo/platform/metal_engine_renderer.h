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

#include <armigo/platform/metal_draw.h>
#include <armigo/graphics/renderer_types.h>

namespace armigo::platform {

// Environment selects ARMIGO_METAL_RENDERER=legacy|persistent. Unset means
// legacy; invalid values fail closed. An explicit mode ignores the environment.
enum class MetalEngineMode : std::uint8_t { Environment, Legacy, Persistent };
struct MetalEngineStatistics {
    std::uint64_t completed_draws = 0, completed_clears = 0, completed_blits = 0;
    std::uint64_t completed_dispatches = 0;
    std::uint64_t device_creations = 0, memory_domains = 0;
    std::uint64_t command_buffers = 0, render_passes = 0;
    // Completed draws divided by committed command buffers. One per draw
    // without batching; the batch size under deferred write-back.
    double draws_per_command_buffer = 0.0;
    // Nanoseconds the CPU spent blocked in drain waiting for GPU completion.
    std::uint64_t cpu_wait_ns = 0;
};
struct MetalEngineOptions {
    MetalEngineMode mode = MetalEngineMode::Environment;
    CompiledGraphicsLookup lookup = nullptr;
    void* lookup_context = nullptr;
    // Owns the package and lookup context. Required for a nonnull context.
    std::shared_ptr<void> lookup_owner;
    // Optional compute stage lookup over the same package and context. Absent
    // (or null with a present graphics lookup) makes every compute launch a
    // structured unsupported stop.
    CompiledComputeLookup compute_lookup = nullptr;
    // The lookup returns stage objects that never change after they are first
    // returned (a loaded shader package's stages): their content identities
    // are then derived once per stage instead of on every request.
    bool immutable_stages = false;
    std::shared_ptr<MetalEngineStatistics> statistics;
    // Zero sizes residency from the device: half its recommended working set,
    // at least 512 MiB. Textures take half of it, buffers a quarter.
    std::uint64_t resident_bytes = 0;
    std::uint64_t staging_bytes = 64ull << 20;
    std::uint64_t completion_timeout_ns = 30'000'000'000ull;
    // Persistent mode only. GPU results stay GPU-authoritative after each
    // draw/clear instead of being written back to guest memory; the returned
    // callbacks then provide RenderCallbacks::coherence, which the engine front
    // end calls before every guest-visible effect or front-end memory access.
    // Draws and clears encode into the current batch and return after the
    // renderer accepts them: they commit no command buffer and wait for no GPU
    // work. The batch commits (and, where the guest can observe results, the
    // CPU waits for coherent completion) at coherence(), mapping ownership
    // changes, the batch size below, and destruction. Destruction can commit
    // and wait but never writes guest memory: no CPU capability remains after
    // the callbacks return, so the last batch's bytes stay GPU-authoritative.
    // A GPU failure that surfaces asynchronously is returned, with its
    // original diagnostic, by the next draw, clear or coherence call and
    // stays sticky afterwards.
    bool defer_gpu_writeback = false;
    // Persistent deferred mode only: maximum draws and clears accepted into
    // one unflushed batch before it commits for budget pressure. Batched
    // draws still flush earlier at every coherence or ownership boundary.
    // A budget flush ends the open packet that repeats extend, so the next
    // draw is prepared in full; 256 balanced that against commit latency
    // for the first acceptance workload (64 and 1024 were both slower).
    std::uint32_t max_batch_draws = 256;
    // Registries created by this engine observe native guest CPU stores
    // through the process page write tracker (one per process, installed on
    // the first tracked registry). Disabled keeps conservative epoch
    // revalidation. ARMIGO_CPU_WRITE_TRACKING=0 overrides the runner default.
    bool cpu_write_tracking = false;
    // Guest addresses of the mappings are host pointers to their bytes (native
    // guest execution). Buffer bindings may then use zero-copy views of guest
    // memory, through zero-copy memory pools.
    bool guest_memory_host_addressable = false;
    // Persistent deferred mode only. A full coherence keeps rendered textures
    // GPU-resident instead of writing them back to guest memory because Metal textures own their content until CPU readback;
    // buffer results are still written back. Registry CPU access, ranged
    // coherence and mapping changes resolve them. Native guest CPU reads of
    // rendered texture memory do not observe GPU results; a native store into
    // it is reported (with CPU write tracking) as a structured stop.
    bool gpu_resident_textures = false;
};

// Register the returned callbacks with DriverSession::set_render_backend.
// They own one device/pipeline domain and persistent L4/L5 caches. Mapping pins
// are mandatory in persistent mode. Driver notifications retire old lookup
// identities after preserving live backing bytes; acquisitions use current VAs.
// Each accepted operation waits for GPU feedback AND coherent Complete before
// returning success, except under defer_gpu_writeback where batches commit at
// the boundaries described on that option. Native guest access must be stopped
// throughout the call. Calls are serialized internally; callback reentry is
// forbidden. Errors are sticky, preserve their exact diagnostics and retain
// failed dirty authority.
// No asynchronous guest access, runtime shader translation or silent fallback.
[[nodiscard]] graphics::RendererResult<graphics::RenderCallbacks>
create_metal_engine_backend(MetalEngineOptions options = {});

} // namespace armigo::platform
