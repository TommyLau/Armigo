// ============================================================================
// Armigo NVN - Driver Backends
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "objects.h"
#include <armigo/platform/metal_draw.h>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>

namespace armigo::platform { class MetalShaderPackage; }
namespace armigo::nvn {

// A stage's own constants (hardware constant buffer 1): the tail of its code
// buffer, from the control section's constant offset to the code size.
struct ProgramConstants {
    std::uint64_t address = 0;
    std::uint32_t size = 0;
};

// A stage the resolver prepares in the background. wait() blocks until it is
// prepared and returns it, or nullptr when its preparation failed (the
// resolver reports why); it may be called from any thread, any number of times.
class PendingStage {
public:
    virtual ~PendingStage() = default;
    [[nodiscard]] virtual std::shared_ptr<const platform::CompiledGraphicsStage> wait() = 0;
};

// The stages a program was published without. retain_program moves them into
// the program's stages (once) before it returns the program.
struct PendingStages {
    std::array<std::shared_ptr<PendingStage>, 6> stages;
    std::once_flag once;
    std::atomic<bool> finished{false};
};

// An API 231 compute stage's driver data (hardware constant buffer 0), laid
// out per shader from its control section. Words 1-3 hold the dispatch's
// workgroup counts; from 0x10 come the push block (push_bytes, rounded up to
// 16), then 16-byte uniform and storage buffer entries (address, size), then
// 8-byte handles: textures, separate textures, separate samplers and images,
// `size` bytes in all.
// The driver data the replay binds for a stage (hardware constant buffer 0).
inline constexpr std::uint32_t compute_driver_data_bytes = 0xa00;
struct ComputeDriverLayout {
    std::uint32_t push_bytes = 0;
    std::uint32_t uniforms = 0, storages = 0;
    std::uint32_t textures = 0, separate_textures = 0, separate_samplers = 0, images = 0;
    std::uint32_t size = 0;
    std::uint32_t uniform_base() const { return 0x10 + ((push_bytes + 15) & ~15u); }
    std::uint32_t storage_base() const { return uniform_base() + 16 * uniforms; }
    std::uint32_t texture_base() const { return storage_base() + 16 * storages; }
    std::uint32_t separate_texture_base() const { return texture_base() + 8 * textures; }
    std::uint32_t separate_sampler_base() const { return separate_texture_base() + 8 * separate_textures; }
    std::uint32_t image_base() const { return separate_sampler_base() + 8 * separate_samplers; }
};

struct ResolvedProgram {
    ProgramState state{};
    // Complete for every caller of retain_program. A program published with
    // pending stages gets them filled in by the first retain_program; nothing
    // reads them before that (command recording only keeps the program).
    std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 6> stages;
    std::array<ProgramConstants, 6> constants{};
    // API 231 compute stages only.
    std::optional<ComputeDriverLayout> compute_layout;
    // Set once before publication; command recording can retain a live
    // snapshot directly without entering the serial registry.
    std::weak_ptr<const ResolvedProgram> owner;
    std::shared_ptr<PendingStages> pending;
    // Bit s: stage s was pending and its preparation failed, so stages[s] is
    // empty although the guest set it. Final together with stages.
    std::uint32_t failed_stages = 0;
};
using ProgramResolver = bool (*)(void*, const DeviceState*, std::span<const NVNshaderData>,
    std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 6>&, std::string&);
// A resolver that may leave stages to background preparation: for each stage
// it fills either `stages` or `pending`. SetShaders then succeeds once every
// stage is accounted for, and the program waits for its pending stages when
// the replay first retains it.
using DeferredProgramResolver = bool (*)(void*, const DeviceState*, std::span<const NVNshaderData>,
    std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 6>&,
    std::array<std::shared_ptr<PendingStage>, 6>&, std::string&);

using ProgramExtendedValidator = bool (*)(void*, const NVNshaderDataExt&, std::string&);

// Configure before guest execution. The context outlives every SetShaders call:
// resolution runs outside the program lock, so replacing the resolver does not
// wait for a resolution already in progress.
void set_program_resolver(ProgramResolver resolver, void* context,
                          ProgramExtendedValidator extended = nullptr) noexcept;
// Replaces any resolver with a deferred one (same rules as above).
void set_deferred_program_resolver(DeferredProgramResolver resolver, void* context,
                                   ProgramExtendedValidator extended = nullptr) noexcept;
// Retains executable state by serial, including after the original object is
// finalized. Waits for the program's pending stages, if any, so the result's
// stages are complete; a pending stage that failed stays empty.
[[nodiscard]] std::shared_ptr<const ResolvedProgram> retain_program(const ProgramState& state);
// As retain_program, but never waits: a program whose pending stages are not
// all prepared yet is not returned, and `pending` is set instead.
[[nodiscard]] std::shared_ptr<const ResolvedProgram> retain_ready_program(const ProgramState& state,
                                                                          bool& pending);
// The caller must hold a live program object or an existing snapshot owner.
// Does not wait for pending stages: the caller must not read the stages.
[[nodiscard]] std::shared_ptr<const ResolvedProgram> retain_program_snapshot(const ProgramState& state);
void register_program_entries() noexcept;
// Uses the same program identities and validated artifacts as command-stream rendering.
// Stages the package lacks are prepared at run time (see nvn_replay_programs.mm);
// `title_cache` is the title's cache directory: the run-time stage cache is
// its shaders/ unless ARMIGO_SHADER_CACHE names another place.
void set_program_package(std::shared_ptr<platform::MetalShaderPackage> package,
                         const std::string& title_cache = {});

} // namespace armigo::nvn
