// ============================================================================
// Armigo NVN - Driver Backends
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "programs.h"
#include "program_capture.h"
#include "entry_table.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace armigo::nvn {
namespace {
constexpr auto program_tag = object_tag('P', 'R', 'O', 'G');
// The installed driver is never destroyed, and its replay thread binds
// programs until the process ends, so the registry is never destroyed either:
// exit's static destructors must not free it under that thread.
std::mutex& program_mutex = *new std::mutex;
std::atomic<std::uint64_t> next_serial{1};
ProgramResolver resolver = nullptr;
DeferredProgramResolver deferred_resolver = nullptr;
ProgramExtendedValidator extended_validator = nullptr;
void* resolver_context = nullptr;
auto& live_programs = *new std::unordered_map<std::uint64_t, std::shared_ptr<const ResolvedProgram>>;
auto& recorded_programs = *new std::unordered_map<std::uint64_t, std::weak_ptr<const ResolvedProgram>>;

ProgramState& state(NVNprogram* object) { return *reinterpret_cast<ProgramState*>(object); }
void diagnostic(const char* text) { armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN program: %s\n", text); }

NVNboolean initialize(NVNprogram* object, NVNdevice* device) {
    if (!object) return 0;
    std::memset(object, 0, sizeof(*object));
    state(object).tag = program_tag;
    state(object).device = device;
    return 1;
}
void finalize(NVNprogram* object) {
    if (!object) return;
    std::lock_guard lock(program_mutex);
    live_programs.erase(state(object).serial);
    auto entry = recorded_programs.find(state(object).serial);
    if (entry != recorded_programs.end() && entry->second.expired()) recorded_programs.erase(entry);
    std::memset(object, 0, sizeof(*object));
}

NVNboolean set_shaders(NVNprogram* object, int count, const NVNshaderData* shaders) {
    program_capture::Call captured(reinterpret_cast<std::uint64_t>(object), count,
                                   reinterpret_cast<std::uint64_t>(shaders), false);
    if (!object || state(object).tag != program_tag || count < 0 || count > 6 || (count && !shaders)) {
        diagnostic("Invalid shader descriptor range or program object");
        return 0;
    }
    try {
        // Resolution reads guest memory and the shader package (which has its
        // own lock) and can take milliseconds on first use. It runs outside
        // program_mutex, so the replay thread's retain_program never waits
        // for it; only the publication below takes the lock.
        ProgramResolver resolve = nullptr;
        DeferredProgramResolver resolve_deferred = nullptr;
        void* context = nullptr;
        {
            std::lock_guard lock(program_mutex);
            resolve = resolver;
            resolve_deferred = deferred_resolver;
            context = resolver_context;
        }
        auto executable = std::make_shared<ResolvedProgram>();
        executable->state.tag = program_tag;
        executable->state.device = state(object).device;
        executable->state.stage_count = static_cast<std::uint32_t>(count);
        std::string error;
        std::array<std::shared_ptr<PendingStage>, 6> pending;
        const auto* device = reinterpret_cast<const DeviceState*>(state(object).device);
        const std::span<const NVNshaderData> descriptors{shaders, static_cast<std::size_t>(count)};
        if (count && !(resolve_deferred ? resolve_deferred(context, device, descriptors, executable->stages, pending, error)
                                        : resolve && resolve(context, device, descriptors, executable->stages, error))) {
            diagnostic(error.empty() ? "No shader package resolver is configured" : error.c_str());
            return 0;
        }
        unsigned resolved_count = 0;
        bool any_pending = false;
        for (std::size_t i = 0; i != pending.size(); ++i) {
            if (executable->stages[i] && pending[i]) {
                diagnostic("Shader resolver returned a stage both prepared and pending");
                return 0;
            }
            resolved_count += bool(executable->stages[i]) || bool(pending[i]);
            any_pending |= bool(pending[i]);
            if (const auto& shader = executable->stages[i]) {
                const auto logical = shader->task ? 6u : shader->mesh ? 7u : static_cast<unsigned>(i);
                executable->logical_stages |= 1u << logical;
                if (shader->mesh) executable->mesh_uses_task = shader->mesh->launched_by_task;
            }
        }
        if (any_pending) {
            executable->pending = std::make_shared<PendingStages>();
            executable->pending->stages = std::move(pending);
        }
        if (resolved_count != static_cast<unsigned>(count)) {
            diagnostic("Shader resolver returned an incomplete or duplicate stage set");
            return 0;
        }
        // Control section: constant byte size at +0x6fc, constant offset from
        // the shader data at +0x700, stage at +0x714. A stage without
        // constants has size zero. API 231 controls are native: constant size
        // at +0x3c, offset at +0x40 and stage selector byte at +0x56, and the
        // constants are hardware constant buffer 1 as well.
        const auto api = selected_api_version();
        const bool native = api && api->major == 231;
        for (int i = 0; i < count; ++i) {
            const auto* control = static_cast<const std::byte*>(shaders[i].control);
            if (!control) continue;
            std::uint32_t c1_size = 0, c1_offset = 0, stage = 0;
            std::memcpy(&c1_size, control + (native ? 0x3c : 0x6fc), 4);
            std::memcpy(&c1_offset, control + (native ? 0x40 : 0x700), 4);
            if (native) stage = std::to_integer<std::uint32_t>(control[0x56]);
            else std::memcpy(&stage, control + 0x714, 4);
            if (stage < 8) executable->logical_stages |= 1u << stage;
            if (native && stage == 7) {
                const auto route = std::to_integer<std::uint8_t>(control[0x5e]);
                executable->mesh_uses_task = route != 0;
                std::uint16_t auxiliary = 0;
                std::memcpy(&auxiliary, control + 0x68, sizeof auxiliary);
                executable->mesh_primitive_header = auxiliary != 0;
                if (captured.id()) {
                    captured.metadata(i, stage, route, auxiliary);
                    executable->capture_mesh_metadata = true;
                    executable->captured_mesh_route = route;
                    executable->captured_auxiliary_word = auxiliary;
                }
            } else if (captured.id()) captured.metadata(i, stage);
            const auto artifact = native && stage == 6 ? 3u : native && stage == 7 ? 4u : stage;
            if (artifact < executable->constants.size() && c1_size)
                executable->constants[artifact] = {shaders[i].data + c1_offset, c1_size};
            // Compute driver data counts: push words at +0xa9, uniform and
            // storage entries at +0xb1 and +0xb2, texture, separate texture,
            // separate sampler and image handles at +0xb3..+0xb6, and the
            // total byte size (u16) at +0xba, which the counts must make. A
            // program whose counts do not, or whose driver data is larger than
            // the replay's driver block, is kept without a layout, and the
            // replay refuses its dispatches.
            if (native && stage == NVN_SHADER_STAGE_COMPUTE) {
                const auto byte = [&](std::size_t offset) { return std::to_integer<std::uint32_t>(control[offset]); };
                ComputeDriverLayout layout;
                layout.push_bytes = 4 * byte(0xa9);
                layout.uniforms = byte(0xb1);
                layout.storages = byte(0xb2);
                layout.textures = byte(0xb3);
                layout.separate_textures = byte(0xb4);
                layout.separate_samplers = byte(0xb5);
                layout.images = byte(0xb6);
                layout.size = byte(0xba) | byte(0xbb) << 8;
                if (layout.image_base() + 8 * layout.images != layout.size)
                    diagnostic("Compute control section's driver data counts do not make its size");
                else if (layout.size > compute_driver_data_bytes)
                    armigo::diag::write(armigo::diag::Level::error,
                                        "Armigo NVN program: compute driver data of %u bytes exceeds the %u bytes the "
                                        "replay binds\n", layout.size, compute_driver_data_bytes);
                else
                    executable->compute_layout = layout;
            }
        }
        executable->capture_call = captured.id();
        executable->capture_matches = captured.matches();
        // Descriptors are diagnostic metadata only; replay consumes the retained artifacts.
        if (count) std::copy_n(shaders, count, executable->state.stages);
        std::lock_guard lock(program_mutex);
        // A guest that finalized the object while it resolved gets nothing
        // published into it (finalize clears the tag).
        if (state(object).tag != program_tag) {
            diagnostic("Program was finalized while its shaders were set");
            return 0;
        }
        executable->state.serial = next_serial.fetch_add(1, std::memory_order_relaxed);
        executable->state.snapshot = &executable->state;
        executable->owner = executable;
        live_programs.emplace(executable->state.serial, executable);
        recorded_programs.emplace(executable->state.serial, executable);
        const auto old_serial = state(object).serial;
        state(object) = executable->state;
        live_programs.erase(old_serial);
        for (auto it = recorded_programs.begin(); it != recorded_programs.end();) {
            if (it->second.expired()) it = recorded_programs.erase(it);
            else ++it;
        }
        captured.finish(true, executable->state.serial);
        return 1;
    } catch (const std::exception& error) {
        diagnostic(error.what());
        return 0;
    }
}
NVNboolean set_shaders_ext(NVNprogram* object, int count, const NVNshaderDataExt* shaders) {
    program_capture::Call captured(reinterpret_cast<std::uint64_t>(object), count,
                                   reinterpret_cast<std::uint64_t>(shaders), true);
    if (count < 0 || count > 6 || (count && !shaders)) return 0;
    NVNshaderData basic[6]{};
    for (int i = 0; i < count; ++i) {
        if (shaders[i].extra) {
            // Validation reads guest memory, so it too runs outside the lock.
            ProgramExtendedValidator validate = nullptr;
            void* context = nullptr;
            {
                std::lock_guard lock(program_mutex);
                validate = extended_validator;
                context = resolver_context;
            }
            std::string error;
            if (!validate || !validate(context, shaders[i], error)) {
                diagnostic(error.empty() ? "Extended shader metadata validation is unavailable" : error.c_str());
                return 0;
            }
        }
        basic[i] = {shaders[i].data, shaders[i].control};
    }
    const auto result = set_shaders(object, count, basic);
    captured.finish(result != 0, std::nullopt);
    return result;
}
void set_label(NVNprogram*, const char*) {}
NVNboolean subroutines(NVNprogram*, int count, const NVNsubroutineLinkageMapPtr*) {
    if (count) diagnostic("Shader subroutine linkage is unsupported");
    return count == 0;
}
} // namespace

void set_program_resolver(ProgramResolver callback, void* context, ProgramExtendedValidator extended) noexcept {
    std::lock_guard lock(program_mutex);
    resolver = callback;
    deferred_resolver = nullptr;
    resolver_context = context;
    extended_validator = extended;
}
void set_deferred_program_resolver(DeferredProgramResolver callback, void* context,
                                   ProgramExtendedValidator extended) noexcept {
    std::lock_guard lock(program_mutex);
    resolver = nullptr;
    deferred_resolver = callback;
    resolver_context = context;
    extended_validator = extended;
}
static std::shared_ptr<const ResolvedProgram> recorded_program(const ProgramState& object) {
    std::lock_guard lock(program_mutex);
    const auto found = recorded_programs.find(object.serial);
    return found == recorded_programs.end() ? nullptr : found->second.lock();
}
std::shared_ptr<const ResolvedProgram> retain_ready_program(const ProgramState& object, bool& pending) {
    pending = false;
    auto program = recorded_program(object);
    if (program && program->pending && !program->pending->finished.load(std::memory_order_acquire)) {
        pending = true;
        return nullptr;
    }
    return program;
}
std::shared_ptr<const ResolvedProgram> retain_program(const ProgramState& object) {
    auto program = recorded_program(object);
    // Pending stages are waited for outside the lock, so SetShaders on other
    // programs goes on meanwhile. The first caller fills the stages in; the
    // others wait for it (call_once), or see the finished flag after it.
    if (program && program->pending && !program->pending->finished.load(std::memory_order_acquire)) {
        auto& pending = *program->pending;
        std::call_once(pending.once, [&] {
            // The program was created mutable; only its stages change, and
            // nothing reads them before this returns.
            auto& filled = const_cast<ResolvedProgram&>(*program);
            for (std::size_t i = 0; i != filled.stages.size(); ++i) {
                if (!pending.stages[i]) continue;
                filled.stages[i] = std::exchange(pending.stages[i], nullptr)->wait();
                const auto& shader = filled.stages[i];
                const auto logical = shader && shader->task ? 6u : shader && shader->mesh ? 7u :
                    i == 3 && (filled.logical_stages & (1u << 6)) ? 6u :
                    i == 4 && (filled.logical_stages & (1u << 7)) ? 7u : static_cast<unsigned>(i);
                filled.logical_stages |= 1u << logical;
                if (shader && shader->mesh) filled.mesh_uses_task = shader->mesh->launched_by_task;
                if (!filled.stages[i]) filled.failed_stages |= 1u << i;
            }
            pending.finished.store(true, std::memory_order_release);
        });
    }
    return program;
}
std::shared_ptr<const ResolvedProgram> retain_program_snapshot(const ProgramState& object) {
    static_assert(std::is_standard_layout_v<ResolvedProgram>);
    static_assert(offsetof(ResolvedProgram, state) == 0);
    const auto* snapshot = static_cast<const ResolvedProgram*>(object.snapshot);
    return snapshot ? snapshot->owner.lock() : nullptr;
}
void register_program_entries() noexcept {
    install<Entry::nvnProgramInitialize>(initialize);
    install<Entry::nvnProgramFinalize>(finalize);
    install<Entry::nvnProgramSetDebugLabel>(set_label);
    install<Entry::nvnProgramSetShaders>(set_shaders);
    install<Entry::nvnProgramSetShadersExt>(set_shaders_ext);
    install<Entry::nvnProgramSetSubroutineLinkage>(subroutines);
}
} // namespace armigo::nvn
