// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/guest_dispatch.h>
#include <armigo/execution/call_hook.h>
#include <armigo/compat/patches.h>
#include <armigo/runtime/launch_profile.h>
#include <armigo/runtime/module_set.h>
#include <armigo/platform/thread_slot.h>
#include <vector>
#include <armigo/runtime/prepared_modules.h>
#include <armigo/runtime/embedded_modules.h>
#include <memory>
#include <string>

namespace armigo::runtime {

// The original executable bootstrap follows the verified firmware-20.5.0
// process entry and guest-loader ownership contract. No fixed SDK export
// sequence, duplicated relocation/constructors or guessed TLS structures are
// permitted. Lane F creates stack/TLS/handles; this plan identifies code and
// ownership only.
enum class LaunchOwnership { unresolved, guest_bootstrap, host_loader };
enum class LaunchEntryAbi { unresolved, synthetic_exact_sp, original_process, host_startup };
struct LaunchPlan {
    // Artifact manifest indices, in load order: an exact permutation. All
    // ModuleLocations retain manifest indices, not the reordered ModuleSet index.
    std::vector<std::size_t> module_order;
    ModuleLocation entry;
    // original_process selects the verified exact-SP process bootstrap. Lane F
    // supplies x0=0 and x1=the main-thread handle at the execution boundary.
    LaunchEntryAbi entry_abi = LaunchEntryAbi::unresolved;
    LaunchOwnership relocation_owner = LaunchOwnership::unresolved;
    LaunchOwnership initialization_owner = LaunchOwnership::unresolved;
};
// One SDK startup function the host calls on the main thread, in order, each
// returning before the next. Used when the host loader owns initialization.
struct StartupCall {
    enum class Argument { none, main_thread_handle, host_arguments };
    std::string phase;
    memory::GuestAddress function = 0;
    std::uint64_t x0 = 0;
    // Replaces x0 with a value the runtime only knows when the thread starts.
    Argument argument = Argument::none;
};
struct PreparedEntryLanding {
    memory::GuestAddress logical_root = 0;
    memory::GuestAddress landing = 0;
};
struct PreparedLaunch {
    // Retains existing module mappings; never a second module ownership model.
    std::shared_ptr<ModuleSet> modules;
    std::shared_ptr<const execution::GuestSiteTable> sites;
    std::shared_ptr<const execution::CallHookTable> call_hooks;
    std::shared_ptr<const compat::InstalledPatch> compatibility_patch;
    // Same key used for process-specific x18 fixups and every native binding.
    // Its native offset is resolved during loading, never persisted offline.
    std::shared_ptr<platform::ThreadSlotKey> thread_slot;
    LaunchPlan plan;
    memory::GuestAddress entry_address = 0;
    // Compatibility alias for the initial process root's trusted landing.
    // Synthetic launches leave this zero.
    memory::GuestAddress process_entry_address = 0;
    // Strictly sorted by logical_root. Original-process launches contain one
    // runtime-generated landing for every admitted root; synthetic launches
    // leave the vector empty. Landings are sealed appendix code, never roots.
    std::vector<PreparedEntryLanding> original_entry_landings;
    // host_startup: the main thread's call sequence; entry_address is its first.
    std::vector<StartupCall> startup_calls;
    // The SDK's nvnBootstrapLoader is redirected to Armigo's NVN driver.
    bool nvn_driver = false;
    // The title's pipeline warm-up seed manifest path, when it names one.
    std::string nvn_pso_manifest;
    // The title requires a startup user: the launch preselects the local user.
    bool startup_user = false;
    // Admitted title mode after renderer filtering; zero means no rate policy.
    std::uint32_t title_frame_rate = 0;
};
// No guest execution. Failure publishes no addresses; successful objects retain
// mappings/site tables through all thread bindings. Synthetic relocations are
// host-applied once; original relocation and initialization remain guest-owned. A
// host_loader initialization request is unsupported until an explicit ordered
// initialization API is reviewed; callers must not improvise SDK calls.
[[nodiscard]] LaunchResult<PreparedLaunch> load_prepared_launch(
    const loader::ExeFsDirectoryInput& input, const PreparedModuleArtifacts& artifacts,
    const LaunchProfile& profile, const LaunchPlan& plan,
    std::shared_ptr<platform::ThreadSlotKey> thread_slot,
    const ModuleLoadOptions& options = {},
    std::span<const compat::PatchDescription> patches = {});
struct NativeLaunchOptions {
    // Selects key-page rewrites while preparing. This still prepares at load
    // time; artifact capture/mapping is a separate operation, not implied here.
    bool embedded_preparation = false;
    // Optional pack-time output, replaced only after a successful full launch
    // plan. Requires embedded_preparation; contains no process addresses.
    EmbeddedNativeArtifacts* capture = nullptr;
    // Mutually exclusive with capture. Loading these maps prepared pages and
    // binds data/metadata only; no instruction scan or rewrite runs.
    const EmbeddedNativeArtifacts* artifacts = nullptr;
    std::span<const EmbeddedNativeBacking> backings{};
};
// Derives order and entry from supported executable metadata, not directory
// order. Synthetic inputs use the same validation with an explicit test profile.
// Loads NPDM-backed input without offline artifacts: the host links the
// modules, prepares every text word at load time and owns SDK startup.
[[nodiscard]] LaunchResult<PreparedLaunch> load_native_launch(
    const loader::ExeFsDirectoryInput& input, const LaunchProfile& profile,
    std::shared_ptr<platform::ThreadSlotKey> thread_slot,
    std::span<const compat::PatchDescription> patches = {},
    const NativeLaunchOptions& options = {});
[[nodiscard]] LaunchResult<LaunchPlan> plan_launch(
    const loader::ExeFsDirectoryInput& input, const LaunchProfile& profile,
    const PreparedModuleArtifacts& artifacts);

} // namespace armigo::runtime
