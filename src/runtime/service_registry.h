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

#include <armigo/horizon/ipc.h>
#include <armigo/platform/presenter.h>
#include <armigo/romfs/storage.h>
#include <armigo/services/addon_content_catalog.h>
#include <armigo/services/graphics_presentation.h>
#include <armigo/runtime/guest_clock.h>
#include <armigo/runtime/guest_memory.h>
#include <armigo/runtime/launch_profile.h>
#include <armigo/runtime/presentation_clock.h>
#include <armigo/services/operation_mode.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>

namespace armigo::services::vi {
class DisplaySession;
}

namespace armigo::runtime {

// A shader package opening on a background thread, started by the launch
// before the guest image loads so the two overlap (see
// start_shader_package_open).
class PendingShaderPackage;

struct ServiceRegistryDependencies {
    LaunchProfile profile;
    std::uint64_t process_id = 0;
    services::OperationMode operation_mode = services::OperationMode::Console;
    std::shared_ptr<horizon::HandleTable> handles;
    std::shared_ptr<GuestMemory> memory;
    std::shared_ptr<GuestClock> clock;
    std::shared_ptr<PresentationClock> presentation_clock;
    std::shared_ptr<platform::Presenter> presenter;
    std::shared_ptr<romfs::Storage> romfs;
    // The application's installed add-on content (aoc:u lists it, fsp-srv
    // opens its data storage). Null behaves as none installed, with no base id.
    std::shared_ptr<const services::AddOnContentCatalog> addon_content;
    // The display version the application reports when its package has no
    // control data (a NUP), derived from the package version.
    std::optional<std::string> display_version;
    std::filesystem::path shader_package_directory;
    // The package from shader_package_directory, when already opening; the
    // registry waits for it instead of opening the directory itself.
    std::shared_ptr<PendingShaderPackage> pending_shader_package;
    // The launch replaced the title's NVN driver: install the Metal driver
    // with the shader package and bind its native windows to vi.
    bool nvn_driver = false;
    std::string nvn_pso_manifest;
    // The launch preselects the local user (LaunchPlan's startup_user).
    bool startup_user = false;
    std::function<LaunchStatus(const platform::PresentedFrame&)> observe_frame;
    // Whether presentation returns a frame's completed pixels; see
    // PresentationDependencies::wants_readback. Empty reads back every frame.
    std::function<bool(std::uint64_t)> wants_readback;
    // Called with each vi display session the registry creates, before the
    // guest can use it. A host that produces frames in process binds its
    // native windows here.
    std::function<void(const std::shared_ptr<services::vi::DisplaySession>&)>
        observe_display;
    // The renderer requires native guest access to be stopped for the whole
    // synchronous callback. The returned owner holds that exclusion until the
    // callback completes. An absent callback refuses GPU execution.
    std::function<LaunchResult<std::shared_ptr<void>>()> acquire_graphics_access;
};

struct ServiceRegistryStatistics {
    std::uint64_t named_port_connections = 0;
    std::uint64_t service_connections = 0;
    std::uint64_t queued_frames = 0;
    std::uint64_t completed_frames = 0;
    std::uint64_t completed_draws = 0;
    std::uint64_t completed_clears = 0;
};

class ServiceRegistry {
public:
    virtual ~ServiceRegistry() = default;

    // Suitable for IpcDispatcher's connector callback. The context is a
    // borrowed ServiceRegistry pointer; the process runtime retains this object
    // until every per-thread dispatcher has been destroyed.
    static horizon::SessionConnection connector(void* context,
                                                 std::string_view port);
    virtual horizon::SessionConnection connect_named_port(
        std::string_view port) = 0;

    virtual std::optional<LaunchFailure> failure() const = 0;
    virtual void request_stop() noexcept = 0;
    virtual LaunchStatus drain(std::optional<ClockInstant> deadline) = 0;
    // Second shutdown phase, after every guest thread has joined. A failure
    // retains the process owners so a later call can finish draining workers.
    virtual LaunchStatus finish_guest_shutdown(std::optional<ClockInstant>) {
        return std::monostate{};
    }
    virtual ServiceRegistryStatistics statistics() const noexcept = 0;
};

// Returns the callback unchanged when observation is disabled. An enabled
// observer borrows guest memory for each bounded copy; it must not retain the
// handle-table owner through a VI callback owned by that same handle table.
[[nodiscard]] services::GraphicsPresenter attach_presentation_observer(
    services::GraphicsPresenter callback,
    const std::shared_ptr<PresentationClock>& clock,
    const std::shared_ptr<GuestMemory>& memory);

// Starts opening the shader package in a directory on a background thread.
// Null when there is no directory or the build has no trusted package chain
// (the registry then reports that itself).
[[nodiscard]] std::shared_ptr<PendingShaderPackage>
start_shader_package_open(const std::filesystem::path& directory);

[[nodiscard]] LaunchResult<std::shared_ptr<ServiceRegistry>>
create_service_registry(ServiceRegistryDependencies dependencies);

} // namespace armigo::runtime
