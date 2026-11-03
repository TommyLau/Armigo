// ============================================================================
// Armigo Services - Initial Applet Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/horizon/event.h>
#include <armigo/services/operation_mode.h>
#include <armigo/services/software_keyboard.h>

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <deque>
#include <vector>

namespace armigo::horizon {
class TransferMemoryRecord;
}

namespace armigo::services::am {

// Portable launch policy, separate from application metadata and permissions.
enum class GameplayRecordingPolicy : std::uint8_t { Unavailable };

// Host-owned running-notification state for the ApplicationFunctions
// NotifyRunning request. It records whether the current launch reports running
// notifications at all, whether the notification recipient is ready, and
// whether the single allowed delivery already happened. The default-constructed
// state keeps notifications disabled: that is an explicit Armigo host launch
// policy for headless starts, not a recovered firmware initial flag and not an
// application property. Full launch-flag sourcing remains unresolved.
// A host that owns the launch shares one
// state object with every session serving that launch.
class ApplicationRunningState final {
public:
    ApplicationRunningState(bool notifications_enabled = false,
                            bool delivery_ready = false,
                            std::function<void()> observer = {});
    bool enabled() const noexcept;
    bool delivery_ready() const noexcept;
    bool notification_sent() const noexcept;

private:
    friend class AppletServiceSession;
    // Reports the enabled state and, when delivery is ready, marks and performs
    // the one-time delivery. Runs with the owning session's request mutex held,
    // so the observer must not re-enter that session.
    bool notify();
    const bool enabled_;
    const bool delivery_ready_;
    const std::function<void()> observer_;
    std::atomic<bool> notification_sent_{false};
};

// Initial appletOE root for a registered host application. The supplied process
// record and handle table identify the same caller as the trusted process ID.
// Proxy children beyond this startup factory remain unsupported.
class AppletServiceSession final : public horizon::ClientSession {
public:
    AppletServiceSession(std::uint64_t process_id,
                         std::shared_ptr<horizon::ProcessRecord> process,
                         horizon::HandleTable& handles,
                         GameplayRecordingPolicy recording_policy = GameplayRecordingPolicy::Unavailable,
                         std::shared_ptr<ApplicationRunningState> running_state = {},
                         OperationMode operation_mode = OperationMode::Console);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    // Launch policy for an application whose control data requires a user at
    // startup: the launcher preselects `uid` and queues it once as the
    // preselected-user launch parameter (kind 2), which PopLaunchParameter
    // hands out as a storage object.
    void preselect_user(const std::array<std::byte, 16>& uid);
    // Host input for the software keyboard applet. The default is
    // headless_keyboard_input. It runs with the session's request lock held, on
    // the guest thread that starts the applet.
    void set_keyboard_input(KeyboardInput input);
    const char* diagnostic() const;
    std::size_t domain_object_count() const;
    bool exit_to_library_on_next_program_enabled() const;
    // Host launch policy grants foreground ownership once the application is
    // ready to run. The transition produces a readable focus notification.
    void grant_foreground();

    // CommonStateGetter command66 stores one small enum argument. Firmware
    // names were not recovered; the value is recorded as host state only.
    std::optional<std::uint32_t> common_state_command66() const;
    // Host inspection of the logical state controlled by CommonStateGetter
    // commands 66 and 67. This does not model host CPU frequency changes.
    bool common_state_boost_active() const;

    // SelfController one-byte boolean setters: command11 operation-mode-changed
    // notification, command12 performance-configuration-changed notification and
    // command14 restart-message flag. Each returns the stored value.
    std::optional<bool> self_controller_flag(std::uint32_t command) const;

    // SelfController command1 LockExit and command2 UnlockExit: whether the
    // application has asked to receive exit requests instead of being ended.
    // Nothing requests an exit yet, so the state is only kept.
    bool exit_locked() const;

    // SelfController command10 SetScreenShotPermission: 0 inherit, 1 allow,
    // 2 deny. Kept only; this runtime takes no captures for the guest.
    std::uint32_t screenshot_permission() const;

    // ApplicationFunctions command22 stores the caller's termination result.
    // Terminating the process from it is not implemented; the recorded value is
    // host observation state, exactly as the firmware keeps it for later.
    std::uint32_t terminate_result() const;

private:
    const std::uint64_t process_id_;
    const OperationMode operation_mode_;
    const GameplayRecordingPolicy recording_policy_;
    // Shared by every session the host associates with one launch. A null
    // constructor argument installs the default disabled state.
    const std::shared_ptr<ApplicationRunningState> running_state_;
    std::shared_ptr<horizon::ProcessRecord> process_;
    horizon::HandleTable* handles_;
    enum class ObjectKind {
        application_proxy, application_functions, common_state, self_controller,
        window_controller, audio_controller, display_controller, library_creator, debug_functions,
        storage, storage_accessor, library_applet, closed, transfer_storage,
    };
    using Bytes = std::shared_ptr<std::vector<std::byte>>;
    // A storage passed to an applet: bytes the application wrote, or a large
    // storage over application memory lent through transfer memory.
    struct AppletStorage {
        Bytes data;
        std::shared_ptr<horizon::TransferMemoryRecord> memory;
        std::uint64_t size = 0;
    };
    // A library applet the application created. Armigo runs no applet
    // programs: the network-connection applet (15) and the full-screen
    // software keyboard (17) finish as soon as they are started; other applets
    // are unsupported.
    struct LibraryApplet {
        std::uint32_t id = 0;
        std::uint32_t mode = 0;
        bool started = false;
        std::vector<AppletStorage> in;
        std::deque<Bytes> out;
        std::shared_ptr<horizon::Event> state_changed;
        // Never signaled: neither applet sends interactive data.
        std::shared_ptr<horizon::Event> interactive_out;
    };
    struct DomainObject {
        ObjectKind kind;
        std::shared_ptr<horizon::KernelObject> copied_object;
        // The bytes of a storage object and of its accessors.
        Bytes data;
        std::shared_ptr<LibraryApplet> applet;
        // A transfer storage's memory and size.
        std::shared_ptr<horizon::TransferMemoryRecord> memory;
        std::uint64_t memory_size = 0;
    };
    // Runs the software keyboard for a started applet and queues its output.
    const char* run_keyboard(LibraryApplet& applet, const horizon::GuestBuffers* buffers);
    horizon::IpcReply handle(const horizon::IpcMessage&, std::uint64_t sender_pid,
                             const horizon::GuestBuffers* buffers);
    std::vector<DomainObject> objects_;
    bool domain_ = false;
    bool exit_to_library_on_next_program_ = false;
    std::uint32_t terminate_result_ = 0;
    std::optional<std::uint32_t> common_state_command66_;
    bool common_state_boost_active_ = false;
    std::array<std::optional<bool>, 3> self_controller_flags_{};
    std::uint32_t idle_time_detection_extension_ = 0;
    bool exit_locked_ = false;
    std::uint32_t screenshot_permission_ = 0;
    // Launch parameters not yet popped, by kind (1 user channel, 2 preselected
    // user, 3 unknown); each is handed out once.
    std::array<Bytes, 3> launch_parameters_{};
    // Layer ids handed out by SelfController command44. The application's first
    // managed layer must have a non-zero id, so the counter starts at one.
    std::uint64_t next_layer_id_ = 0;
    std::shared_ptr<horizon::Event> suspended_tick_event_;
    // Signaled while the application may launch a library applet.
    std::shared_ptr<horizon::Event> library_applet_launchable_event_;
    std::shared_ptr<horizon::Event> gpu_error_event_;
    std::shared_ptr<horizon::Event> common_state_event_;
    std::shared_ptr<horizon::Event> display_resolution_event_;
    // Readable event the application waits on for application state changes.
    std::shared_ptr<horizon::Event> application_state_event_;
    bool focused_ = false;
    bool focus_notification_pending_ = false;
    bool foreground_rights_acquired_ = false;
    bool focus_notifications_enabled_ = true;
    bool background_mode_enabled_ = false;
    bool focus_suspending_enabled_ = true;
    bool out_of_focus_suspending_enabled_ = false;
    KeyboardInput keyboard_input_ = headless_keyboard_input;
    const char* diagnostic_ = nullptr;
    mutable std::mutex mutex_;
};

} // namespace armigo::services::am
