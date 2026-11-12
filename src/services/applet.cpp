// ============================================================================
// Armigo Services - Initial Applet Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/common/diagnostics.h>
#include <armigo/services/applet.h>
#include <armigo/services/graphics_system.h>

#include <armigo/horizon/startup_queries.h>
#include <armigo/services/settings.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>
#include <new>
#include <stdexcept>

namespace armigo::services::am {
namespace {
using namespace horizon::cmif;
} // namespace

ApplicationRunningState::ApplicationRunningState(bool notifications_enabled, bool delivery_ready,
                                                 std::function<void()> observer)
    : enabled_(notifications_enabled), delivery_ready_(delivery_ready),
      observer_(std::move(observer)) {}

bool ApplicationRunningState::enabled() const noexcept { return enabled_; }

bool ApplicationRunningState::delivery_ready() const noexcept { return delivery_ready_; }

bool ApplicationRunningState::notification_sent() const noexcept {
    return notification_sent_.load(std::memory_order_acquire);
}

bool ApplicationRunningState::notify() {
    if (!enabled_) return false;
    if (delivery_ready_) {
        // Exactly one caller wins the exchange and delivers; an already-set
        // mark skips the observer while the enabled result stays true, matching
        // the firmware callee's once-only delivery including its null-recipient
        // branch, which still records the notification.
        const bool already_sent = notification_sent_.exchange(true, std::memory_order_acq_rel);
        if (!already_sent && observer_) observer_();
    }
    return true;
}

AppletServiceSession::AppletServiceSession(std::uint64_t process_id,
    std::shared_ptr<horizon::ProcessRecord> process, horizon::HandleTable& handles,
    GameplayRecordingPolicy recording_policy, std::shared_ptr<ApplicationRunningState> running_state,
    OperationMode operation_mode, AppletContract contract)
    : process_id_(process_id), contract_(contract), operation_mode_(operation_mode), recording_policy_(recording_policy),
      // The default disabled state is an explicit Armigo host launch policy for
      // headless starts, not a recovered firmware initial flag.
      running_state_(running_state ? std::move(running_state)
                                   : std::make_shared<ApplicationRunningState>()),
      process_(std::move(process)), handles_(&handles) {}

horizon::IpcReply AppletServiceSession::close_server() {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    closed_ = true;
    domain_ = false;
    application_proxy_configuration_.reset();
    std::vector<DomainObject>{}.swap(objects_);
    launch_parameters_ = {};
    suspended_tick_event_.reset();
    library_applet_launchable_event_.reset();
    gpu_error_event_.reset();
    common_state_event_.reset();
    display_resolution_event_.reset();
    application_state_event_.reset();
    process_.reset();
    keyboard_input_ = {};
    diagnostic_ = "Applet server session is closed";
    return {.disposition = horizon::IpcDisposition::handled};
}

horizon::IpcReply AppletServiceSession::request(const horizon::IpcMessage& message,
                                               std::uint64_t sender_pid) {
    return handle(message, sender_pid, nullptr);
}

horizon::IpcReply AppletServiceSession::request_with_buffers(const horizon::IpcMessage& message,
                                                            std::uint64_t sender_pid,
                                                            const horizon::GuestBuffers& buffers) {
    return handle(message, sender_pid, &buffers);
}

void AppletServiceSession::set_keyboard_input(KeyboardInput input) {
    const std::lock_guard lock(mutex_);
    keyboard_input_ = input ? std::move(input) : KeyboardInput(headless_keyboard_input);
}

void AppletServiceSession::preselect_user(const std::array<std::byte, 16>& uid) {
    // The preselected-user parameter: magic, a selected flag, then the user.
    constexpr std::uint32_t magic = 0xc79497ca;
    auto data = std::make_shared<std::vector<std::byte>>(0x88);
    for (std::size_t index = 0; index < 4; ++index)
        (*data)[index] = static_cast<std::byte>(magic >> (8 * index));
    (*data)[4] = std::byte{1};
    std::copy(uid.begin(), uid.end(), data->begin() + 8);
    const std::lock_guard lock(mutex_);
    launch_parameters_[1] = std::move(data);
}

horizon::IpcReply AppletServiceSession::handle(const horizon::IpcMessage& request,
                                               std::uint64_t sender_pid,
                                               const horizon::GuestBuffers* buffers) {
    // The parser reads a request's raw data at byte 16. Buffer descriptors
    // (X, A, B, W) precede the raw data, which starts at the next 16-byte
    // boundary after them; such a request is read from a copy with its raw
    // data moved to byte 16. The buffers themselves arrive separately.
    const horizon::IpcMessage* source = &request;
    horizon::IpcMessage moved;
    const auto request_type = read(request, 0) & 0xffff;
    if (const auto descriptors = read(request, 0) >> 16;
        descriptors != 0 && (request_type == 4 || request_type == 6) && !(read(request, 4) >> 31)) {
        const std::size_t raw = (8 + 8 * (descriptors & 15) +
            12 * (((descriptors >> 4) & 15) + ((descriptors >> 8) & 15) + ((descriptors >> 12) & 15)) + 15) & ~std::size_t{15};
        if (raw >= request.size()) return {};
        moved = request;
        std::copy(request.begin() + static_cast<std::ptrdiff_t>(raw), request.end(), moved.begin() + 16);
        write(moved, 0, read(request, 0) & 0xffff);
        write(moved, 4, read(request, 4) & 0x3ff);  // raw words, without receive-list flags
        source = &moved;
    }
    const bool buffered = source != &request;
    // A request whose special header carries exactly one copied handle (no
    // process id, no moved handle) has the header word and the handle in the
    // 8 bytes before the raw data, which still starts at byte 16. It is read
    // from a copy without the handle-descriptor flag; the handle is kept aside.
    std::optional<std::uint32_t> copied_handle;
    if (!buffered && (request_type == 4 || request_type == 6) && (read(request, 4) >> 31) &&
        read(request, 8) == 2) {
        moved = request;
        write(moved, 4, read(request, 4) & 0x3ff);
        copied_handle = read(request, 12);
        source = &moved;
    }
    const auto& message = *source;
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    if (contract_ != AppletContract::switch1 && contract_ != AppletContract::switch2)
        return {};
    diagnostic_ = "Applet service request is not implemented";
    if (sender_pid != process_id_) {
        diagnostic_ = "Applet service session belongs to a different process";
        return {};
    }
    const auto type = read(message, 0);
    const auto words = read(message, 4);
    if ((type == 4 || type == 6) && domain_ && source == &request && words >= 6 && words < 12 &&
        (read(message, 16) & 0xff) == 2) {
        // A domain close releases one object; its reply has no CMIF payload.
        // Ids stay assigned, so the slot is only marked closed. The Switch 2
        // SDK sends the same CMIF framework message (type 2 and the object
        // id) when it releases a child, such as a launch-parameter storage.
        const auto id = read(message, 20);
        if (id < 2 || id - 2 >= objects_.size() || objects_[id - 2].kind == ObjectKind::closed ||
            objects_[id - 2].kind == ObjectKind::application_proxy)
            return {};
        objects_[id - 2] = {ObjectKind::closed, nullptr, nullptr};
        horizon::IpcReply reply{horizon::IpcDisposition::handled};
        reply.size = 32;
        write(reply.bytes, 4, 6);
        write(reply.bytes, 16, 0x4f434653);
        diagnostic_ = nullptr;
        return reply;
    }
    if ((type == 4 || type == 6) && domain_ && words >= 12 &&
        words <= (message.size() - 8) / 4) {
        const auto header = read(message, 16);
        const auto size = header >> 16;
        const auto id = read(message, 20);
        // Input object ids follow the raw data; only PushInData takes one.
        const auto in_objects = (header >> 8) & 0xff;
        if ((header & 0xff) != 1 || in_objects > 1 || size < 16 || size + 4 * in_objects > 8 + words * 4 - 32 ||
            id < 2 || id - 2 >= objects_.size() || read(message, 32) != 0x49434653 ||
            objects_.size() >= std::numeric_limits<std::uint32_t>::max() - 1)
            return {};
        // Switch 2 provisional handlers: zero replies adopted from practical
        // reference evidence ahead of firmware verification. Each succeeds with
        // zeroed output of the listed size and warns once per session. The
        // table below lists them all.
        struct Provisional {
            ObjectKind kind; std::uint32_t command; std::uint32_t output_bytes; std::uint32_t token; const char* name;
            // The first output word; the rest of the output is zero.
            std::uint32_t first_word = 0;
        };
        static constexpr Provisional switch2_provisional[] = {
            {ObjectKind::application_functions, 320, 2, 0x21f9610e, "ApplicationFunctions 320"},
            {ObjectKind::application_functions, 321, 1, 0x21f9610e, "ApplicationFunctions 321"},
            {ObjectKind::application_functions, 330, 1, 0x21f9610e, "ApplicationFunctions 330"},
            // GetCurrentGraphicsScale: scale 10, the first of the accepted 10, 20 and 30.
            {ObjectKind::graphics_scale, 10, 12, 0, "GraphicsScale 10", 10},
        };
        static_assert(std::size(switch2_provisional) <= 64);
        if (contract_ == AppletContract::switch2) {
            for (std::size_t index = 0; index != std::size(switch2_provisional); ++index) {
                const auto& entry = switch2_provisional[index];
                if (entry.kind != objects_[id - 2].kind || entry.command != read(message, 40)) continue;
                if (in_objects || copied_handle || buffered) return {};
                if (!(provisional_warned_ & (std::uint64_t{1} << index))) {
                    provisional_warned_ |= std::uint64_t{1} << index;
                    armigo::diag::write(armigo::diag::Level::warning,
                        "Armigo am: provisional Switch 2 handler %s answers with an unverified reply\n", entry.name);
                }
                const auto output_words = (entry.output_bytes + 3) / 4;
                horizon::IpcReply reply{horizon::IpcDisposition::handled};
                reply.size = 56 + 4 * output_words;
                write(reply.bytes, 4, 12 + output_words);
                write(reply.bytes, 32, 0x4f434653);
                write(reply.bytes, 44, entry.token);
                if (output_words) write(reply.bytes, 48, entry.first_word);
                diagnostic_ = nullptr;
                return reply;
            }
        }
        // Switch 2 child commands admitted so far; each shares the handler below
        // with the Switch 1 platform contract.
        struct Admitted { ObjectKind kind; std::uint32_t command; };
        static constexpr Admitted switch2_admitted[] = {
            {ObjectKind::window_controller, 1}, {ObjectKind::window_controller, 10},
            {ObjectKind::common_state, 0}, {ObjectKind::common_state, 1}, {ObjectKind::common_state, 5},
            {ObjectKind::common_state, 6}, {ObjectKind::common_state, 9}, {ObjectKind::common_state, 50},
            {ObjectKind::common_state, 66}, {ObjectKind::common_state, 67}, {ObjectKind::common_state, 900},
            {ObjectKind::self_controller, 1}, {ObjectKind::self_controller, 2}, {ObjectKind::self_controller, 9},
            {ObjectKind::self_controller, 10}, {ObjectKind::self_controller, 11}, {ObjectKind::self_controller, 12},
            {ObjectKind::self_controller, 13}, {ObjectKind::self_controller, 14}, {ObjectKind::self_controller, 16},
            {ObjectKind::self_controller, 62}, {ObjectKind::self_controller, 63}, {ObjectKind::self_controller, 91},
            {ObjectKind::self_controller, 2000}, {ObjectKind::self_controller, 2400},
            {ObjectKind::application_functions, 1}, {ObjectKind::application_functions, 20},
            {ObjectKind::application_functions, 21}, {ObjectKind::application_functions, 22},
            {ObjectKind::application_functions, 23},
            {ObjectKind::application_functions, 40}, {ObjectKind::application_functions, 67},
            {ObjectKind::application_functions, 130}, {ObjectKind::application_functions, 210},
            {ObjectKind::storage, 0}, {ObjectKind::storage_accessor, 0}, {ObjectKind::storage_accessor, 10},
            {ObjectKind::storage_accessor, 11},
            {ObjectKind::graphics_scale, 1050},
        };
        if (contract_ == AppletContract::switch2 &&
            objects_[id - 2].kind != ObjectKind::application_proxy &&
            std::none_of(std::begin(switch2_admitted), std::end(switch2_admitted), [&](const Admitted& entry) {
                return entry.kind == objects_[id - 2].kind && entry.command == read(message, 40);
            })) {
            diagnostic_ = "Switch 2 applet child command requires a verified contract";
            return {};
        }
        if (in_objects && (objects_[id - 2].kind != ObjectKind::library_applet || read(message, 40) != 100))
            return {};
        // Only a storage read or write takes a buffer, and only a transfer
        // storage takes a copied handle.
        if (copied_handle && (objects_[id - 2].kind != ObjectKind::library_creator || read(message, 40) != 11))
            return {};
        if (buffered &&
            (objects_[id - 2].kind != ObjectKind::storage_accessor ||
             (read(message, 40) != 10 && read(message, 40) != 11)))
            return {};
        if (contract_ == AppletContract::switch2 &&
            objects_[id - 2].kind == ObjectKind::self_controller && read(message, 40) == 2000) {
            // GetGraphicsSystemSession: one copied handle to a new graphics
            // system session, which the SDK then speaks TIPC on.
            std::shared_ptr<gsv::GraphicsSystemSession> session;
            try {
                session = std::make_shared<gsv::GraphicsSystemSession>(process_id_, *handles_,
                                                                     graphics_system_dependencies_);
            } catch (const std::bad_alloc&) {
                return {};
            }
            const auto handle = handles_->insert(session);
            if (!handle) return {};
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 64;
            write(reply.bytes, 4, 0x8000000c);
            write(reply.bytes, 8, 2);
            write(reply.bytes, 12, *handle);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x56efeaa1);
            diagnostic_ = nullptr;
            return reply;
        }
        if (contract_ == AppletContract::switch2 &&
            objects_[id - 2].kind == ObjectKind::self_controller && read(message, 40) == 2400) {
            // The Switch 2 AM program takes one boolean byte, keeps it in the
            // controller and the applet's display state, and replies without
            // output. The display update it then triggers is not modeled.
            if (words < 13 || size < 17) return {};
            self_controller_2400_ = (read(message, 48) & 1) != 0;
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x56efeaa1);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::graphics_scale && read(message, 40) == 1050) {
            // SetGraphicsScaleChangedNotificationEnabled takes one boolean byte. The
            // graphics scale never changes on this host, so the flag is only kept.
            if (words < 13 || size < 17) return {};
            graphics_scale_notifications_ = (read(message, 48) & 1) != 0;
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::window_controller && read(message, 40) == 1) {
            // The host application instance uses its process identity for the
            // lifetime of the launch, independently of handles and title IDs.
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 64;
            write(reply.bytes, 4, 14);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x0279422d);
            write(reply.bytes, 48, static_cast<std::uint32_t>(process_id_));
            write(reply.bytes, 52, static_cast<std::uint32_t>(process_id_ >> 32));
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::window_controller && read(message, 40) == 10) {
            // AcquireForegroundRights returns success without changing focus
            // or any other session state. Foreground selection remains host
            // launch policy on both platforms.
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x0279422d);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::application_functions && read(message, 40) == 67) {
            if (words < 13 || size < 20) return {};
            // Firmware validates the low byte before checking recording capability.
            const auto state = read(message, 48) & 0xff;
            const auto result = state <= 1 ? 0u : 0x3f480u;
            if (result == 0 && recording_policy_ != GameplayRecordingPolicy::Unavailable) {
                diagnostic_ = "Gameplay recording launch policy is not implemented";
                return {};
            }
            // Unavailable recording accepts valid requests without changing state
            // or creating capture resources, matching the capability-zero branch.
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 40, result);
            write(reply.bytes, 44, 0x21f9610e);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::common_state && read(message, 40) == 66) {
            // Command66 takes one inline mode and produces no output. Both
            // 23.0.1 AM programs read its low byte and answer a mode above the
            // platform's limit with 0x3f480 without changing state; Switch 2
            // adds the GPU-only mode 2 (S1 0x80360-0x80390, S2 0xccf20-0xccf54).
            if (words < 13 || size < 20) return {};
            const auto limit = contract_ == AppletContract::switch2 ? 2u : 1u;
            const auto value = read(message, 48) & 0xff;
            std::uint32_t result = 0;
            if (value > limit) {
                result = 0x3f480;
            } else {
                common_state_command66_ = value;
                common_state_boost_active_ = true;
            }
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 40, result);
            write(reply.bytes, 44, 0x32f27d77);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::common_state && read(message, 40) == 67) {
            // The cancellation clears the logical active state while retaining
            // the last mode selected by command66 for later inspection.
            common_state_boost_active_ = false;
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x32f27d77);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::common_state &&
            (read(message, 40) == 5 || read(message, 40) == 6 || read(message, 40) == 50)) {
            // Launch mode is independent of foreground ownership and CPU boost.
            // These queries neither consume input nor change notifications.
            // VR remains inactive; byte outputs are zero-padded.
            const auto command = read(message, 40);
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 60;
            write(reply.bytes, 4, 13);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x32f27d77);
            write(reply.bytes, 48, command == 5 ? static_cast<std::uint32_t>(operation_mode_) :
                  command == 6 ? operation_mode_profile(operation_mode_).performance_mode : 0);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::common_state && read(message, 40) == 60) {
            const auto profile = operation_mode_profile(operation_mode_);
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 64;
            write(reply.bytes, 4, 14);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x32f27d77);
            write(reply.bytes, 48, profile.display_width);
            write(reply.bytes, 52, profile.display_height);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::common_state && read(message, 40) == 54) {
            // EndVrModeEx: the runtime never enters VR mode, so ending it is a
            // successful no-op with no output payload.
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x32f27d77);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::common_state && read(message, 40) == 900) {
            exit_to_library_on_next_program_ = true;
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x32f27d77);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::common_state &&
            (read(message, 40) == 1 || read(message, 40) == 9)) {
            const bool receive = read(message, 40) == 1;
            const bool empty = receive && (!focus_notification_pending_ || !focus_notifications_enabled_);
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = empty ? 56 : 60;
            write(reply.bytes, 4, empty ? 12 : 13);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 40, empty ? 0x680 : 0);
            write(reply.bytes, 44, 0x32f27d77);
            if (!empty) write(reply.bytes, 48, receive ? 15 : focused_ ? 1 : 2);
            if (receive && !empty) {
                focus_notification_pending_ = false;
                if (common_state_event_) common_state_event_->reset();
            }
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::self_controller &&
            (read(message, 40) == 13 || read(message, 40) == 16)) {
            const bool handling = read(message, 40) == 13;
            if (words < 13 || size < (handling ? 19u : 17u)) return {};
            const auto value = read(message, 48);
            if (handling) {
                focus_notifications_enabled_ = (value & 1) != 0;
                background_mode_enabled_ = (value & 0x100) != 0;
                focus_suspending_enabled_ = (value & 0x10000) != 0;
                if (common_state_event_) {
                    if (focus_notifications_enabled_ && focus_notification_pending_) common_state_event_->signal();
                    else common_state_event_->reset();
                }
            } else out_of_focus_suspending_enabled_ = (value & 1) != 0;
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x56efeaa1);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::self_controller && read(message, 40) == 44) {
            // CreateManagedDisplaySeparableLayer answers with the display and the
            // recording layer id. The application keys its managed layer on the
            // display id, so a zero id leaves it without a layer.
            const auto display_layer = ++next_layer_id_;
            const auto recording_layer = ++next_layer_id_;
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 64;
            write(reply.bytes, 4, 14);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x56efeaa1);
            write(reply.bytes, 48, static_cast<std::uint32_t>(display_layer));
            write(reply.bytes, 52, static_cast<std::uint32_t>(display_layer >> 32));
            write(reply.bytes, 56, static_cast<std::uint32_t>(recording_layer));
            write(reply.bytes, 60, static_cast<std::uint32_t>(recording_layer >> 32));
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::self_controller) {
            const auto command = read(message, 40);
            if (command == 9) {
                // GetLibraryAppletLaunchableEvent: a copied readable event that is
                // signaled while the application may launch a library applet, which
                // the foreground application always may here.
                try {
                    if (!library_applet_launchable_event_)
                        library_applet_launchable_event_ = std::make_shared<horizon::Event>(false);
                } catch (const std::bad_alloc&) {
                    return {};
                }
                if (focused_) library_applet_launchable_event_->signal();
                const auto handle = handles_->insert(library_applet_launchable_event_);
                if (!handle) return {};
                horizon::IpcReply reply{horizon::IpcDisposition::handled};
                reply.size = 64;
                write(reply.bytes, 4, 0x8000000c);
                write(reply.bytes, 8, 2);
                write(reply.bytes, 12, *handle);
                write(reply.bytes, 32, 0x4f434653);
                write(reply.bytes, 44, 0x56efeaa1);
                diagnostic_ = nullptr;
                return reply;
            }
            if (command == 10) {
                // SetScreenShotPermission: one u32, 0 inherits the policy, 1
                // allows and 2 denies captures. The setting is kept; this
                // runtime takes no captures for the guest.
                if (words < 13 || size < 20) return {};
                const auto permission = read(message, 48);
                if (permission > 2) return {};
                screenshot_permission_ = permission;
                horizon::IpcReply reply{horizon::IpcDisposition::handled};
                reply.size = 56;
                write(reply.bytes, 4, 12);
                write(reply.bytes, 32, 0x4f434653);
                write(reply.bytes, 44, 0x56efeaa1);
                diagnostic_ = nullptr;
                return reply;
            }
            if (command == 1 || command == 2) {
                // LockExit / UnlockExit: no input, no output.
                if (words < 12 || size < 16) return {};
                exit_locked_ = command == 1;
                horizon::IpcReply reply{horizon::IpcDisposition::handled};
                reply.size = 56;
                write(reply.bytes, 4, 12);
                write(reply.bytes, 32, 0x4f434653);
                write(reply.bytes, 44, 0x56efeaa1);
                diagnostic_ = nullptr;
                return reply;
            }
            if (command == 62 || command == 63) {
                const bool setter = command == 62;
                std::uint32_t result = 0;
                if (setter) {
                    if (words < 13 || size < 20) return {};
                    const auto value = read(message, 48);
                    // Validate the full u32 before changing the retained preference.
                    if (value > 2) result = 0x3f480;
                    else idle_time_detection_extension_ = value;
                }
                // The headless launch retains the idle policy preference; it has
                // no host idle-detection worker or display power transitions.
                horizon::IpcReply reply{horizon::IpcDisposition::handled};
                reply.size = setter ? 56 : 60;
                write(reply.bytes, 4, setter ? 12 : 13);
                write(reply.bytes, 32, 0x4f434653);
                write(reply.bytes, 40, result);
                write(reply.bytes, 44, 0x56efeaa1);
                if (!setter) write(reply.bytes, 48, idle_time_detection_extension_);
                diagnostic_ = nullptr;
                return reply;
            }
            // Commands 11, 12 and 14 each carry one boolean byte and produce no
            // output. They only configure which notifications the application
            // wants; no notification source exists yet in this runtime.
            static constexpr std::uint32_t boolean_setters[] = {11, 12, 14};
            for (std::size_t index = 0; index != std::size(boolean_setters); ++index) {
                if (command != boolean_setters[index]) continue;
                if (words < 13 || size < 17) return {};
                self_controller_flags_[index] = (read(message, 48) & 1) != 0;
                horizon::IpcReply reply{horizon::IpcDisposition::handled};
                reply.size = 56;
                write(reply.bytes, 4, 12);
                write(reply.bytes, 32, 0x4f434653);
                write(reply.bytes, 44, 0x56efeaa1);
                diagnostic_ = nullptr;
                return reply;
            }
        }
        const bool suspended_tick = objects_[id - 2].kind == ObjectKind::self_controller &&
                                    read(message, 40) == 91;
        const bool gpu_error = objects_[id - 2].kind == ObjectKind::application_functions &&
                               read(message, 40) == 130;
        const bool common_state = objects_[id - 2].kind == ObjectKind::common_state &&
                                  read(message, 40) == 0;
        const bool display_resolution = objects_[id - 2].kind == ObjectKind::common_state &&
                                        read(message, 40) == 61;
        if (suspended_tick || gpu_error || common_state || display_resolution) {
            auto& event = suspended_tick ? suspended_tick_event_ :
                          gpu_error ? gpu_error_event_ :
                          display_resolution ? display_resolution_event_ : common_state_event_;
            try {
                if (!event) event = std::make_shared<horizon::Event>(
                    common_state && focus_notification_pending_ && focus_notifications_enabled_);
            } catch (const std::bad_alloc&) {
                return {};
            }
            const auto handle = handles_->insert(event);
            if (!handle) return {};
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 64;
            write(reply.bytes, 4, 0x8000000c);
            write(reply.bytes, 8, 2);
            write(reply.bytes, 12, *handle);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, suspended_tick ? 0x56efeaa1 :
                                   gpu_error ? 0x21f9610e : 0x32f27d77);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::application_functions && read(message, 40) == 210) {
            // The application takes a readable event handle and attaches it to a
            // system event it can wait on, so the reply copies one handle.
            try {
                if (!application_state_event_)
                    application_state_event_ = std::make_shared<horizon::Event>(false);
            } catch (const std::bad_alloc&) {
                return {};
            }
            const auto handle = handles_->insert(application_state_event_);
            if (!handle) return {};
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 64;
            write(reply.bytes, 4, 0x8000000c);
            write(reply.bytes, 8, 2);
            write(reply.bytes, 12, *handle);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0xc0ffee01);
            diagnostic_ = nullptr;
            return reply;
        }
        // A storage object's bytes are read through an accessor it opens.
        const auto output_object = [&](ObjectKind kind, Bytes data, std::uint32_t marker,
                                       std::shared_ptr<LibraryApplet> applet = nullptr) -> horizon::IpcReply {
            try {
                objects_.push_back({kind, nullptr, std::move(data), std::move(applet), nullptr, 0});
            } catch (const std::bad_alloc&) {
                return {};
            }
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 60;
            write(reply.bytes, 4, 13);
            write(reply.bytes, 16, 1);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, marker);
            write(reply.bytes, 48, static_cast<std::uint32_t>(objects_.size() + 1));
            diagnostic_ = nullptr;
            return reply;
        };
        // Storage and StorageAccessor replies carry their own interface words.
        constexpr std::uint32_t storage_interface = 0x8e211f3d;
        constexpr std::uint32_t accessor_interface = 0x919f469a;
        if (objects_[id - 2].kind == ObjectKind::storage && read(message, 40) == 0) {
            // IStorage command0 Open: an accessor over the same bytes.
            return output_object(ObjectKind::storage_accessor, objects_[id - 2].data, storage_interface);
        }
        if (objects_[id - 2].kind == ObjectKind::storage_accessor && read(message, 40) == 0) {
            // IStorageAccessor command0 GetSize: the size as a signed 64-bit value.
            const auto bytes = objects_[id - 2].data->size();
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, accessor_interface);
            write(reply.bytes, 48, static_cast<std::uint32_t>(bytes));
            write(reply.bytes, 52, static_cast<std::uint32_t>(std::uint64_t{bytes} >> 32));
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::storage_accessor && read(message, 40) == 11) {
            // IStorageAccessor command11 Read(offset): fills the caller's output
            // buffer from the offset. A range past the end is not handed out.
            if (words < 14 || size < 24) return {};
            const auto& data = *objects_[id - 2].data;
            const auto offset = std::uint64_t{read(message, 48)} | std::uint64_t{read(message, 52)} << 32;
            if (!buffers || buffers->size() != 1 || !(*buffers)[0].output) {
                diagnostic_ = "Storage read needs one output buffer";
                return {};
            }
            const auto length = (*buffers)[0].size;
            if (offset > data.size() || length > data.size() - offset) {
                diagnostic_ = "Storage read is past the end of the storage";
                return {};
            }
            if (!buffers->write(0, std::span(data).subspan(offset, length))) {
                diagnostic_ = "Storage read buffer is not writable";
                return {};
            }
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 48;
            write(reply.bytes, 4, 10);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, accessor_interface);
            diagnostic_ = nullptr;
            return reply;
        }
        // A domain reply with `result` in its CMIF header and `output` words after it.
        const auto plain_reply = [&](std::uint32_t result, std::initializer_list<std::uint32_t> output = {}) {
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 48 + 4 * output.size();
            write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 36, result ? 1 : 0);
            write(reply.bytes, 40, result);
            write(reply.bytes, 44, objects_[id - 2].kind == ObjectKind::storage_accessor
                                       ? accessor_interface : 0x21f9610e);
            unsigned at = 48;
            for (const auto value : output) {
                write(reply.bytes, at, value);
                at += 4;
            }
            diagnostic_ = nullptr;
            return reply;
        };
        const auto read_s64 = [&](unsigned offset) {
            return static_cast<std::int64_t>(std::uint64_t{read(message, offset)} |
                                             std::uint64_t{read(message, offset + 4)} << 32);
        };
        if (objects_[id - 2].kind == ObjectKind::storage_accessor && read(message, 40) == 10) {
            // IStorageAccessor command10 Write(offset): copies the caller's input
            // buffer into the storage at the offset; a range past the end is refused.
            if (words < 14 || size < 24) return {};
            auto& data = *objects_[id - 2].data;
            const auto offset = static_cast<std::uint64_t>(read_s64(48));
            if (!buffers || buffers->size() != 1 || !(*buffers)[0].input) {
                diagnostic_ = "Storage write needs one input buffer";
                return {};
            }
            const auto length = (*buffers)[0].size;
            if (offset > data.size() || length > data.size() - offset) {
                diagnostic_ = "Storage write is past the end of the storage";
                return {};
            }
            if (!buffers->read(0, std::span(data).subspan(offset, length))) {
                diagnostic_ = "Storage write buffer is not readable";
                return {};
            }
            return plain_reply(0);
        }
        if (objects_[id - 2].kind == ObjectKind::library_creator && read(message, 40) == 10) {
            // ILibraryAppletCreator command10 CreateStorage(s64 size): a zero-filled storage.
            constexpr std::int64_t storage_limit = 16 << 20;
            if (words < 14 || size < 24) return {};
            const auto bytes = read_s64(48);
            if (bytes <= 0 || bytes > storage_limit) {
                diagnostic_ = "Library applet storage size is not supported";
                return {};
            }
            Bytes data;
            try {
                data = std::make_shared<std::vector<std::byte>>(static_cast<std::size_t>(bytes));
            } catch (const std::bad_alloc&) {
                return {};
            }
            return output_object(ObjectKind::storage, std::move(data), 0x21f9610e);
        }
        if (objects_[id - 2].kind == ObjectKind::library_creator && read(message, 40) == 11) {
            // ILibraryAppletCreator command11 CreateTransferMemoryStorage(handle, u8
            // writable, s64 size): a large storage over the application memory the
            // transfer memory lends, which the applet reads in place.
            if (!copied_handle || words < 16 || size < 32) return {};
            const auto memory = std::dynamic_pointer_cast<horizon::TransferMemoryRecord>(
                handles_->lookup(*copied_handle));
            if (!memory) {
                diagnostic_ = "Transfer storage needs a transfer memory handle";
                return {};
            }
            const auto bytes = read_s64(56);
            if (bytes <= 0 || static_cast<std::uint64_t>(bytes) > memory->range().size) {
                diagnostic_ = "Transfer storage is larger than its transfer memory";
                return {};
            }
            auto reply = output_object(ObjectKind::transfer_storage, nullptr, 0x21f9610e);
            if (reply.disposition == horizon::IpcDisposition::handled) {
                objects_.back().memory = memory;
                objects_.back().memory_size = static_cast<std::uint64_t>(bytes);
            }
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::library_creator && read(message, 40) == 0) {
            // ILibraryAppletCreator command0 CreateLibraryApplet(u32 applet id, u32 mode).
            // Stand-ins exist for the network-connection applet (15) and the
            // software keyboard (17), both in mode 0 (all foreground).
            if (words < 14 || size < 24) return {};
            const auto applet_id = read(message, 48);
            const auto mode = read(message, 52);
            if ((applet_id != 15 && applet_id != 17) || mode != 0) {
                diagnostic_ = applet_id == 17
                    ? "Only the full-screen software keyboard is implemented"
                    : "Only the network-connection and software keyboard library applets are implemented";
                return {};
            }
            std::shared_ptr<LibraryApplet> applet;
            try {
                applet = std::make_shared<LibraryApplet>();
                applet->state_changed = std::make_shared<horizon::Event>(false);
            } catch (const std::bad_alloc&) {
                return {};
            }
            applet->id = applet_id;
            applet->mode = mode;
            return output_object(ObjectKind::library_applet, nullptr, 0x21f9610e, std::move(applet));
        }
        if (objects_[id - 2].kind == ObjectKind::library_applet) {
            auto& applet = *objects_[id - 2].applet;
            // A copied readable event.
            const auto copy_event = [&](const std::shared_ptr<horizon::Event>& event) -> horizon::IpcReply {
                const auto handle = handles_->insert(event);
                if (!handle) return {};
                horizon::IpcReply reply{horizon::IpcDisposition::handled};
                reply.size = 64;
                write(reply.bytes, 4, 0x8000000c);
                write(reply.bytes, 8, 2);
                write(reply.bytes, 12, *handle);
                write(reply.bytes, 32, 0x4f434653);
                write(reply.bytes, 44, 0x21f9610e);
                diagnostic_ = nullptr;
                return reply;
            };
            switch (read(message, 40)) {
            case 0:  // GetAppletStateChangedEvent
                return copy_event(applet.state_changed);
            case 1:  // IsCompleted
                return plain_reply(0, {applet.started ? 1u : 0u});
            case 10: {
                // Start: the stand-in finishes at once. The network-connection
                // applet behaves as when the user dismisses it, leaving one output
                // storage (u32 1: not connected); the software keyboard leaves the
                // close result and text the host input decided.
                if (applet.started) {
                    diagnostic_ = "Library applet was already started";
                    return {};
                }
                if (applet.id == 17) {
                    if (const auto failure = run_keyboard(applet, buffers)) {
                        diagnostic_ = failure;
                        return {};
                    }
                } else {
                    Bytes output;
                    try {
                        output = std::make_shared<std::vector<std::byte>>(4);
                        applet.out.push_back(output);
                    } catch (const std::bad_alloc&) {
                        return {};
                    }
                    (*output)[0] = std::byte{1};
                }
                applet.started = true;
                applet.state_changed->signal();
                return plain_reply(0);
            }
            case 30:  // GetResult: the applet exited normally
                if (!applet.started) {
                    diagnostic_ = "Library applet result before it was started";
                    return {};
                }
                return plain_reply(0);
            case 100: {  // PushInData(IStorage): the input storage is kept
                if (!in_objects) return {};
                const auto in = read(message, 32 + size);
                if (in < 2 || in - 2 >= objects_.size() ||
                    (objects_[in - 2].kind != ObjectKind::storage &&
                     objects_[in - 2].kind != ObjectKind::transfer_storage)) {
                    diagnostic_ = "Library applet input is not a storage";
                    return {};
                }
                const auto& storage = objects_[in - 2];
                try {
                    applet.in.push_back({storage.data, storage.memory,
                                         storage.data ? storage.data->size() : storage.memory_size});
                } catch (const std::bad_alloc&) {
                    return {};
                }
                return plain_reply(0);
            }
            case 101: {  // PopOutData: the next output storage, or no data in the channel
                if (applet.out.empty()) return plain_reply(0x480);
                auto output = applet.out.front();
                auto reply = output_object(ObjectKind::storage, output, 0x21f9610e);
                if (reply.disposition == horizon::IpcDisposition::handled) applet.out.pop_front();
                return reply;
            }
            case 106:  // GetPopInteractiveOutDataEvent
                if (!applet.interactive_out) {
                    try {
                        applet.interactive_out = std::make_shared<horizon::Event>(false);
                    } catch (const std::bad_alloc&) {
                        return {};
                    }
                }
                return copy_event(applet.interactive_out);
            default:
                diagnostic_ = "Library applet accessor method is not implemented";
                return {};
            }
        }
        if (objects_[id - 2].kind == ObjectKind::application_functions && read(message, 40) == 1) {
            // IApplicationFunctions command1 PopLaunchParameter(kind): hands out
            // the next queued parameter of that kind as a storage object. Only the
            // launch policy queues one (the preselected user); otherwise the
            // channel is empty. The empty-channel result is a CMIF result, so it
            // travels in the reply header where the client's CMIF parser reads it.
            if (words >= 13 && size >= 20) {
                const auto kind = read(message, 48);
                if (kind >= 1 && kind <= 3 && launch_parameters_[kind - 1]) {
                    auto reply = output_object(ObjectKind::storage, launch_parameters_[kind - 1], 0x21f9610e);
                    if (reply.disposition == horizon::IpcDisposition::handled)
                        launch_parameters_[kind - 1].reset();
                    return reply;
                }
            }
            constexpr std::uint32_t no_data_in_channel = 0x480;
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 48;
            write(reply.bytes, 4, 10);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 36, 1);
            write(reply.bytes, 40, no_data_in_channel);
            write(reply.bytes, 44, 0x21f9610e);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::application_functions && read(message, 40) == 20) {
            // IApplicationFunctions command20 EnsureSaveData(uid) makes the
            // title's save data exist and returns the extra space that needs, as
            // a signed 64-bit byte count. A zero UID skips only the user save
            // (23.0.1). The application has one save tree, which fsp-srv opens
            // for every user and creates on first use, so nothing has to be made
            // here and no space is required.
            if (words < 16 || size < 32) return {};
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x21f9610e);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::application_functions && read(message, 40) == 21) {
            // Command21 applies the title mask through NS's priority table,
            // then converts the selected NACP index to its canonical code.
            const auto language = set::desired_console_language();
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            if (!language) {
                reply.size = 48;
                write(reply.bytes, 4, 10);
                write(reply.bytes, 32, 0x4f434653);
                write(reply.bytes, 40, 0x25810);
                write(reply.bytes, 44, 0x21f9610e);
                diagnostic_ = "Desired application language is unavailable";
                return reply;
            }
            reply.size = 56;
            write(reply.bytes, 4, 12);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x21f9610e);
            write(reply.bytes, 48, static_cast<std::uint32_t>(*language));
            write(reply.bytes, 52, static_cast<std::uint32_t>(*language >> 32));
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::application_functions && read(message, 40) == 22) {
            // IApplicationFunctions command22 records the caller's termination
            // result. Firmware keeps it for the eventual exit path; there is no
            // output payload, so the reply is the plain domain success header.
            if (words < 13 || size < 20) return {};
            terminate_result_ = read(message, 48);
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 48;
            write(reply.bytes, 4, 10);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x21f9610e);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::application_functions && read(message, 40) == 23) {
            // GetDisplayVersion: no input, the 16-byte NUL-terminated display
            // version. The launch's package has no control data, so the
            // version is the one the launch derived from the package version;
            // it stays provisional and warns once.
            if (!display_version_) {
                diagnostic_ = "Display version needs the application's control data";
                return {};
            }
            if (!display_version_warned_) {
                display_version_warned_ = true;
                armigo::diag::write(armigo::diag::Level::warning,
                    "Armigo am: provisional Switch 2 handler ApplicationFunctions 23 reports display version %s "
                    "derived from the package version; the package has no control data\n",
                    display_version_->c_str());
            }
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 64;
            write(reply.bytes, 4, 14);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x21f9610e);
            std::memcpy(reply.bytes.data() + 48, display_version_->data(), std::min<std::size_t>(display_version_->size(), 15));
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind == ObjectKind::application_functions && read(message, 40) == 40) {
            // ApplicationFunctions command40 NotifyRunning: the application
            // reports that application startup is complete. The request carries no
            // inline arguments, so stale payload bytes beyond the declared CMIF
            // extent are ignored. The reply is the enabled state on every call;
            // when delivery is ready the state also performs its once-only
            // observer delivery. No asynchronous work or fence is involved.
            bool running = false;
            try {
                running = running_state_->notify();
            } catch (...) {
                // The observer failed after the sent mark was taken: the
                // delivery is not retried and the request fails explicitly
                // instead of reporting success.
                diagnostic_ = "Application running notification observer failed";
                return {};
            }
            horizon::IpcReply reply{horizon::IpcDisposition::handled};
            reply.size = 60;
            write(reply.bytes, 4, 13);
            write(reply.bytes, 32, 0x4f434653);
            write(reply.bytes, 44, 0x21f9610e);
            write(reply.bytes, 48, running ? 1u : 0u);
            diagnostic_ = nullptr;
            return reply;
        }
        if (objects_[id - 2].kind != ObjectKind::application_proxy) return {};
        ObjectKind child_kind;
        switch (read(message, 40)) {
        case 0: child_kind = ObjectKind::common_state; break;
        case 1: child_kind = ObjectKind::self_controller; break;
        case 2: child_kind = ObjectKind::window_controller; break;
        case 3: child_kind = ObjectKind::audio_controller; break;
        case 4: child_kind = ObjectKind::display_controller; break;
        case 11: child_kind = ObjectKind::library_creator; break;
        case 20: child_kind = ObjectKind::application_functions; break;
        case 40:
            if (contract_ != AppletContract::switch2) return {};
            child_kind = ObjectKind::application_status;
            break;
        case 60:
            if (contract_ != AppletContract::switch2) return {};
            child_kind = ObjectKind::graphics_scale;
            break;
        case 1000: child_kind = ObjectKind::debug_functions; break;
        default: return {};
        }
        try {
            objects_.push_back({child_kind, objects_[id - 2].copied_object});
        } catch (const std::bad_alloc&) {
            return {};
        } catch (const std::length_error&) {
            return {};
        }
        horizon::IpcReply reply{horizon::IpcDisposition::handled};
        reply.size = 60;
        write(reply.bytes, 4, 13);
        write(reply.bytes, 16, 1);
        write(reply.bytes, 32, 0x4f434653);
        write(reply.bytes, 44, 0x7d59c154);
        write(reply.bytes, 48, static_cast<std::uint32_t>(objects_.size() + 1));
        diagnostic_ = nullptr;
        return reply;
    }
    if (type == 4 || type == 6) {
        const auto command = read(message, 56);
        const bool extended_proxy = contract_ == AppletContract::switch2 && command == 10;
        const auto raw_words = words & 0x3ffu;
        if (!domain_ || !process_ || (words & ~0x3ffu) != 0x80000000u ||
            raw_words < (extended_proxy ? 16u : 14u) || raw_words > (message.size() - 24) / 4 || read(message, 8) != 3)
            return {};
        const auto domain_header = read(message, 32);
        const auto data_size = domain_header >> 16;
        if ((domain_header & 0xffff) != 1 || data_size < (extended_proxy ? 32u : 24u) ||
            data_size > 24 + raw_words * 4 - 48 || read(message, 36) != 1 ||
            read(message, 48) != 0x49434653 || (command != 0 && !extended_proxy))
            return {};
        if (extended_proxy && read(message, 72) != 0) {
            diagnostic_ = "Switch 2 application proxy configuration is not implemented";
            return {};
        }
        const auto copied_handle = read(message, 20);
        auto object = copied_handle == horizon::current_process_handle
            ? std::static_pointer_cast<horizon::KernelObject>(process_)
            : handles_->lookup(copied_handle);
        if (!object || objects_.size() >= std::numeric_limits<std::uint32_t>::max() - 1)
            return {};
        try {
            // Retain a copy in the registered application's proxy state. The
            // caller's handle remains owned by the caller, including on error.
            objects_.push_back({ObjectKind::application_proxy, std::move(object)});
        } catch (const std::bad_alloc&) {
            return {};
        } catch (const std::length_error&) {
            return {};
        }
        if (extended_proxy) {
            // Practical reference evidence admits the registered host application
            // through the same factory. Switch 2 launch-policy effects remain
            // unverified against console; retain the extra configuration word.
            application_proxy_configuration_ = read(message, 72);
            armigo::diag::opt_in("Switch 2 application proxy configuration=%u\n",
                *application_proxy_configuration_);
        }
        horizon::IpcReply reply{horizon::IpcDisposition::handled};
        reply.size = 60;
        write(reply.bytes, 4, 13);
        write(reply.bytes, 16, 1);
        write(reply.bytes, 32, 0x4f434653);
        write(reply.bytes, 44, 0xbb83bce3);
        write(reply.bytes, 48, static_cast<std::uint32_t>(objects_.size() + 1));
        diagnostic_ = nullptr;
        return reply;
    }
    if ((type != 5 && type != 7) || words < 8 || words > (message.size() - 8) / 4 ||
        read(message, 16) != 0x49434653)
        return {};
    const auto command = read(message, 24);
    if (command != 3 && (command != 0 || domain_)) return {};
    if (command == 0) domain_ = true;
    diagnostic_ = nullptr;
    return control_reply(command == 0 ? 1 : 0, 4);
}

const char* AppletServiceSession::run_keyboard(LibraryApplet& applet,
                                               const horizon::GuestBuffers* buffers) {
    // The input channel holds the common arguments, the configuration and the
    // work buffer, in that order.
    if (applet.in.size() < 3 || !applet.in[0].data || !applet.in[1].data)
        return "Software keyboard needs its arguments, configuration and work buffer";
    KeyboardRequest request;
    if (const auto failure = parse_keyboard_request(*applet.in[0].data, *applet.in[1].data, request))
        return failure;
    if (request.text_check) return "Software keyboard text check is not implemented";
    if (request.initial_length) {
        // The initial text is UTF-16 in the work buffer, at most what the text
        // area holds.
        const auto& work = applet.in[2];
        const auto bytes = std::uint64_t{request.initial_length} * 2;
        if (request.initial_length > keyboard_string_size / 2 || request.initial_offset > work.size ||
            bytes > work.size - request.initial_offset)
            return "Software keyboard initial text is outside the work buffer";
        std::vector<std::byte> raw(static_cast<std::size_t>(bytes));
        if (work.data) {
            std::copy_n(work.data->begin() + request.initial_offset, raw.size(), raw.begin());
        } else if (!buffers || !work.memory ||
                   !buffers->read_memory(work.memory->range().address + request.initial_offset, raw)) {
            return "Software keyboard work buffer is not readable";
        }
        for (std::size_t index = 0; index + 1 < raw.size(); index += 2) {
            const auto unit = static_cast<char16_t>(std::to_integer<unsigned>(raw[index]) |
                                                    std::to_integer<unsigned>(raw[index + 1]) << 8);
            if (unit == 0) break;
            request.initial.push_back(unit);
        }
    }
    const auto response = keyboard_input_(request);
    try {
        applet.out.push_back(std::make_shared<std::vector<std::byte>>(keyboard_output(request, response)));
    } catch (const std::bad_alloc&) {
        return "Software keyboard output could not be allocated";
    }
    armigo::diag::write(armigo::diag::Level::info,  "Armigo am: software keyboard (initial %zu units, limits %u-%u, %s) returns %s with %zu units\n",
                 request.initial.size(), request.min_length, request.max_length,
                 request.utf8 ? "UTF-8" : "UTF-16", response.accepted ? "OK" : "cancel",
                 response.text.size());
    return nullptr;
}

const char* AppletServiceSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

std::uint32_t AppletServiceSession::screenshot_permission() const {
    const std::lock_guard lock(mutex_);
    return screenshot_permission_;
}

bool AppletServiceSession::exit_locked() const {
    const std::lock_guard lock(mutex_);
    return exit_locked_;
}

std::optional<bool> AppletServiceSession::self_controller_flag(std::uint32_t command) const {
    const std::lock_guard lock(mutex_);
    static constexpr std::uint32_t boolean_setters[] = {11, 12, 14};
    for (std::size_t index = 0; index != std::size(boolean_setters); ++index)
        if (command == boolean_setters[index]) return self_controller_flags_[index];
    return {};
}

std::optional<std::uint32_t> AppletServiceSession::common_state_command66() const {
    const std::lock_guard lock(mutex_);
    return common_state_command66_;
}

bool AppletServiceSession::common_state_boost_active() const {
    const std::lock_guard lock(mutex_);
    return common_state_boost_active_;
}

std::uint32_t AppletServiceSession::terminate_result() const {
    const std::lock_guard lock(mutex_);
    return terminate_result_;
}

void AppletServiceSession::set_display_version(std::string version) {
    const std::lock_guard lock(mutex_);
    display_version_ = std::move(version);
}

void AppletServiceSession::set_graphics_system_dependencies(gsv::GraphicsSystemDependencies dependencies) {
    const std::lock_guard lock(mutex_);
    graphics_system_dependencies_ = std::move(dependencies);
}

void AppletServiceSession::grant_foreground() {
    const std::lock_guard lock(mutex_);
    if (focused_) return;
    focused_ = true;
    focus_notification_pending_ = true;
    if (common_state_event_ && focus_notifications_enabled_) common_state_event_->signal();
}

std::optional<std::uint32_t> AppletServiceSession::application_proxy_configuration() const {
    const std::lock_guard lock(mutex_);
    return application_proxy_configuration_;
}

std::size_t AppletServiceSession::domain_object_count() const {
    const std::lock_guard lock(mutex_);
    return domain_ ? 1 + objects_.size() : 0;
}

bool AppletServiceSession::exit_to_library_on_next_program_enabled() const {
    const std::lock_guard lock(mutex_);
    return exit_to_library_on_next_program_;
}

} // namespace armigo::services::am
