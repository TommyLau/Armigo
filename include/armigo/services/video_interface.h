// ============================================================================
// Armigo Services - Initial Video Interface Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/handle_table.h>
#include <armigo/horizon/ipc.h>
#include <armigo/services/graphics_buffer_backing.h>
#include <armigo/services/graphics_presentation.h>
#include <armigo/services/operation_mode.h>
#include <armigo/services/vi_producer.h>
#include <armigo/services/vsync_source.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace armigo::services::vi {

class NativeWindowProducerImpl;

// One display the application opened. The service keeps the name so later layer and
// buffer commands can be attributed to the display they belong to.
class DisplayRecord final : public horizon::KernelObject {
public:
    explicit DisplayRecord(std::string name) : name_(std::move(name)) {}
    const std::string& name() const noexcept { return name_; }

private:
    std::string name_;
};

// The display service the application opens first. Opening a display hands the
// caller a handle to a display object, which is how the client materialises its
// display; the objects themselves stay owned by this session's process.
class DisplaySession final : public horizon::ClientSession {
public:
    explicit DisplaySession(horizon::HandleTable& handles,
                            OperationMode operation_mode = OperationMode::Console);
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    std::size_t domain_object_count() const;
    const char* diagnostic() const;
    // Names of the displays this session opened, in order.
    std::vector<std::string> displays() const;
    // Overrides the process-wide refresh source. Tests use a short interval; a null
    // source restores the shared one.
    void set_vsync_source(std::shared_ptr<VsyncSource> source);
    // Binds the application's exported graphics-buffer namespace. The callback
    // and returned owners are shared by every transferred or cloned session.
    void set_buffer_backing_lookup(services::GraphicsBufferBackingLookup lookup);
    // Shared synchronous consumer. It must not re-enter VI; completion means
    // producer fences, GPU/readback work and host pacing have finished.
    void set_presenter(services::GraphicsPresenter presenter);
    // Associates an opaque host handle with this session's exact VI queue.
    // The handle remains registered until explicitly unbound.
    bool bind_native_window(void* native_window);

private:
    friend class NativeWindowProducerImpl;
    struct QueueState;
    horizon::IpcReply unsupported(const char* detail);
    horizon::IpcReply request_locked(const horizon::IpcMessage& message);
    mutable std::mutex mutex_;
    bool converted_ = false;
    const char* diagnostic_ = nullptr;
    horizon::HandleTable* handles_;
    const OperationMode operation_mode_;
    std::vector<std::shared_ptr<DisplayRecord>> displays_;
    // Handle and selector of the last session-level call.
    std::uint32_t session_handle_ = 0;
    std::uint32_t selector_ = 0;
    // Handle the client passed to the display session call.
    std::uint32_t service_handle_ = 0;
    std::vector<std::string> display_names_;
    // Display ids are handed out in order; the client keys its layer managers by
    // them, so a display without an id cannot own a layer.
    // Buffers of the request being served, or null outside a buffered request.
    const horizon::GuestBuffers* buffers_ = nullptr;
    std::uint32_t display_id_ = 0;
    std::uint32_t next_display_id_ = 0;
    // True for the session the service manager hands out; the display service it
    // transfers answers the display and binder commands.
    bool root_ = true;
    bool application_display_ = false;
    // Binder driver object ids handed out on this session.
    std::string binder_diagnostic_;
    std::shared_ptr<VsyncSource> vsync_;
    std::shared_ptr<QueueState> queue_;
};

}  // namespace armigo::services::vi
