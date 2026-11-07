// ============================================================================
// Armigo Services - Switch 2 Graphics System Session
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>

#include <array>
#include <cstdint>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stop_token>
#include <thread>
#include <vector>

namespace armigo::services::gsv {

// What the graphics system sessions need from the launch: the NVIDIA driver
// session that command 1402 hands out (it speaks the nvdrv protocol), and the
// display refresh. wait_refresh blocks until the refresh after `sequence` and
// returns the new sequence, or nothing when the display stops or `stop` is set.
struct GraphicsSystemDependencies {
    std::function<std::shared_ptr<horizon::ClientSession>()> open_driver;
    std::function<std::optional<std::uint64_t>(std::uint64_t sequence, std::stop_token stop)> wait_refresh;
};

// The Switch 2 graphics system session that SelfController command 2000
// hands to the application. It speaks TIPC: the request type is 16 plus the
// command, the reply echoes that type and carries a one-word result followed
// by the data. Commands follow the practical reference until the firmware
// pass; reference stubs answer as provisional handlers that warn once each.
// Any other command is unsupported with an explicit diagnostic.
class GraphicsSystemSession final : public horizon::ClientSession {
public:
    GraphicsSystemSession(std::uint64_t process_id, horizon::HandleTable& handles,
                          GraphicsSystemDependencies dependencies = {});
    horizon::IpcReply close_server() override;
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    const char* diagnostic() const;

private:
    horizon::IpcReply handle(const horizon::IpcMessage&, std::uint64_t sender_pid,
                             const horizon::GuestBuffers* buffers);
    horizon::IpcReply attach_syncpoint_memory(const horizon::IpcMessage&, std::uint32_t type,
                                              const horizon::GuestBuffers* buffers);
    const std::uint64_t process_id_;
    horizon::HandleTable* handles_;
    const GraphicsSystemDependencies dependencies_;
    // Command 204: the client signals the first event to ask for an
    // acknowledgement and waits on the second.
    std::shared_ptr<horizon::Event> syncpoint_request_event_;
    std::shared_ptr<horizon::Event> syncpoint_notify_event_;
    bool closed_ = false;
    bool client_handle_registered_ = false;
    // GPU address spaces attached with command 434, and the command-1111 value.
    std::set<std::uint32_t> attached_address_spaces_;
    std::optional<std::uint32_t> feature_version_;
    // Provisional handlers that already logged their one-time warning.
    std::uint64_t provisional_warned_ = 0;
    const char* diagnostic_ = nullptr;
    mutable std::mutex mutex_;
};

// The syncpoint session that graphics system command 202 hands out
// (nn::syncpt). Each slot owns an event: a request signals it once the
// syncpoint counter reaches a value, and a cancel always signals it. No GPU
// work advances a counter yet, so every counter reads zero.
class SyncpointSession final : public horizon::ClientSession {
public:
    SyncpointSession(std::uint64_t process_id, horizon::HandleTable& handles);
    horizon::IpcReply close_server() override;
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;

private:
    std::shared_ptr<horizon::Event> slot_event(std::uint32_t slot);
    const std::uint64_t process_id_;
    horizon::HandleTable* handles_;
    bool closed_ = false;
    std::map<std::uint32_t, std::shared_ptr<horizon::Event>> events_;
    // Slots with a request whose value has not been reached.
    std::map<std::uint32_t, std::uint64_t> pending_;
    std::uint64_t provisional_warned_ = 0;
    const char* diagnostic_ = nullptr;
    mutable std::mutex mutex_;
};

// The GPU session that graphics system command 497 hands out (nn::gpu::sdk).
// Command 22 hands out another handle to this same session for its memory
// and job commands. Each id in [0, 31] owns an event that job completion
// signals; nothing runs on a GPU here, so jobs complete at once.
class GpuSession final : public horizon::ClientSession, public std::enable_shared_from_this<GpuSession> {
public:
    GpuSession(std::uint64_t process_id, horizon::HandleTable& handles);
    horizon::IpcReply close_server() override;
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;

private:
    std::shared_ptr<horizon::Event> id_event(std::uint32_t id);
    const std::uint64_t process_id_;
    horizon::HandleTable* handles_;
    bool closed_ = false;
    std::array<std::shared_ptr<horizon::Event>, 32> events_{};
    // Graphics device ids from command 41; id 0 is never valid.
    std::array<bool, 256> devices_{};
    std::uint64_t provisional_warned_ = 0;
    const char* diagnostic_ = nullptr;
    mutable std::mutex mutex_;
};

// The display session that graphics system command 305 hands out (nn::vi2).
// Eight native event slots: a new slot fires on every display refresh until
// it is bound elsewhere; a change notification fires once and stays quiet.
class DisplaySession final : public horizon::ClientSession {
public:
    DisplaySession(std::uint64_t process_id, horizon::HandleTable& handles,
                   std::function<std::optional<std::uint64_t>(std::uint64_t, std::stop_token)> wait_refresh);
    ~DisplaySession() override;
    horizon::IpcReply close_server() override;
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;

private:
    struct Slot { std::shared_ptr<horizon::Event> event; bool refresh = false; };
    horizon::IpcReply slot_reply(std::uint32_t type, std::uint32_t index);
    void signal_refresh(std::stop_token stop);
    const std::uint64_t process_id_;
    horizon::HandleTable* handles_;
    const std::function<std::optional<std::uint64_t>(std::uint64_t, std::stop_token)> wait_refresh_;
    bool closed_ = false;
    std::array<Slot, 8> slots_{};
    // Layers opened with command 201: the value and the two client-allocated ids.
    struct Layer { std::uint64_t value; std::uint32_t first_id; std::uint32_t second_id; };
    std::vector<Layer> layers_;
    std::uint64_t provisional_warned_ = 0;
    const char* diagnostic_ = nullptr;
    mutable std::mutex mutex_;
    // Signals the slots bound to the display refresh; declared last so it
    // stops before the slots it reads.
    std::jthread refresh_;
};

} // namespace armigo::services::gsv
