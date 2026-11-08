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
#include <armigo/host1x/video_decoder.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <tuple>
#include <span>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
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
    // The display that native windows of Armigo's NVN driver present through,
    // owned with the graphics system; null when the driver is not selected.
    std::shared_ptr<void> native_display;
    // Writes guest memory outside a request: video decoder sessions put
    // decoded pictures into the client's output images with it.
    std::function<bool(std::uint64_t address, std::span<const std::byte> bytes)> write_memory;
    // Reads guest memory outside a request: the syncpoint memory's armed waits.
    std::function<bool(std::uint64_t address, std::span<std::byte> bytes)> read_memory;
    // The host clock movie pacing follows; the steady clock when empty.
    std::function<std::chrono::steady_clock::time_point()> clock;
};

// Syncpoint counters that video decoder services advance, one per decoder;
// syncpoint sessions read them and wait on them. Counters nobody allocated
// stay at zero: no GPU work advances them here.
class SyncpointCounters {
public:
    // The client's syncpoint memory (graphics system command 204).
    struct SharedMemory {
        std::uint64_t writable = 0;
        std::uint64_t readable = 0;
        std::uint32_t count = 0;
        std::function<bool(std::uint64_t, std::span<std::byte>)> read;
        std::function<bool(std::uint64_t, std::span<const std::byte>)> write;
        std::shared_ptr<horizon::Event> notify;
    };
    std::uint32_t allocate();
    std::uint64_t value(std::uint32_t id) const;
    void increment(std::uint32_t id);
    // Signals `event` once counter `id` reaches `value` (mode 0 compares
    // wrapping 32-bit values, mode 1 64-bit values); true when it already has.
    bool wait(std::uint32_t id, std::uint64_t value, std::uint32_t mode, std::shared_ptr<horizon::Event> event);
    void cancel(const std::shared_ptr<horizon::Event>& event);
    // Publishes every counter value into the read-only memory from now on and
    // acknowledges the waits the client arms in the read-write memory.
    void attach(SharedMemory memory);
    // The client signalled its request event after arming waits.
    void serve_requests();

private:
    struct Waiter {
        std::uint32_t id;
        std::uint64_t value;
        std::uint32_t mode;
        std::weak_ptr<horizon::Event> event;
    };
    static bool reached(std::uint64_t counter, std::uint64_t value, std::uint32_t mode);
    bool acknowledge();
    mutable std::mutex mutex_;
    std::uint32_t next_ = 64;
    std::map<std::uint32_t, std::uint64_t> counters_;
    std::vector<Waiter> waiters_;
    std::optional<SharedMemory> memory_;
    // Each syncpoint's counter value when its wait was last acknowledged.
    std::map<std::uint32_t, std::uint64_t> acknowledged_;
};

// Native objects the graphics system names by id: memories (command 111,
// bound to process memory with 114) and decoder output images in them
// (911). Video decoder sessions write decoded pictures into the images.
struct NativeObjects {
    struct Memory {
        std::uint64_t size = 0;
        std::uint64_t address = 0;
    };
    struct Image {
        std::uint32_t memory = 0;
        std::uint64_t offset = 0;
        std::uint32_t code = 0;
        std::uint32_t extent = 0;
    };
    std::mutex mutex;
    std::uint32_t next_memory = 1;
    std::uint32_t next_image = 1;
    std::map<std::uint32_t, Memory> memories;
    std::map<std::uint32_t, Image> images;
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
    const std::shared_ptr<NativeObjects> objects_ = std::make_shared<NativeObjects>();
    const std::shared_ptr<SyncpointCounters> counters_ = std::make_shared<SyncpointCounters>();
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
    SyncpointSession(std::uint64_t process_id, horizon::HandleTable& handles,
                     std::shared_ptr<SyncpointCounters> counters = std::make_shared<SyncpointCounters>());
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
    const std::shared_ptr<SyncpointCounters> counters_;
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

// The video decoder session that graphics system commands 906 and 1002 hand
// out (nn::codec::video::detail::ipc); its command 1 hands out a decoder
// service, another session of this class. Its server is not part of the
// supplied system-code dump, so the protocol follows the practical reference
// as a provisional contract that warns once per session. Requests it does not
// serve stay explicitly unsupported, naming the command.
class VideoSession final : public horizon::ClientSession {
public:
    VideoSession(std::uint64_t process_id, horizon::HandleTable& handles,
                 std::shared_ptr<NativeObjects> objects = std::make_shared<NativeObjects>(),
                 std::shared_ptr<SyncpointCounters> counters = std::make_shared<SyncpointCounters>(),
                 std::function<bool(std::uint64_t, std::span<const std::byte>)> write_memory = {},
                 std::function<std::chrono::steady_clock::time_point()> clock = {});
    ~VideoSession() override;
    horizon::IpcReply close_server() override;
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    const char* diagnostic() const;
private:
    horizon::IpcReply handle(const horizon::IpcMessage&, std::uint64_t sender_pid, const horizon::GuestBuffers*);
    horizon::IpcReply logged(horizon::IpcReply reply) const;
    horizon::IpcReply unsupported_command(std::uint32_t command, const horizon::IpcMessage&);
    struct Output {
        std::uint32_t image;
        std::uint32_t width;
        std::uint32_t height;
        std::uint64_t pts_ns;
        bool end;
    };
    std::array<std::uint32_t, 4> hand_out(std::uint32_t image);
    bool place_frames(const char*& failure);
    void update_ready();
    void complete_handed();
    std::int64_t now_us() const;
    void anchor_clock(std::uint64_t pts_us);
    std::optional<std::uint64_t> due_pts_us() const;
    void drop_overtaken_frames();
    const std::uint64_t process_id_;
    horizon::HandleTable* handles_;
    const std::shared_ptr<NativeObjects> objects_;
    const std::shared_ptr<SyncpointCounters> counters_;
    const std::function<bool(std::uint64_t, std::span<const std::byte>)> write_memory_;
    const std::function<std::chrono::steady_clock::time_point()> clock_;
    std::unique_ptr<host1x::Av1Decoder> decoder_;
    std::string decoder_failure_;
    std::optional<std::uint32_t> syncpoint_;
    std::uint64_t handed_ = 0;
    std::uint64_t submitted_ = 0;
    std::uint64_t placed_ = 0;
    std::chrono::steady_clock::time_point first_placed_;
    std::deque<std::uint32_t> free_images_;
    // Decoded frames waiting for a free image, with their pts and end flag;
    // a null frame is an end of stream without a picture.
    std::deque<std::tuple<std::shared_ptr<const host1x::DecodedFrame>, std::uint64_t, bool>> pending_frames_;
    std::deque<Output> outputs_;
    // Movie pacing: host time minus the pts of the frames the client shows,
    // from its acquires (213). A longer gap between acquires is a pause, a
    // longer lag a jump in the stream; both restart the clock.
    std::optional<std::int64_t> clock_offset_us_;
    std::int64_t last_acquire_us_ = 0;
    std::uint64_t last_acquired_pts_us_ = 0;
    std::uint32_t dropped_in_a_row_ = 0;
    std::uint64_t dropped_ = 0;
    static constexpr std::int64_t pause_gap_us = 1'000'000;
    static constexpr std::int64_t max_catch_up_us = 3'000'000;
    static constexpr std::uint32_t max_dropped_in_a_row = 30;
    // nn::codec description 3603: the decoder takes no input now.
    static constexpr std::uint32_t decoder_busy_result = 0x1c266f;
    static constexpr std::size_t busy_frames = 12;
    bool closed_ = false;
    bool warned_ = false;
    bool busy_logged_ = false;
    bool running_ = false;
    // Command 3: signaled while the decoder takes another input.
    std::shared_ptr<horizon::Event> ready_event_;
    std::string codec_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    char diagnostic_text_[96] = {};
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
