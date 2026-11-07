// ============================================================================
// Armigo Host1x - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace armigo::host1x {

// Engine memory: the multimedia engines address guest memory through device
// addresses (the 32-bit addresses MapBuffer pins NVMap handles at). Reads and
// writes fail for an address no pin covers.
struct DeviceMemory {
    void* context = nullptr;
    bool (*read)(void*, std::uint64_t address, std::span<std::byte> out) noexcept = nullptr;
    bool (*write)(void*, std::uint64_t address, std::span<const std::byte> in) noexcept = nullptr;
};

// The process's syncpoints: an engine increments one when its work completes
// and can wait until one reaches a threshold. A wait returns false when the
// processing is stopped.
struct Syncpoints {
    void* context = nullptr;
    void (*increment)(void*, std::uint32_t id) noexcept = nullptr;
    bool (*wait)(void*, std::uint32_t id, std::uint32_t threshold, std::stop_token stop) noexcept = nullptr;
};

// Where an engine reports work it cannot do. Processing of that channel stops.
struct FailureSink {
    void* context = nullptr;
    void (*fail)(void*, std::uint32_t syncpoint, std::string detail) noexcept = nullptr;
};

// Command header submission modes and engine classes of the host1x command stream.
enum class SubmissionMode : std::uint32_t {
    set_class = 0,
    incrementing = 1,
    non_incrementing = 2,
    mask = 3,
    immediate = 4,
    restart = 5,
    gather = 6,
};

enum class ClassId : std::uint32_t {
    none = 0x0,
    control = 0x1,
    video_compositor = 0x5d,
    video_decoder = 0xf0,
};

enum class ChannelType : std::uint32_t { video_compositor = 1, video_decoder = 3 };

class Hub;

// A decoded picture: 8-bit Y plane and interleaved UV plane (4:2:0).
struct DecodedFrame {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t luma_stride = 0;
    std::uint32_t chroma_stride = 0;
    std::vector<std::uint8_t> luma;
    std::vector<std::uint8_t> chroma;  // U, V interleaved, (height + 1) / 2 rows
};

// Decoded frames waiting for the compositor. A decoder channel queues the
// pictures it shows in decode order, each with the luma surface offset it was
// decoded for. The compositor reads the surface it names: the channel holding
// that surface supplies its latest picture, and the older pictures queued
// before it go too, because the player skipped them and the hardware overwrote
// them. A picture shown again from an earlier surface is not queued under the
// surface named, so the oldest queued picture of the channel is next instead.
class FrameQueue {
public:
    void open(std::uint32_t fd);
    void close(std::uint32_t fd);
    void push(std::uint32_t fd, std::uint64_t luma_offset, std::shared_ptr<const DecodedFrame> frame);
    // The decoder channel with a queued picture for `luma_offset`, or -1.
    std::int64_t find_decoder(std::uint64_t luma_offset);
    // The latest queued picture of decoder channel `fd` for `luma_offset`,
    // dropping the pictures queued before it; without one, the channel's
    // oldest queued picture. Null when the channel has none.
    std::shared_ptr<const DecodedFrame> take(std::int64_t fd, std::uint64_t luma_offset);
    // Whether decoder channel `fd` is open.
    bool is_open(std::int64_t fd);

private:
    static constexpr std::size_t max_frames = 100;
    std::mutex mutex_;
    std::map<std::uint32_t, std::deque<std::pair<std::uint64_t, std::shared_ptr<const DecodedFrame>>>> devices_;
};

// One engine channel: its command lists run in order on a worker thread. The
// host-interface registers route class methods to the engine and increment
// syncpoints; the control class waits for syncpoints.
class CommandPusher {
public:
    virtual ~CommandPusher();
    CommandPusher(const CommandPusher&) = delete;
    CommandPusher& operator=(const CommandPusher&) = delete;
    void push(std::vector<std::uint32_t> words);
    // Blocks until every pushed command list has run (or processing stopped).
    void drain();

protected:
    CommandPusher(Hub& hub, ClassId initial, std::uint32_t syncpoint);
    // Starts the worker; derived engines call it once they are constructed.
    void start();
    // Stops the worker; derived engines call it before their state is destroyed.
    void stop();
    // An engine method. False with `failure` set when the engine cannot do it.
    virtual bool process_method(std::uint32_t method, std::uint32_t argument, std::string& failure) = 0;
    Hub& hub_;
    const std::uint32_t syncpoint_;

private:
    bool execute(std::uint32_t method, std::uint32_t argument, std::stop_token stop, std::string& failure);
    void run(std::stop_token stop);
    std::array<std::uint32_t, 0x20> host_registers_{};
    std::uint32_t control_payload_ = 0;
    ClassId current_class_;
    std::mutex mutex_;
    std::condition_variable_any changed_;
    std::deque<std::vector<std::uint32_t>> lists_;
    bool busy_ = false;
    bool failed_ = false;
    std::jthread thread_;
};

// The multimedia engines of one process, by channel descriptor.
class Hub {
public:
    Hub(DeviceMemory memory, Syncpoints syncpoints, FailureSink failures);
    ~Hub();
    Hub(const Hub&) = delete;
    Hub& operator=(const Hub&) = delete;
    // Creates the engine of a newly opened channel.
    void start_device(std::uint32_t fd, ChannelType type, std::uint32_t syncpoint);
    void stop_device(std::uint32_t fd);
    // Queues one command list for the channel; false when it has no engine.
    bool push(std::uint32_t fd, std::vector<std::uint32_t> words);
    void drain(std::uint32_t fd);
    const DeviceMemory& memory() const noexcept { return memory_; }
    const Syncpoints& syncpoints() const noexcept { return syncpoints_; }
    const FailureSink& failures() const noexcept { return failures_; }
    FrameQueue& frames() noexcept { return frames_; }

private:
    DeviceMemory memory_;
    Syncpoints syncpoints_;
    FailureSink failures_;
    FrameQueue frames_;
    std::mutex mutex_;
    // Shared so a drain keeps its engine alive while the channel closes.
    std::map<std::uint32_t, std::shared_ptr<CommandPusher>> devices_;
};

} // namespace armigo::host1x
