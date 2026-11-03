// ============================================================================
// Armigo Host1x - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/host1x/host1x.h>

#include "engines.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>

namespace armigo::host1x {
namespace {

// Host-interface register indices every engine class shares.
constexpr std::uint32_t host_increment_syncpoint = 0x00;
constexpr std::uint32_t host_method_0 = 0x10;
constexpr std::uint32_t host_method_1 = 0x11;
// Control-class methods.
constexpr std::uint32_t control_wait_syncpoint = 0x08;
constexpr std::uint32_t control_load_syncpoint_payload = 0x4e;
constexpr std::uint32_t control_wait_syncpoint_32 = 0x50;

} // namespace

void FrameQueue::open(std::uint32_t fd) {
    const std::lock_guard lock(mutex_);
    devices_[fd].clear();
}

void FrameQueue::close(std::uint32_t fd) {
    const std::lock_guard lock(mutex_);
    devices_.erase(fd);
}

void FrameQueue::push(std::uint32_t fd, std::uint64_t luma_offset, std::shared_ptr<const DecodedFrame> frame) {
    const std::lock_guard lock(mutex_);
    const auto found = devices_.find(fd);
    if (found == devices_.end()) return;
    auto& frames = found->second;
    if (frames.size() >= max_frames) {
        // Nothing composes this channel's pictures fast enough (or at all).
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true))
            armigo::diag::write(armigo::diag::Level::warning,  "Armigo host1x: decoder channel %u dropped a picture nobody composed\n", fd);
        frames.pop_front();
    }
    frames.emplace_back(luma_offset, std::move(frame));
}

bool FrameQueue::is_open(std::int64_t fd) {
    const std::lock_guard lock(mutex_);
    return fd >= 0 && devices_.contains(static_cast<std::uint32_t>(fd));
}

std::int64_t FrameQueue::find_decoder(std::uint64_t luma_offset) {
    const std::lock_guard lock(mutex_);
    for (const auto& [fd, frames] : devices_)
        for (const auto& entry : frames)
            if (entry.first == luma_offset) return fd;
    return -1;
}

std::shared_ptr<const DecodedFrame> FrameQueue::take(std::int64_t fd) {
    const std::lock_guard lock(mutex_);
    if (fd < 0) return nullptr;
    const auto found = devices_.find(static_cast<std::uint32_t>(fd));
    if (found == devices_.end() || found->second.empty()) return nullptr;
    auto frame = std::move(found->second.front().second);
    found->second.pop_front();
    return frame;
}

CommandPusher::CommandPusher(Hub& hub, ClassId initial, std::uint32_t syncpoint)
    : hub_(hub), syncpoint_(syncpoint), current_class_(initial) {}

CommandPusher::~CommandPusher() { stop(); }

void CommandPusher::start() {
    thread_ = std::jthread([this](std::stop_token stop) { run(stop); });
}

void CommandPusher::stop() {
    if (!thread_.joinable()) return;
    thread_.request_stop();
    changed_.notify_all();
    thread_.join();
}

void CommandPusher::push(std::vector<std::uint32_t> words) {
    const std::lock_guard lock(mutex_);
    lists_.push_back(std::move(words));
    changed_.notify_all();
}

void CommandPusher::drain() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return failed_ || (lists_.empty() && !busy_) || !thread_.joinable(); });
}

bool CommandPusher::execute(std::uint32_t method, std::uint32_t argument, std::stop_token stop,
                            std::string& failure) {
    if (current_class_ == ClassId::control) {
        switch (method) {
        case control_load_syncpoint_payload:
            control_payload_ = argument;
            return true;
        case control_wait_syncpoint:
        case control_wait_syncpoint_32: {
            const auto& syncpoints = hub_.syncpoints();
            if (!syncpoints.wait || !syncpoints.wait(syncpoints.context, argument, control_payload_, stop))
                return stop.stop_requested() ? true : (failure = "Host1x syncpoint wait failed", false);
            return true;
        }
        default:
            failure = "Host1x control method " + std::to_string(method) + " is not implemented";
            return false;
        }
    }
    if (method >= host_registers_.size()) {
        failure = "Host1x host-interface method " + std::to_string(method) + " is out of range";
        return false;
    }
    host_registers_[method] = argument;
    switch (method) {
    case host_increment_syncpoint: {
        // The syncpoint id is the low byte; the condition (bits 8..15) is when the
        // engine has finished the work queued before it, which processing in
        // order already satisfies.
        const auto& syncpoints = hub_.syncpoints();
        if (syncpoints.increment) syncpoints.increment(syncpoints.context, argument & 0xffu);
        return true;
    }
    case host_method_1:
        return process_method(host_registers_[host_method_0], argument, failure);
    default:
        return true;
    }
}

void CommandPusher::run(std::stop_token stop) {
    std::uint32_t count = 0;
    std::uint32_t method = 0;
    std::uint32_t mask = 0;
    bool incrementing = false;
    while (!stop.stop_requested()) {
        std::vector<std::uint32_t> words;
        {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, stop, [&] { return !lists_.empty(); });
            if (stop.stop_requested()) return;
            words = std::move(lists_.front());
            lists_.pop_front();
            busy_ = true;
        }
        std::string failure;
        bool ok = true;
        for (std::size_t index = 0; ok && index < words.size() && !stop.stop_requested(); ++index) {
            const auto value = words[index];
            if (mask != 0) {
                const auto bit = static_cast<std::uint32_t>(std::countr_zero(mask));
                mask &= ~(1u << bit);
                ok = execute(method + bit, value, stop, failure);
                continue;
            }
            if (count != 0) {
                --count;
                ok = execute(method, value, stop, failure);
                if (incrementing) ++method;
                continue;
            }
            const auto mode = static_cast<SubmissionMode>(value >> 28);
            const auto offset = (value >> 16) & 0xfffu;
            const auto data = value & 0xffffu;
            switch (mode) {
            case SubmissionMode::set_class:
                mask = data & 0x3fu;
                method = offset;
                current_class_ = static_cast<ClassId>((data >> 6) & 0x3ffu);
                break;
            case SubmissionMode::incrementing:
            case SubmissionMode::non_incrementing:
                count = data;
                method = offset;
                incrementing = mode == SubmissionMode::incrementing;
                break;
            case SubmissionMode::mask:
                mask = data;
                method = offset;
                break;
            case SubmissionMode::immediate:
                method = offset;
                ok = execute(method, value & 0xfffu, stop, failure);
                break;
            default:
                failure = "Host1x submission mode " + std::to_string(static_cast<unsigned>(mode)) +
                          " is not implemented";
                ok = false;
                break;
            }
        }
        // The failure sink takes the driver's lock, which a submitting thread
        // holds while it queues work here, so report before taking ours.
        if (!ok) {
            const auto& failures = hub_.failures();
            if (failures.fail) failures.fail(failures.context, syncpoint_, std::move(failure));
        }
        const std::lock_guard lock(mutex_);
        busy_ = false;
        if (!ok) {
            failed_ = true;
            lists_.clear();
            changed_.notify_all();
            return;
        }
        changed_.notify_all();
    }
}

Hub::Hub(DeviceMemory memory, Syncpoints syncpoints, FailureSink failures)
    : memory_(memory), syncpoints_(syncpoints), failures_(failures) {}

Hub::~Hub() {
    std::map<std::uint32_t, std::shared_ptr<CommandPusher>> devices;
    {
        const std::lock_guard lock(mutex_);
        devices.swap(devices_);
    }
    devices.clear();
}

void Hub::start_device(std::uint32_t fd, ChannelType type, std::uint32_t syncpoint) {
    std::unique_ptr<CommandPusher> device;
    if (type == ChannelType::video_decoder) {
        frames_.open(fd);
        device = make_video_decoder(*this, fd, syncpoint);
    } else {
        device = make_video_compositor(*this, fd, syncpoint);
    }
    const std::lock_guard lock(mutex_);
    devices_[fd] = std::move(device);
}

void Hub::stop_device(std::uint32_t fd) {
    std::shared_ptr<CommandPusher> device;
    {
        const std::lock_guard lock(mutex_);
        const auto found = devices_.find(fd);
        if (found == devices_.end()) return;
        device = std::move(found->second);
        devices_.erase(found);
    }
    device.reset();
    frames_.close(fd);
}

bool Hub::push(std::uint32_t fd, std::vector<std::uint32_t> words) {
    const std::lock_guard lock(mutex_);
    const auto found = devices_.find(fd);
    if (found == devices_.end()) return false;
    found->second->push(std::move(words));
    return true;
}

void Hub::drain(std::uint32_t fd) {
    std::shared_ptr<CommandPusher> device;
    {
        const std::lock_guard lock(mutex_);
        const auto found = devices_.find(fd);
        if (found == devices_.end()) return;
        device = found->second;
    }
    device->drain();
}

} // namespace armigo::host1x
