// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/audio/pcm_output.h>
#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace armigo::services::audout {

using OutputFactory = std::function<std::shared_ptr<audio::PcmOutput>()>;

// One opened PCM output (IAudioOut): 48 kHz PCM16 with 2 or 6 channels.
// QueryPointerBufferSize reports no pointer buffer, so the SDK sends mapped
// buffers. Start (1), Stop (2), RegisterBufferEvent (4),
// AppendAudioOutBufferAuto (7) and GetReleasedAudioOutBuffersAuto (8) are
// served; other methods are explicit unsupported diagnostics.
//
// Appended samples are copied at Append. A worker plays the queue in order
// while started: it writes each buffer to the host output (when there is one)
// and releases it once its duration has passed on the host clock, signalling
// the buffer event, so a muted run is paced like an audible one. Queued
// buffers follow each other on an absolute schedule, so releases keep real
// time however late the worker wakes.
class AudioOutSession final : public horizon::ClientSession {
public:
    AudioOutSession(std::uint64_t process_id, unsigned channels, std::shared_ptr<audio::PcmOutput> output);
    ~AudioOutSession() override;
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;
    // Most buffers queued at once, as firmware allows.
    static constexpr std::size_t max_buffers = 32;

private:
    struct Entry {
        std::uint64_t tag;
        std::vector<std::int16_t> samples;
    };
    horizon::IpcReply method(const horizon::IpcMessage&, const horizon::GuestBuffers*);
    void play(std::stop_token stop);

    const std::uint64_t process_id_;
    const unsigned channels_;
    const std::shared_ptr<audio::PcmOutput> output_;
    const std::shared_ptr<horizon::Event> buffer_event_;
    mutable std::mutex mutex_;
    std::condition_variable_any wake_;
    std::deque<Entry> queued_;
    std::deque<std::uint64_t> released_;
    bool started_ = false;
    const char* diagnostic_ = nullptr;
    std::jthread worker_;
};

// The application's audout:u manager: QueryPointerBufferSize and
// OpenAudioOut (1) for the default output "DeviceOut" at 48 kHz.
class ManagerSession final : public horizon::ClientSession {
public:
    ManagerSession(std::uint64_t process_id, OutputFactory output_factory = {});
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;

private:
    horizon::IpcReply method(const horizon::IpcMessage&, const horizon::GuestBuffers*);
    const std::uint64_t process_id_;
    const OutputFactory output_factory_;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::audout
