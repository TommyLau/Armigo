// ============================================================================
// Armigo Services - Audio Renderer Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/audio/audio_renderer.h>
#include <armigo/audio/pcm_output.h>
#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>

#include <cstdint>
#include <functional>
#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace armigo::services::audren {

// AudioRendererParameterInternal as the audren client serializes it. The
// parameter is the request payload after the CMIF header.
using RendererParameter = audio::AudioRendererParameter;

// Creates the host output for one renderer. An absent factory, or a factory
// that returns null, leaves the renderer on an explicit muted output.
using OutputFactory = std::function<std::shared_ptr<audio::PcmOutput>()>;

// The host outputs of one audren:u manager: renderer outputs share the audio
// device's output volume, which scales their device sink output.
class HostAudio {
public:
    explicit HostAudio(OutputFactory factory = {});
    std::shared_ptr<audio::PcmOutput> open_output();
    void set_volume(float volume);
    float volume() const noexcept;

private:
    OutputFactory factory_;
    mutable std::mutex mutex_;
    float volume_ = 1.0f;
    std::vector<std::weak_ptr<audio::PcmOutput>> outputs_;
};

// Pins the guest ranges of attached memory pools through the runtime's mapper.
// Every IPC request refreshes the mapper; each pin remembers the mapper that
// made it so the matching unpin releases it there.
class PoolPins final : public audio::GuestMemoryPins {
public:
    void set_mapper(horizon::GuestMemoryMapper mapper);
    bool pin(std::uint64_t address, std::uint64_t size) override;
    void unpin(std::uint64_t address, std::uint64_t size) override;
    std::size_t pinned() const;

private:
    struct Pin {
        std::uint64_t address = 0;
        std::uint64_t size = 0;
        horizon::GuestMemoryMapper mapper;
    };
    mutable std::mutex mutex_;
    horizon::GuestMemoryMapper mapper_{};
    std::vector<Pin> pins_;
};

// The audio renderer object returned by OpenAudioRenderer
// (nn::audio::detail::IAudioRenderer). RequestUpdate applies the guest's
// voice, mix, effect, splitter, sink and memory-pool updates and writes their
// statuses; while started, the renderer renders one frame per
// sample_count / sample_rate on its own thread, signals the system event per
// frame and plays the device sink through the host output. RequestUpdate
// clears the event; it never signals it.
class RendererSession final : public horizon::ClientSession {
public:
    struct Options {
        std::uint64_t work_buffer_size = 0;
        std::shared_ptr<HostAudio> host;
        // False renders only through render_frame (tests).
        bool render_thread = true;
    };

    RendererSession(const RendererParameter& parameter, Options options);
    ~RendererSession() override;

    // Horizon result of initialization; the manager returns it to the guest.
    std::uint32_t result() const noexcept { return result_; }

    void set_memory_mapper(horizon::GuestMemoryMapper mapper) noexcept override;
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;

    std::uint32_t sample_rate() const noexcept { return parameter_.sample_rate; }
    std::uint32_t sample_count() const noexcept { return parameter_.sample_count; }
    std::uint32_t mix_buffer_count() const noexcept { return parameter_.mixes; }
    bool started() const noexcept;
    const char* diagnostic() const noexcept { return diagnostic_; }
    // Renders one frame immediately, as the render thread does.
    void render_frame() { renderer_.render_frame(); }
    std::uint64_t rendered_frames() const noexcept { return renderer_.rendered_frames(); }
    const std::shared_ptr<PoolPins>& pins() const noexcept { return pins_; }

private:
    horizon::IpcReply unsupported(const char* detail);

    RendererParameter parameter_{};
    std::shared_ptr<horizon::Event> event_;
    std::shared_ptr<PoolPins> pins_;
    std::shared_ptr<audio::PcmOutput> output_;
    audio::AudioRenderer renderer_;
    std::uint32_t result_ = 0;
    std::mutex diagnostic_mutex_;
    const char* diagnostic_ = nullptr;
};

// The audio device object returned by GetAudioDeviceService. It exposes one
// built-in output device with a fixed stereo channel count and a settable
// volume that scales the host output.
class AudioDeviceSession final : public horizon::ClientSession {
public:
    AudioDeviceSession(std::uint64_t aruid, std::uint32_t revision,
                       std::shared_ptr<HostAudio> host = {});

    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;

    const char* name() const noexcept { return "AudioBuiltInSpeakerOutput"; }
    std::uint32_t channel_count() const noexcept { return 2; }
    float volume() const noexcept;
    const char* diagnostic() const noexcept { return diagnostic_; }

private:
    horizon::IpcReply unsupported(const char* detail);
    [[maybe_unused]] std::uint64_t aruid_ = 0;
    [[maybe_unused]] std::uint32_t revision_ = 0;
    std::shared_ptr<HostAudio> host_;
    mutable std::mutex mutex_;
    float volume_ = 1.0f;
    std::shared_ptr<horizon::Event> event_;
    std::shared_ptr<horizon::Event> notification_;  // device-set changes, never signalled
    const char* diagnostic_ = nullptr;
};

// The audren:u manager. Commands are the CMIF methods of
// nn::audio::detail::IAudioRendererManager.
class ManagerSession final : public horizon::ClientSession {
public:
    explicit ManagerSession(std::uint64_t process_id, OutputFactory output_factory = {});

    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;

    // The transport accepts 0x14 bytes of inline data; the client uses the
    // smaller value from this session's response.
    static constexpr std::uint16_t pointer_buffer_capacity = 0x14;

    const char* diagnostic() const noexcept { return diagnostic_; }
    // Renderer sessions render on their own threads unless disabled (tests).
    void set_render_threads(bool enabled) noexcept { render_threads_ = enabled; }

private:
    horizon::IpcReply unsupported(const char* detail);

    std::uint64_t process_id_ = 0;
    std::shared_ptr<HostAudio> host_;
    bool render_threads_ = true;
    const char* diagnostic_ = nullptr;
};

}  // namespace armigo::services::audren
