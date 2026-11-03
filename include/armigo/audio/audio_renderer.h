// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/audio/pcm_output.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stop_token>
#include <thread>

namespace armigo::audio {

// AudioRendererParameterInternal: the OpenAudioRenderer and GetWorkBufferSize
// input as the guest serializes it.
struct AudioRendererParameter {
    std::uint32_t sample_rate = 0;
    std::uint32_t sample_count = 0;
    std::uint32_t mixes = 0;
    std::uint32_t sub_mixes = 0;
    std::uint32_t voices = 0;
    std::uint32_t sinks = 0;
    std::uint32_t effects = 0;
    std::uint32_t performance_frames = 0;
    std::uint8_t voice_drop_enabled = 0;
    std::uint8_t reserved = 0;
    std::uint8_t rendering_device = 0;
    std::uint8_t execution_mode = 0;
    std::uint32_t splitter_infos = 0;
    std::int32_t splitter_destinations = 0;
    std::uint32_t external_context_size = 0;
    std::uint32_t revision = 0;
};
static_assert(sizeof(AudioRendererParameter) == 0x34);

// Horizon audio results the service layer returns itself.
inline constexpr std::uint32_t result_not_supported = 153 | (513u << 9);

// Pins guest ranges the renderer may access: an attached memory pool stays
// mapped while it is pinned. Implementations are supplied by the service.
class GuestMemoryPins {
public:
    virtual ~GuestMemoryPins() = default;
    virtual bool pin(std::uint64_t address, std::uint64_t size) = 0;
    virtual void unpin(std::uint64_t address, std::uint64_t size) = 0;
};

class AudioRendererSystem;

// One Horizon audio renderer (an IAudioRenderer object's state): the guest's
// voices, mixes, effects, splitters, sinks and memory pools as updated through
// RequestUpdate, and the command list rendered from them every audio frame.
// In auto execution mode a host render thread renders one frame per
// sample_count / sample_rate seconds while the renderer is started, signals
// `frame_rendered` per frame and feeds the device sink into `output`.
class AudioRenderer {
public:
    struct Options {
        std::shared_ptr<GuestMemoryPins> memory;
        std::shared_ptr<PcmOutput> output;
        // Called on the render thread after each frame's command list is
        // generated; the service signals the renderer's system event here.
        std::function<void()> frame_rendered;
        // Called under the renderer lock after each successful update; the
        // service clears the renderer's system event here.
        std::function<void()> update_completed;
        // False keeps rendering manual (render_frame); used by tests.
        bool render_thread = true;
    };

    AudioRenderer();
    ~AudioRenderer();
    AudioRenderer(const AudioRenderer&) = delete;
    AudioRenderer& operator=(const AudioRenderer&) = delete;

    // Horizon result codes (0 on success).
    static std::uint32_t work_buffer_size(const AudioRendererParameter& parameter,
                                          std::uint64_t& size);
    std::uint32_t initialize(const AudioRendererParameter& parameter,
                             std::uint64_t work_buffer_size, Options options);
    void finalize();

    std::uint32_t update(std::span<const std::byte> input, std::span<std::byte> performance,
                         std::span<std::byte> output);
    void start();
    void stop();
    // Renders one audio frame now, exactly as the render thread does.
    void render_frame();

    bool initialized() const noexcept;
    bool active() const noexcept;
    std::uint32_t sample_rate() const noexcept;
    std::uint32_t sample_count() const noexcept;
    std::uint32_t mix_buffer_count() const noexcept;
    std::uint8_t execution_mode() const noexcept;
    std::uint32_t rendering_time_limit() const noexcept;
    void set_rendering_time_limit(std::uint32_t percent) noexcept;
    float voice_drop_parameter() const noexcept;
    void set_voice_drop_parameter(float value) noexcept;
    std::uint64_t rendered_frames() const noexcept;

private:
    void render_loop(std::stop_token stop);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace armigo::audio
