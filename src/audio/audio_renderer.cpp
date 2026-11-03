// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/audio/audio_renderer.h>

#include "renderer/common/audio_renderer_parameter.h"
#include "renderer/common/feature_support.h"
#include "renderer/common/result.h"
#include "renderer/system.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace armigo::audio {
namespace {

using renderer::AudioRendererParameterInternal;

AudioRendererParameterInternal internal_parameter(const AudioRendererParameter& parameter) {
    static_assert(sizeof(AudioRendererParameterInternal) == sizeof(AudioRendererParameter));
    AudioRendererParameterInternal internal{};
    std::memcpy(&internal, &parameter, sizeof internal);
    return internal;
}

// Renderer session ids. The firmware renders at most MaxRendererSessions
// renderers at once; the id selects the renderer's DSP command buffer slot.
class SessionIds {
public:
    std::int32_t acquire() {
        const std::lock_guard lock(mutex_);
        for (std::size_t index = 0; index < used_.size(); ++index) {
            if (!used_[index]) {
                used_[index] = true;
                return static_cast<std::int32_t>(index);
            }
        }
        return -1;
    }
    void release(std::int32_t id) {
        const std::lock_guard lock(mutex_);
        if (id >= 0 && static_cast<std::size_t>(id) < used_.size()) used_[static_cast<std::size_t>(id)] = false;
    }

private:
    std::mutex mutex_;
    std::array<bool, renderer::MaxRendererSessions> used_{};
};

SessionIds& session_ids() {
    static SessionIds ids;
    return ids;
}

std::uint64_t monotonic_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count());
}

}  // namespace

struct AudioRenderer::Impl {
    renderer::System system;
    std::function<void()> frame_rendered;
    std::function<void()> update_completed;
    std::int32_t session_id = -1;
    std::uint64_t frame_ns = 5'000'000;
    std::mutex wake_mutex;
    std::condition_variable_any wake;
    std::atomic<std::uint64_t> rendered{0};
    std::atomic<std::uint64_t> late_wakes{0};
    std::jthread thread;
};

AudioRenderer::AudioRenderer() : impl_(std::make_unique<Impl>()) {}

AudioRenderer::~AudioRenderer() { finalize(); }

std::uint32_t AudioRenderer::work_buffer_size(const AudioRendererParameter& parameter,
                                              std::uint64_t& size) {
    const auto internal = internal_parameter(parameter);
    if (!renderer::check_valid_revision(internal.revision))
        return renderer::ResultInvalidRevision.raw;
    size = renderer::System::get_work_buffer_size(internal);
    return 0;
}

std::uint32_t AudioRenderer::initialize(const AudioRendererParameter& parameter,
                                        std::uint64_t work_buffer_size, Options options) {
    if (impl_->system.is_initialized()) return renderer::ResultOperationFailed.raw;
    const auto session = session_ids().acquire();
    if (session < 0) return renderer::ResultOutOfSessions.raw;
    const auto internal = internal_parameter(parameter);
    const auto result = impl_->system.initialize(internal, work_buffer_size, std::move(options.memory),
                                                 std::move(options.output), session);
    if (result.is_error()) {
        session_ids().release(session);
        return result.raw;
    }
    impl_->session_id = session;
    impl_->frame_rendered = std::move(options.frame_rendered);
    impl_->update_completed = std::move(options.update_completed);
    if (internal.sample_rate != 0 && internal.sample_count != 0)
        impl_->frame_ns = 1'000'000'000ull * internal.sample_count / internal.sample_rate;
    // Only auto execution renders on its own; manual execution renders when
    // the guest asks, which this renderer does not support.
    if (options.render_thread && internal.execution_mode == renderer::ExecutionMode::Auto)
        impl_->thread = std::jthread([this](std::stop_token stop) { render_loop(stop); });
    return 0;
}

void AudioRenderer::finalize() {
    if (impl_->thread.joinable()) {
        impl_->thread.request_stop();
        impl_->wake.notify_all();
        impl_->thread.join();
    }
    if (!impl_->system.is_initialized()) return;
    impl_->system.finalize();
    session_ids().release(impl_->session_id);
    impl_->session_id = -1;
}

std::uint32_t AudioRenderer::update(std::span<const std::byte> input,
                                    std::span<std::byte> performance,
                                    std::span<std::byte> output) {
    const auto as_bytes = [](auto span) {
        using Element = std::conditional_t<std::is_const_v<typename decltype(span)::element_type>,
                                           const renderer::u8, renderer::u8>;
        return std::span<Element>(reinterpret_cast<Element*>(span.data()), span.size());
    };
    return impl_->system
        .update(as_bytes(input), as_bytes(performance), as_bytes(output), impl_->update_completed)
        .raw;
}

void AudioRenderer::start() {
    impl_->system.start();
    const std::lock_guard lock(impl_->wake_mutex);
    impl_->wake.notify_all();
}

void AudioRenderer::stop() { impl_->system.stop(); }

void AudioRenderer::render_frame() {
    const bool rendered = impl_->system.render_frame([this] {
        if (impl_->frame_rendered) impl_->frame_rendered();
    });
    if (rendered) impl_->rendered.fetch_add(1, std::memory_order_relaxed);
}

// One frame per sample_count / sample_rate on an absolute monotonic deadline.
// A late frame resumes the cadence from now instead of rendering a burst.
// While the renderer is stopped the thread parks until it starts again.
void AudioRenderer::render_loop(std::stop_token stop) {
#if defined(__APPLE__)
    pthread_setname_np("armigo.audio_renderer");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    std::uint64_t next = 0;
    std::unique_lock lock(impl_->wake_mutex);
    while (!stop.stop_requested()) {
        if (!impl_->system.is_active()) {
            next = 0;
            impl_->wake.wait(lock, stop, [this] { return impl_->system.is_active(); });
            continue;
        }
        auto now = monotonic_ns();
        if (next == 0) next = now + impl_->frame_ns;
        if (now < next) {
            impl_->wake.wait_for(lock, stop, std::chrono::nanoseconds(next - now), [] { return false; });
            continue;
        }
        lock.unlock();
        render_frame();
        lock.lock();
        next += impl_->frame_ns;
        now = monotonic_ns();
        if (next <= now) {
            impl_->late_wakes.fetch_add(1, std::memory_order_relaxed);
            next = now + impl_->frame_ns;
        }
    }
}

bool AudioRenderer::initialized() const noexcept { return impl_->system.is_initialized(); }
bool AudioRenderer::active() const noexcept { return impl_->system.is_active(); }
std::uint32_t AudioRenderer::sample_rate() const noexcept { return impl_->system.get_sample_rate(); }
std::uint32_t AudioRenderer::sample_count() const noexcept { return impl_->system.get_sample_count(); }
std::uint32_t AudioRenderer::mix_buffer_count() const noexcept {
    return impl_->system.get_mix_buffer_count();
}
std::uint8_t AudioRenderer::execution_mode() const noexcept {
    return static_cast<std::uint8_t>(impl_->system.get_execution_mode());
}
std::uint32_t AudioRenderer::rendering_time_limit() const noexcept {
    return impl_->system.get_rendering_time_limit();
}
void AudioRenderer::set_rendering_time_limit(std::uint32_t percent) noexcept {
    impl_->system.set_rendering_time_limit(percent);
}
float AudioRenderer::voice_drop_parameter() const noexcept {
    return impl_->system.get_voice_drop_parameter();
}
void AudioRenderer::set_voice_drop_parameter(float value) noexcept {
    impl_->system.set_voice_drop_parameter(value);
}
std::uint64_t AudioRenderer::rendered_frames() const noexcept {
    return impl_->rendered.load(std::memory_order_relaxed);
}

}  // namespace armigo::audio
