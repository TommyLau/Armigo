// ============================================================================
// Armigo Services - Audio Renderer Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/audio_renderer.h>

#include <algorithm>
#include <cstring>
#include <iterator>
#include <vector>

namespace armigo::services::audren {
namespace {

using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;

std::uint32_t read(const IpcMessage& message, std::size_t offset, unsigned width = 4) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index != width; ++index)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + index])}
                 << (8 * index);
    return value;
}

std::uint64_t read64(const IpcMessage& message, std::size_t offset) {
    return std::uint64_t{read(message, offset)} |
           (std::uint64_t{read(message, offset + 4)} << 32);
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value, unsigned width = 4) {
    for (unsigned index = 0; index != width; ++index)
        message[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
}


// A CMIF reply with no output parameters. The request's CMIF offset is used so
// the client finds the output header where its proxy expects it.
IpcReply success_reply(std::size_t cmif) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = cmif + 16;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, cmif, sfco);
    return reply;
}

IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 28, 0xec6be3ff);
    write(reply.bytes, 32, value, width);
    return reply;
}

// A CMIF reply carrying only a Horizon result code.
IpcReply result_reply(std::size_t cmif, std::uint32_t result) {
    IpcReply reply = success_reply(cmif);
    write(reply.bytes, cmif + 8, result);
    return reply;
}

IpcReply value_reply(std::size_t cmif, std::uint64_t value, unsigned width) {
    IpcReply reply = success_reply(cmif);
    reply.size = cmif + 16 + width;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, cmif + 16, static_cast<std::uint32_t>(value));
    if (width == 8) write(reply.bytes, cmif + 20, static_cast<std::uint32_t>(value >> 32));
    return reply;
}

RendererParameter parameter_from(const IpcMessage& message, std::size_t offset) {
    RendererParameter parameter{};
    parameter.sample_rate = read(message, offset + 0x00);
    parameter.sample_count = read(message, offset + 0x04);
    parameter.mixes = read(message, offset + 0x08);
    parameter.sub_mixes = read(message, offset + 0x0C);
    parameter.voices = read(message, offset + 0x10);
    parameter.sinks = read(message, offset + 0x14);
    parameter.effects = read(message, offset + 0x18);
    parameter.performance_frames = read(message, offset + 0x1C);
    parameter.voice_drop_enabled = static_cast<std::uint8_t>(read(message, offset + 0x20, 1));
    parameter.reserved = static_cast<std::uint8_t>(read(message, offset + 0x21, 1));
    parameter.rendering_device = static_cast<std::uint8_t>(read(message, offset + 0x22, 1));
    parameter.execution_mode = static_cast<std::uint8_t>(read(message, offset + 0x23, 1));
    parameter.splitter_infos = read(message, offset + 0x24);
    parameter.splitter_destinations = static_cast<std::int32_t>(read(message, offset + 0x28));
    parameter.external_context_size = read(message, offset + 0x2C);
    parameter.revision = read(message, offset + 0x30);
    return parameter;
}

// A CMIF control request (type 5/7): the generated client asks for the pointer
// buffer size and optionally converts the session before its first command.
bool control_request(const IpcMessage& message, IpcReply& reply) {
    const auto type = read(message, 0, 2);
    if (type != 5 && type != 7) return false;
    if (read(message, 16) != sfci) {
        reply = {};
        return true;
    }
    const auto command = read(message, 24);
    if (command == 3) reply = control_reply(0, 2);
    else if (command == 0) reply = control_reply(1, 4);
    else reply = {};
    return true;
}
}  // namespace

HostAudio::HostAudio(OutputFactory factory) : factory_(std::move(factory)) {}

std::shared_ptr<audio::PcmOutput> HostAudio::open_output() {
    std::shared_ptr<audio::PcmOutput> output;
    if (factory_) output = factory_();
    if (!output) output = audio::make_muted_pcm_output();
    const std::lock_guard lock(mutex_);
    output->set_volume(volume_);
    std::erase_if(outputs_, [](const auto& weak) { return weak.expired(); });
    outputs_.push_back(output);
    return output;
}

void HostAudio::set_volume(float volume) {
    const std::lock_guard lock(mutex_);
    volume_ = volume;
    for (const auto& weak : outputs_)
        if (auto output = weak.lock()) output->set_volume(volume);
}

float HostAudio::volume() const noexcept {
    const std::lock_guard lock(mutex_);
    return volume_;
}

void PoolPins::set_mapper(horizon::GuestMemoryMapper mapper) {
    const std::lock_guard lock(mutex_);
    mapper_ = std::move(mapper);
}

bool PoolPins::pin(std::uint64_t address, std::uint64_t size) {
    const std::lock_guard lock(mutex_);
    if (!mapper_.map || !mapper_.map(mapper_.context, address, size)) return false;
    pins_.push_back({address, size, mapper_});
    return true;
}

void PoolPins::unpin(std::uint64_t address, std::uint64_t size) {
    const std::lock_guard lock(mutex_);
    for (auto it = pins_.rbegin(); it != pins_.rend(); ++it) {
        if (it->address != address || it->size != size) continue;
        if (it->mapper.unmap) it->mapper.unmap(it->mapper.context, address, size);
        pins_.erase(std::next(it).base());
        return;
    }
}

std::size_t PoolPins::pinned() const {
    const std::lock_guard lock(mutex_);
    return pins_.size();
}

RendererSession::RendererSession(const RendererParameter& parameter, Options options)
    : parameter_(parameter), event_(std::make_shared<horizon::Event>(false)),
      pins_(std::make_shared<PoolPins>()) {
    // Firmware AudioRenderSystem::SendCommandToDsp signals the renderer's
    // SystemEvent once per rendered frame (0x22a52c in the 20.x server code)
    // and AudioRenderSystem::Update (0x229b7c) never signals it. The render
    // thread reports each frame here; a successful update clears the event.
    output_ = options.host ? options.host->open_output() : audio::make_muted_pcm_output();
    audio::AudioRenderer::Options renderer_options;
    renderer_options.memory = pins_;
    renderer_options.output = output_;
    renderer_options.render_thread = options.render_thread;
    renderer_options.frame_rendered = [event = event_] { event->signal(); };
    renderer_options.update_completed = [event = event_] { event->reset(); };
    result_ = renderer_.initialize(parameter_, options.work_buffer_size, std::move(renderer_options));
}

RendererSession::~RendererSession() { renderer_.finalize(); }

void RendererSession::set_memory_mapper(horizon::GuestMemoryMapper mapper) noexcept {
    pins_->set_mapper(std::move(mapper));
}

bool RendererSession::started() const noexcept { return renderer_.active(); }

IpcReply RendererSession::unsupported(const char* detail) {
    const std::lock_guard lock(diagnostic_mutex_);
    diagnostic_ = detail;
    return {};
}

IpcReply RendererSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    {
        IpcReply control;
        if (control_request(message, control)) return control;
    }
    const auto located = horizon::locate_cmif(message);
    if (!located) return unsupported("Audio renderer request shape is unsupported");
    const auto cmif = *located;
    (void)sender_pid;  // the manager validated the connecting process
    const auto command = read(message, cmif + 8);
    switch (command) {
    case 0:  // GetSampleRate
        return value_reply(cmif, renderer_.sample_rate(), 4);
    case 1:  // GetSampleCount
        return value_reply(cmif, renderer_.sample_count(), 4);
    case 2:  // GetMixBufferCount
        return value_reply(cmif, renderer_.mix_buffer_count(), 4);
    case 3:  // GetState: zero while active, one while stopped
        return value_reply(cmif, renderer_.active() ? 0 : 1, 4);
    case 4:   // RequestUpdate
    case 10:  // RequestUpdateAuto
        return unsupported("Audio renderer update needs its update buffers");
    case 5:  // Start
        renderer_.start();
        return success_reply(cmif);
    case 6:  // Stop
        renderer_.stop();
        return success_reply(cmif);
    case 7: {  // QuerySystemEvent: a readable event the renderer signals per frame
        constexpr std::uint8_t manual_execution = 1;
        if (renderer_.execution_mode() == manual_execution)
            return result_reply(cmif, audio::result_not_supported);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, 16, sfco);
        reply.transferred[0] = event_;
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
        return reply;
    }
    case 8:  // SetRenderingTimeLimit (percent)
        renderer_.set_rendering_time_limit(read(message, cmif + 16));
        return success_reply(cmif);
    case 9:  // GetRenderingTimeLimit (percent)
        return value_reply(cmif, renderer_.rendering_time_limit(), 4);
    case 12: {  // SetVoiceDropParameter
        float value = 1.0f;
        const auto bits = read(message, cmif + 16);
        std::memcpy(&value, &bits, sizeof value);
        renderer_.set_voice_drop_parameter(value);
        return success_reply(cmif);
    }
    case 13: {  // GetVoiceDropParameter
        const float value = renderer_.voice_drop_parameter();
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof bits);
        return value_reply(cmif, bits, 4);
    }
    default:
        break;
    }
    // ExecuteAudioRendererRendering (11) drives manual execution, which this
    // renderer does not implement.
    return unsupported("Audio renderer command is not implemented");
}

IpcReply RendererSession::request_with_buffers(const IpcMessage& message,
                                                 std::uint64_t sender_pid,
                                                 const horizon::GuestBuffers& buffers) {
    {
        IpcReply control;
        if (control_request(message, control)) return control;
    }
    const auto located = horizon::locate_cmif(message);
    if (!located)
        return unsupported("Audio renderer request shape is unsupported");
    const auto cmif = *located;
    const auto command = read(message, cmif + 8);
    if (command != 4 && command != 10) return request(message, sender_pid);
    // RequestUpdate(out, performance, in): the send buffer carries the guest's
    // update data, the first receive buffer receives the update status and the
    // second, when the guest supplies one, the performance metrics. The
    // dispatcher lists send buffers before receive buffers.
    if (buffers.size() < 2) return unsupported("Audio renderer update needs its update buffers");
    std::vector<std::byte> input(buffers[0].size);
    if (!buffers.read(0, input)) return unsupported("Audio renderer update input is not readable");
    std::vector<std::byte> output(buffers[1].size);
    std::vector<std::byte> performance(buffers.size() > 2 ? buffers[2].size : 0);
    const auto result = renderer_.update(input, performance, output);
    // The status buffers are written in place on the guest, so they hold what
    // the update produced even when it fails part way.
    if (!buffers.write(1, output)) return unsupported("Audio renderer update buffer is not writable");
    if (!performance.empty() && !buffers.write(2, performance))
        return unsupported("Audio renderer performance buffer is not writable");
    // The reply repeats the request's HIPC header and buffer descriptors so the
    // client finds the CMIF out header after the same descriptor area it used
    // for the request.
    IpcReply reply = success_reply(cmif);
    for (std::size_t index = 0; index < cmif; ++index) reply.bytes[index] = message[index];
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, cmif, sfco);
    write(reply.bytes, cmif + 8, result);
    return reply;
}

namespace {
constexpr std::size_t audio_device_name_size = 0x100;

IpcReply descriptor_reply(const IpcMessage& message, std::size_t cmif,
                          std::span<const std::byte> output = {}) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = cmif + 16 + output.size();
    for (std::size_t index = 0; index < cmif; ++index) reply.bytes[index] = message[index];
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, cmif, sfco);
    if (!output.empty()) std::memcpy(reply.bytes.data() + cmif + 16, output.data(), output.size());
    return reply;
}
}  // namespace

AudioDeviceSession::AudioDeviceSession(std::uint64_t aruid, std::uint32_t revision,
                                       std::shared_ptr<HostAudio> host)
    : aruid_(aruid), revision_(revision), host_(std::move(host)) {}

float AudioDeviceSession::volume() const noexcept {
    const std::lock_guard lock(mutex_);
    return volume_;
}

IpcReply AudioDeviceSession::unsupported(const char* detail) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = detail;
    return {};
}

IpcReply AudioDeviceSession::request_with_buffers(const IpcMessage& message,
                                                   std::uint64_t sender_pid,
                                                   const horizon::GuestBuffers& buffers) {
    {
        IpcReply control;
        if (control_request(message, control)) return control;
    }
    const auto located = horizon::locate_cmif(message);
    if (!located)
        return unsupported("Audio device request shape is unsupported");
    const auto cmif = *located;
    (void)sender_pid;  // the manager validated the connecting process
    const auto command = read(message, cmif + 8);
    std::array<std::byte, audio_device_name_size> name{};
    std::memcpy(name.data(), "AudioBuiltInSpeakerOutput", 25);
    switch (command) {
    case 0:   // ListAudioDeviceName
    case 6:   // ListAudioDeviceNameAuto
    case 14: {  // ListAudioOutputDeviceName
        if (buffers.empty()) return unsupported("Audio device list needs a name buffer");
        if (!buffers.write(0, name)) return unsupported("Audio device name buffer is not writable");
        std::array<std::byte, 4> count{};
        const std::uint32_t one = 1;
        std::memcpy(count.data(), &one, sizeof one);
        return descriptor_reply(message, cmif, count);
    }
    case 1:   // SetAudioDeviceOutputVolume
    case 7: {  // SetAudioDeviceOutputVolumeAuto
        float volume = 1.0f;
        std::memcpy(&volume, message.data() + cmif + 16, sizeof volume);
        {
            const std::lock_guard lock(mutex_);
            volume_ = volume;
        }
        // The device volume scales the host output of every renderer.
        if (host_) host_->set_volume(volume);
        // The request carries the device-name buffer; the reply repeats its
        // descriptors so the client finds the CMIF header after them.
        return descriptor_reply(message, cmif);
    }
    case 2:   // GetAudioDeviceOutputVolume
    case 8: {  // GetAudioDeviceOutputVolumeAuto
        const std::lock_guard lock(mutex_);
        std::array<std::byte, 4> value{};
        std::memcpy(value.data(), &volume_, sizeof volume_);
        return descriptor_reply(message, cmif, value);
    }
    case 3:   // GetActiveAudioDeviceName
    case 10:  // GetActiveAudioDeviceNameAuto
    case 13:  // GetActiveAudioOutputDeviceName
        if (buffers.empty()) return unsupported("Audio device name needs a buffer");
        if (!buffers.write(0, name)) return unsupported("Audio device name buffer is not writable");
        return descriptor_reply(message, cmif);
    default:
        break;
    }
    return request(message, sender_pid);
}

IpcReply AudioDeviceSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    {
        IpcReply control;
        if (control_request(message, control)) return control;
    }
    const auto located = horizon::locate_cmif(message);
    if (!located)
        return unsupported("Audio device request shape is unsupported");
    const auto cmif = *located;
    (void)sender_pid;
    const auto command = read(message, cmif + 8);
    switch (command) {
    case 4:   // QueryAudioDeviceSystemEvent
    case 11:  // QueryAudioDeviceInputEvent
    case 12: {  // QueryAudioDeviceOutputEvent
        std::shared_ptr<horizon::Event> event;
        {
            const std::lock_guard lock(mutex_);
            if (!event_) event_ = std::make_shared<horizon::Event>(false);
            event = event_;
        }
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, 16, sfco);
        reply.transferred[0] = std::move(event);
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
        return reply;
    }
    case 5:  // GetActiveChannelCount
        return value_reply(cmif, 2, 4);
    case 15:    // AcquireAudioInputDeviceNotification(u64)
    case 17: {  // AcquireAudioOutputDeviceNotification(u64)
        // Fires when the set of devices changes, which it never does here.
        std::shared_ptr<horizon::Event> event;
        {
            const std::lock_guard lock(mutex_);
            if (!notification_) notification_ = std::make_shared<horizon::Event>(false);
            event = notification_;
        }
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, 16, sfco);
        reply.transferred[0] = std::move(event);
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
        return reply;
    }
    case 16:  // ReleaseAudioInputDeviceNotification(u64)
    case 18:  // ReleaseAudioOutputDeviceNotification(u64)
        return success_reply(cmif);
    case 19:  // SetAudioDeviceOutputVolumeAutoTuneEnabled
        return success_reply(cmif);
    case 20:  // IsAudioDeviceOutputVolumeAutoTuneEnabled
        return value_reply(cmif, 0, 4);
    default:
        break;
    }
    return unsupported("Audio device command is not implemented");
}

ManagerSession::ManagerSession(std::uint64_t process_id, OutputFactory output_factory)
    : process_id_(process_id), host_(std::make_shared<HostAudio>(std::move(output_factory))) {}

IpcReply ManagerSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply ManagerSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    {
        IpcReply control;
        if (control_request(message, control)) return control;
    }
    diagnostic_ = nullptr;
    if (sender_pid != process_id_)
        return unsupported("Audio renderer manager session belongs to another process");
    const auto located = horizon::locate_cmif(message);
    if (!located)
        return unsupported("Audio renderer manager request shape is unsupported");
    const auto cmif = *located;
    const auto command = read(message, cmif + 8);
    switch (command) {
    case 1: {  // GetWorkBufferSize(parameter)
        const auto parameter = parameter_from(message, cmif + 16);
        std::uint64_t work_buffer_size = 0;
        const auto result = audio::AudioRenderer::work_buffer_size(parameter, work_buffer_size);
        if (result != 0) return result_reply(cmif, result);
        return value_reply(cmif, work_buffer_size, 8);
    }
    case 2: {  // GetAudioDeviceService(aruid): one built-in output device
        const auto aruid = read64(message, cmif + 16);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, 16, sfco);
        reply.transferred[0] = std::make_shared<AudioDeviceSession>(aruid, 0, host_);
        reply.transferred_count = 1;
        return reply;
    }
    case 4: {  // GetAudioDeviceServiceWithRevisionInfo(revision, aruid)
        // Raw input {u32 revision, u32 padding, u64 aruid}: the u64 is 8-byte aligned.
        const auto revision = read(message, cmif + 16);
        const auto aruid = read64(message, cmif + 24);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, 16, sfco);
        reply.transferred[0] = std::make_shared<AudioDeviceSession>(aruid, revision, host_);
        reply.transferred_count = 1;
        return reply;
    }
    case 0: {  // OpenAudioRenderer(parameter, transfer memory size, aruid; transfer memory, process)
        // The renderer keeps its work buffer in host memory; the guest's
        // transfer memory only declares the size it was given.
        const auto parameter = parameter_from(message, cmif + 16);
        const auto work_buffer_size = read64(message, cmif + 16 + 0x38);
        RendererSession::Options options;
        options.work_buffer_size = work_buffer_size;
        options.host = host_;
        options.render_thread = render_threads_;
        auto renderer = std::make_shared<RendererSession>(parameter, std::move(options));
        if (renderer->result() != 0) return result_reply(cmif, renderer->result());
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, 16, sfco);
        reply.transferred[0] = std::move(renderer);
        reply.transferred_count = 1;
        return reply;
    }
    default:
        break;
    }
    return unsupported("Audio renderer manager command is not implemented");
}

}  // namespace armigo::services::audren
