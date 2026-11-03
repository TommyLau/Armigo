// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/audio_out.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>

namespace armigo::services::audout {
namespace {
using namespace horizon;
constexpr std::uint32_t sample_rate = 48000;
constexpr std::uint32_t pcm_int16 = 2;
// Most PCM bytes one Append may carry: five seconds of six-channel audio.
constexpr std::uint64_t max_buffer_bytes = std::uint64_t{sample_rate} * 6 * 2 * 5;
constexpr std::uint32_t result_bad_state = 0x499;
constexpr std::uint32_t result_queue_full = 0x1099;
constexpr std::uint32_t result_bad_name = 0x299;
constexpr std::uint32_t result_bad_rate = 0x699;
constexpr std::uint32_t result_bad_channels = 0x1499;

std::uint32_t read(const IpcMessage& bytes, std::size_t offset, unsigned width = 4) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i != width; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(bytes[offset + i])) << (8 * i);
    return value;
}
std::uint64_t read64(const IpcMessage& bytes, std::size_t offset) {
    return read(bytes, offset) | std::uint64_t{read(bytes, offset + 4)} << 32;
}
void write(IpcMessage& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
// A non-domain reply with `output_words` of output after the CMIF header.
IpcReply reply_with(unsigned output_words, std::uint32_t result = 0) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 8 + 4 * (8 + output_words);
    write(reply.bytes, 4, 8 + output_words);
    write(reply.bytes, 16, 0x4f434653);
    write(reply.bytes, 24, result);
    return reply;
}
// QueryPointerBufferSize answers no pointer buffer, so the SDK's automatic
// buffers travel as mapped descriptors.
std::optional<IpcReply> control(const IpcMessage& message) {
    const auto type = read(message, 0) & 0xffffu;
    if (type != 5 && type != 7) return std::nullopt;
    if (read(message, 16) != 0x49434653 || read(message, 24) != 3) return IpcReply{};
    auto reply = reply_with(1);
    write(reply.bytes, 32, 0);
    return reply;
}
// The CMIF header of a non-domain method, and its command.
std::optional<std::size_t> method_header(const IpcMessage& message) {
    const auto type = read(message, 0) & 0xffffu;
    if (type != 4 && type != 6) return std::nullopt;
    const auto cmif = locate_cmif(message);
    if (!cmif || *cmif + 16 > message.size() || read(message, *cmif) != 0x49434653) return std::nullopt;
    return cmif;
}
} // namespace

AudioOutSession::AudioOutSession(std::uint64_t process_id, unsigned channels,
                                 std::shared_ptr<audio::PcmOutput> output)
    : process_id_(process_id), channels_(channels), output_(std::move(output)),
      buffer_event_(std::make_shared<Event>(false)),
      worker_([this](std::stop_token stop) { play(stop); }) {}

AudioOutSession::~AudioOutSession() {
    worker_.request_stop();
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (output_) output_->set_active(false);
}

void AudioOutSession::play(std::stop_token stop) {
    using clock = std::chrono::steady_clock;
    // Buffers play back to back on an absolute schedule, as the device
    // consumes them: a wait that wakes late does not delay the next buffer.
    // Only a gap with nothing to play (stopped or drained) restarts the
    // schedule from the moment the next buffer can play.
    auto deadline = clock::now();
    bool gap = true;
    std::unique_lock lock(mutex_);
    while (!stop.stop_requested()) {
        if (!started_ || queued_.empty()) gap = true;
        if (!wake_.wait(lock, stop, [this] { return started_ && !queued_.empty(); })) break;
        // The front buffer is playing; it stays queued until it is released.
        const auto tag = queued_.front().tag;
        const auto frames = queued_.front().samples.size() / channels_;
        if (output_) output_->write(queued_.front().samples.data(), frames, channels_);
        if (gap) deadline = std::max(deadline, clock::now());
        gap = false;
        deadline += std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(static_cast<double>(frames) / sample_rate));
        wake_.wait_until(lock, stop, deadline, [] { return false; });
        if (stop.stop_requested()) break;
        if (!queued_.empty() && queued_.front().tag == tag) {
            queued_.pop_front();
            released_.push_back(tag);
            buffer_event_->signal();
        }
    }
}

horizon::IpcReply AudioOutSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    if (sender_pid != process_id_) {
        diagnostic_ = "AudioOut session belongs to a different process";
        return {};
    }
    return method(message, nullptr);
}

horizon::IpcReply AudioOutSession::request_with_buffers(const horizon::IpcMessage& message,
                                                        std::uint64_t sender_pid,
                                                        const horizon::GuestBuffers& buffers) {
    const std::lock_guard lock(mutex_);
    if (sender_pid != process_id_) {
        diagnostic_ = "AudioOut session belongs to a different process";
        return {};
    }
    return method(message, buffers.empty() ? nullptr : &buffers);
}

horizon::IpcReply AudioOutSession::method(const horizon::IpcMessage& message, const horizon::GuestBuffers* buffers) {
    diagnostic_ = "AudioOut method or packet is unsupported";
    if (auto reply = control(message)) {
        if (reply->disposition == IpcDisposition::handled) diagnostic_ = nullptr;
        return *reply;
    }
    const auto cmif = method_header(message);
    if (!cmif) return {};
    const auto command = read(message, *cmif + 8);
    try {
        switch (command) {
        case 0: {  // GetAudioOutState: 0 started, 1 stopped
            auto reply = reply_with(1);
            write(reply.bytes, 32, started_ ? 0 : 1);
            diagnostic_ = nullptr;
            return reply;
        }
        case 1:    // Start
        case 2: {  // Stop
            const bool start = command == 1;
            if (start && started_) {
                diagnostic_ = nullptr;
                return reply_with(0, result_bad_state);
            }
            if (output_ && !output_->set_active(start) && start) {
                diagnostic_ = "AudioOut host output could not start";
                return {};
            }
            started_ = start;
            wake_.notify_all();
            diagnostic_ = nullptr;
            return reply_with(0);
        }
        case 4: {  // RegisterBufferEvent: one readable copy of the buffer event
            auto reply = reply_with(0);
            reply.size = 32;
            reply.transferred[0] = buffer_event_;
            reply.transferred_count = 1;
            reply.transferred_as_copy = true;
            diagnostic_ = nullptr;
            return reply;
        }
        case 6: {  // ContainsAudioOutBuffer(tag)
            const auto tag = read64(message, *cmif + 16);
            const bool found = std::any_of(queued_.begin(), queued_.end(),
                                           [&](const Entry& entry) { return entry.tag == tag; });
            auto reply = reply_with(1);
            write(reply.bytes, 32, found ? 1 : 0);
            diagnostic_ = nullptr;
            return reply;
        }
        case 7: {  // AppendAudioOutBufferAuto(buffer, tag)
            if (!buffers || (*buffers)[0].size < 40) {
                diagnostic_ = "AudioOut Append needs its 40-byte buffer record";
                return {};
            }
            std::array<std::byte, 40> record{};
            if (!buffers->read(0, record)) {
                diagnostic_ = "AudioOut Append record is not readable";
                return {};
            }
            std::uint64_t address = 0, bytes = 0;
            std::memcpy(&address, record.data() + 8, 8);
            std::memcpy(&bytes, record.data() + 24, 8);
            if (bytes == 0) {
                diagnostic_ = nullptr;
                return reply_with(0, result_bad_state);
            }
            // Released buffers the guest has not collected still count.
            if (queued_.size() + released_.size() >= max_buffers) {
                diagnostic_ = nullptr;
                return reply_with(0, result_queue_full);
            }
            if (bytes > max_buffer_bytes || bytes % (2 * channels_)) {
                diagnostic_ = "AudioOut Append size is not whole frames within the host limit";
                return {};
            }
            Entry entry{read64(message, *cmif + 16), std::vector<std::int16_t>(bytes / 2)};
            if (!buffers->read_memory(address, std::as_writable_bytes(std::span(entry.samples)))) {
                diagnostic_ = "AudioOut Append samples are not readable";
                return {};
            }
            queued_.push_back(std::move(entry));
            wake_.notify_all();
            diagnostic_ = nullptr;
            return reply_with(0);
        }
        case 8: {  // GetReleasedAudioOutBuffersAuto: tags out, count
            if (!buffers || (*buffers)[0].size < 8) {
                diagnostic_ = "AudioOut GetReleased needs its output buffer";
                return {};
            }
            const std::size_t count = std::min<std::size_t>(released_.size(), (*buffers)[0].size / 8);
            for (std::size_t i = 0; i != std::max<std::size_t>(count, 1); ++i) {
                // An empty result still writes a zero tag.
                const std::uint64_t tag = count ? released_[i] : 0;
                if (!buffers->write_at(0, i * 8, std::as_bytes(std::span(&tag, 1)))) {
                    diagnostic_ = "AudioOut GetReleased output is not writable";
                    return {};
                }
            }
            released_.erase(released_.begin(), released_.begin() + static_cast<std::ptrdiff_t>(count));
            if (released_.empty()) buffer_event_->reset();
            auto reply = reply_with(1);
            write(reply.bytes, 32, static_cast<std::uint32_t>(count));
            diagnostic_ = nullptr;
            return reply;
        }
        case 9: {  // GetAudioOutBufferCount
            auto reply = reply_with(1);
            write(reply.bytes, 32, static_cast<std::uint32_t>(queued_.size()));
            diagnostic_ = nullptr;
            return reply;
        }
        default:
            return {};
        }
    } catch (const std::bad_alloc&) {
        diagnostic_ = "AudioOut allocation failed";
    } catch (const std::length_error&) {
        diagnostic_ = "AudioOut storage exhausted";
    }
    return {};
}

const char* AudioOutSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

ManagerSession::ManagerSession(std::uint64_t process_id, OutputFactory output_factory)
    : process_id_(process_id), output_factory_(std::move(output_factory)) {}

horizon::IpcReply ManagerSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    if (sender_pid != process_id_) {
        diagnostic_ = "AudioOut manager belongs to a different process";
        return {};
    }
    return method(message, nullptr);
}

horizon::IpcReply ManagerSession::request_with_buffers(const horizon::IpcMessage& message,
                                                       std::uint64_t sender_pid,
                                                       const horizon::GuestBuffers& buffers) {
    const std::lock_guard lock(mutex_);
    if (sender_pid != process_id_) {
        diagnostic_ = "AudioOut manager belongs to a different process";
        return {};
    }
    return method(message, buffers.empty() ? nullptr : &buffers);
}

horizon::IpcReply ManagerSession::method(const horizon::IpcMessage& message, const horizon::GuestBuffers* buffers) {
    diagnostic_ = "AudioOut manager method or packet is unsupported";
    if (auto reply = control(message)) {
        if (reply->disposition == IpcDisposition::handled) diagnostic_ = nullptr;
        return *reply;
    }
    const auto cmif = method_header(message);
    if (!cmif || read(message, *cmif + 8) != 1) return {};
    // OpenAudioOut(rate, channels, identity; name in, normalized name out).
    if (!buffers || buffers->size() < 2) {
        diagnostic_ = "AudioOut Open needs its name and normalized-name buffers";
        return {};
    }
    std::array<char, 256> name{};
    const auto name_size = std::min<std::size_t>((*buffers)[0].size, name.size());
    if (!buffers->read(0, std::as_writable_bytes(std::span(name.data(), name_size)))) {
        diagnostic_ = "AudioOut Open name is not readable";
        return {};
    }
    name.back() = '\0';
    const std::string_view requested(name.data());
    const auto rate = read(message, *cmif + 16);
    const auto channels = read(message, *cmif + 20, 2);
    diagnostic_ = nullptr;
    if (!requested.empty() && requested != "DeviceOut") return reply_with(0, result_bad_name);
    if (rate != 0 && rate != sample_rate) return reply_with(0, result_bad_rate);
    if (channels != 0 && channels != 2 && channels != 6) return reply_with(0, result_bad_channels);
    const unsigned opened = channels == 6 ? 6 : 2;
    try {
        static constexpr char normalized[] = "DeviceOut";
        if (!buffers->write_at(1, 0, std::as_bytes(std::span(normalized)))) {
            diagnostic_ = "AudioOut Open normalized name is not writable";
            return {};
        }
        auto session = std::make_shared<AudioOutSession>(process_id_, opened,
                                                         output_factory_ ? output_factory_() : nullptr);
        IpcReply reply{IpcDisposition::handled};
        reply.size = 48;
        write(reply.bytes, 16, 0x4f434653);
        write(reply.bytes, 32, sample_rate);
        write(reply.bytes, 36, opened);
        write(reply.bytes, 40, pcm_int16);
        write(reply.bytes, 44, 1);  // stopped
        reply.transferred[0] = std::move(session);
        reply.transferred_count = 1;
        return reply;
    } catch (const std::bad_alloc&) {
        diagnostic_ = "AudioOut allocation failed";
    } catch (const std::system_error&) {
        diagnostic_ = "AudioOut worker could not start";
    }
    return {};
}

const char* ManagerSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::audout
