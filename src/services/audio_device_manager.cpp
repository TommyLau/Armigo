// ============================================================================
// Armigo Services - Switch 2 Audio Device Manager
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/services/audio_device_manager.h>

#include <algorithm>
#include <cstring>
#include <span>
#include <string_view>

namespace armigo::services::auddmg {
namespace {
using namespace horizon;

using namespace horizon::cmif;
constexpr std::string_view device_name = "AudioVirtualOutput0";
constexpr std::string_view physical_device_name = "AudioTvOutput";
// Sample types of a stream: 16-bit integer and 32-bit float PCM.
constexpr std::uint32_t sample_pcm_int16 = 2;
constexpr std::uint32_t sample_pcm_float = 5;



void put(std::span<std::byte> block, std::size_t offset, std::uint64_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i) block[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

void put_name(std::span<std::byte> block, std::size_t offset, std::string_view name) {
    std::memcpy(block.data() + offset, name.data(), std::min(name.size(), device_name_size - 1));
}

// The request's guest buffers in descriptor order (pointers, sends, receives,
// then receive-list entries; empty ones are dropped): a command's name input
// is the first, its output the last.
std::optional<std::size_t> input_buffer(const GuestBuffers* buffers) {
    if (!buffers || buffers->empty() || !(*buffers)[0].input) return std::nullopt;
    return 0;
}
std::optional<std::size_t> output_buffer(const GuestBuffers* buffers) {
    if (!buffers || buffers->empty() || !(*buffers)[buffers->size() - 1].output) return std::nullopt;
    return buffers->size() - 1;
}

// A CMIF reply after the request's HIPC header and descriptors, so the client
// finds the out header where its proxy expects it, with a result and words.
IpcReply reply_at(const IpcMessage& message, std::size_t cmif, std::uint32_t result,
                  std::span<const std::uint32_t> words = {}) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = cmif + 16 + 4 * words.size();
    for (std::size_t i = 0; i < cmif; ++i) reply.bytes[i] = message[i];
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, cmif, sfco);
    write(reply.bytes, cmif + 8, result);
    for (std::size_t i = 0; i < words.size(); ++i) write(reply.bytes, cmif + 16 + 4 * i, words[i]);
    return reply;
}

// A reply that hands out one object or handle (HIPC header, one handle, CMIF).
IpcReply handle_reply(std::shared_ptr<KernelObject> object, bool copy) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 32;
    write(reply.bytes, 4, (reply.size - 8) / 4);
    write(reply.bytes, 16, sfco);
    reply.transferred[0] = std::move(object);
    reply.transferred_count = 1;
    reply.transferred_as_copy = copy;
    return reply;
}
} // namespace

VirtualDeviceManagerSession::VirtualDeviceManagerSession(std::uint64_t process_id) : process_id_(process_id) {}

IpcReply VirtualDeviceManagerSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

const char* VirtualDeviceManagerSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

IpcReply VirtualDeviceManagerSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return dispatch(message, sender_pid, nullptr);
}

IpcReply VirtualDeviceManagerSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                                           const GuestBuffers& buffers) {
    return dispatch(message, sender_pid, &buffers);
}

IpcReply VirtualDeviceManagerSession::dispatch(const IpcMessage& message, std::uint64_t sender_pid,
                                               const GuestBuffers* buffers) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Audio virtual device manager belongs to another process");
    const auto located = locate_cmif(message);
    if (!located) return unsupported("Audio virtual device manager request shape is unsupported");
    const auto cmif = *located;
    std::array<std::byte, device_name_size> name{};
    put_name(name, 0, device_name);
    switch (read(message, cmif + 8)) {
    case 0: {  // ListVirtualDevices(u8 flag, u32 functionality): the count, the names in the out buffer
        // Functionality 1 lists output devices and 2 input devices. The host
        // has no microphone, so an input list is empty and the title skips
        // its input streams.
        const auto functionality = read(message, cmif + 20);
        if (functionality == 2) {
            const std::array<std::uint32_t, 1> none{0};
            return reply_at(message, cmif, 0, none);
        }
        if (functionality != 1) return unsupported("Audio device list functionality is not output or input");
        const auto out = output_buffer(buffers);
        const std::uint32_t count = out && (*buffers)[*out].size >= device_name_size ? 1 : 0;
        if (count && !buffers->write(*out, name)) return unsupported("Audio device list buffer is not writable");
        const std::array words{count};
        return reply_at(message, cmif, 0, words);
    }
    case 100: {  // GetDefaultVirtualOutputDevice: its name
        const auto out = output_buffer(buffers);
        if (!out || (*buffers)[*out].size < device_name_size || !buffers->write(*out, name))
            return unsupported("Default audio device name needs a writable name buffer");
        return reply_at(message, cmif, 0);
    }
    case 300: {  // IsVirtualOutputDeviceStreamFormatSupported(type, channels, rate, functionality, name)
        const auto type = read(message, cmif + 16), channels = read(message, cmif + 20);
        const auto rate = read(message, cmif + 24), functionality = read(message, cmif + 28);
        const bool supported = (type == sample_pcm_int16 || type == sample_pcm_float) && channels == 2 &&
                               rate == 48000 && functionality == 1;
        return reply_at(message, cmif, supported ? 0 : result_not_supported);
    }
    case 401: {  // GetVirtualOutputDeviceProperties(name) -> property block
        const auto in = input_buffer(buffers);
        const auto out = output_buffer(buffers);
        if (!in || !out || *in == *out || (*buffers)[*out].size < device_properties_size)
            return unsupported("Audio device properties need a name buffer and a property buffer");
        std::array<std::byte, device_name_size> asked{};
        std::span<std::byte> requested(asked.data(), std::min<std::size_t>((*buffers)[*in].size, device_name_size));
        if (!buffers->read(*in, requested)) return unsupported("Audio device name buffer is not readable");
        // Name (0x000), type 3 (0x100), 2 (0x104), channel count (0x108),
        // sample rate (0x10c), physical output name (0x110), latency in ns
        // (0x218) and device id (0x220), which the SDK's output streams pass
        // on and which must not be zero.
        std::array<std::byte, device_properties_size> block{};
        std::memcpy(block.data(), asked.data(), device_name_size - 1);
        put(block, 0x100, 3, 4);
        put(block, 0x104, 2, 4);
        put(block, 0x108, 2, 4);
        put(block, 0x10c, 48000, 4);
        put_name(block, 0x110, physical_device_name);
        put(block, 0x218, 10'000'000, 8);
        put(block, 0x220, 1, 4);
        if (!buffers->write(*out, block)) return unsupported("Audio device property buffer is not writable");
        return reply_at(message, cmif, 0);
    }
    case 500: {  // AcquireVirtualOutputDeviceStateEvent: the device's state never changes here
        if (!state_event_) state_event_ = std::make_shared<Event>(false);
        return handle_reply(state_event_, true);
    }
    case 700:
        return unsupported("Attaching an audio stream to a virtual device needs the sound mixer service");
    case 800:
        return unsupported("Detaching an audio stream from a virtual device needs the sound mixer service");
    default:
        return unsupported("Audio virtual device manager command is not implemented");
    }
}

DeviceManagerSession::DeviceManagerSession(std::uint64_t process_id) : process_id_(process_id) {}

IpcReply DeviceManagerSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

const char* DeviceManagerSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

std::optional<std::array<std::byte, 16>> DeviceManagerSession::open_argument() const {
    const std::lock_guard lock(mutex_);
    return argument_;
}

IpcReply DeviceManagerSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Audio device manager session belongs to another process");
    const auto located = locate_cmif(message);
    if (!located) return unsupported("Audio device manager request shape is unsupported");
    const auto cmif = *located;
    if (read(message, cmif + 8) != 0) return unsupported("Audio device manager command is not implemented");
    // Command 0 takes 16 bytes and the process id, and hands out the virtual
    // output device manager.
    std::array<std::byte, 16> argument{};
    std::copy_n(message.begin() + static_cast<std::ptrdiff_t>(cmif + 16), argument.size(), argument.begin());
    argument_ = argument;
    return handle_reply(std::make_shared<VirtualDeviceManagerSession>(process_id_), false);
}

} // namespace armigo::services::auddmg
