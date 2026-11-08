// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/csrng.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <span>

namespace armigo::services::csrng {
namespace {
using namespace horizon;
constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
std::uint32_t read(const IpcMessage& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i != 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(bytes[offset + i])) << (8 * i);
    return value;
}
void write(IpcMessage& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
// A non-domain reply with `output_words` of output after the CMIF header.
IpcReply reply_with(unsigned output_words) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 8 + 4 * (8 + output_words);
    write(reply.bytes, 4, 8 + output_words);
    write(reply.bytes, 16, sfco);
    return reply;
}
} // namespace

Session::Session(std::uint64_t process_id) : process_id_(process_id) {}

IpcReply Session::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    return dispatch(message, sender_pid, nullptr);
}

IpcReply Session::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                       const GuestBuffers& buffers) {
    const std::lock_guard lock(mutex_);
    return dispatch(message, sender_pid, &buffers);
}

IpcReply Session::dispatch(const IpcMessage& message, std::uint64_t sender_pid, const GuestBuffers* buffers) {
    diagnostic_ = "Random-number method or packet is unsupported";
    if (sender_pid != process_id_) {
        diagnostic_ = "Random-number session belongs to a different process";
        return {};
    }
    const auto type = read(message, 0) & 0xffffu;
    if (type == 5 || type == 7) {
        // QueryPointerBufferSize only: this endpoint has no pointer buffer.
        if (read(message, 16) != sfci || read(message, 24) != 3) return {};
        auto reply = reply_with(1);
        write(reply.bytes, 32, 0);
        diagnostic_ = nullptr;
        return reply;
    }
    if (type != 4 && type != 6) return {};
    const auto cmif = locate_cmif(message);
    if (!cmif || read(message, *cmif) != sfci || read(message, *cmif + 8) != 0) return {};
    // GenerateRandomBytes: the whole output buffer, in chunks of host entropy.
    if (!buffers || buffers->size() != 1 || !(*buffers)[0].output) {
        diagnostic_ = "GenerateRandomBytes needs one output buffer";
        return {};
    }
    std::array<std::byte, 256> chunk{};
    const std::size_t size = (*buffers)[0].size;
    for (std::size_t done = 0; done < size; done += chunk.size()) {
        const auto count = std::min(chunk.size(), size - done);
        arc4random_buf(chunk.data(), count);
        if (!buffers->write_at(0, done, std::span(chunk).first(count))) {
            diagnostic_ = "GenerateRandomBytes buffer is not writable";
            return {};
        }
    }
    diagnostic_ = nullptr;
    return reply_with(0);
}

const char* Session::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::csrng
