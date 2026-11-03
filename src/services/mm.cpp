// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/mm.h>

#include <algorithm>
#include <new>

namespace armigo::services::mm {
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

horizon::IpcReply Session::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = "Multimedia clock method or packet is unsupported";
    if (sender_pid != process_id_) {
        diagnostic_ = "Multimedia clock session belongs to a different process";
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
    if (!cmif || read(message, *cmif) != sfci) return {};
    const auto command = read(message, *cmif + 8);
    // Raw words are 4 of alignment padding, the 4-word CMIF header, then the inputs.
    const auto words = read(message, 4) & 0x3ffu;
    const auto inputs = words >= 8 ? words - 8 : 0;
    const auto input = [&](unsigned index) { return read(message, *cmif + 16 + 4 * index); };
    const auto find = [&](std::uint32_t id) {
        return std::find_if(requests_.begin(), requests_.end(), [&](const Request& r) { return r.id == id; });
    };
    try {
        switch (command) {
        case 4: {  // Initialize(module, priority, event clear mode) -> request id
            if (inputs < 3) return {};
            requests_.push_back({input(0), next_id_, 0, -1, input(2) == 1});
            auto reply = reply_with(1);
            write(reply.bytes, 32, next_id_++);
            diagnostic_ = nullptr;
            return reply;
        }
        case 5: {  // Finalize(id)
            if (inputs < 1) return {};
            if (const auto found = find(input(0)); found != requests_.end()) requests_.erase(found);
            diagnostic_ = nullptr;
            return reply_with(0);
        }
        case 6: {  // SetAndWait(id, minimum, maximum)
            if (inputs < 3) return {};
            if (const auto found = find(input(0)); found != requests_.end()) {
                found->minimum = input(1);
                found->maximum = static_cast<std::int32_t>(input(2));
            }
            diagnostic_ = nullptr;
            return reply_with(0);
        }
        case 7: {  // Get(id) -> the request's minimum, 0 for an unknown id
            if (inputs < 1) return {};
            const auto found = find(input(0));
            auto reply = reply_with(1);
            write(reply.bytes, 32, found == requests_.end() ? 0 : found->minimum);
            diagnostic_ = nullptr;
            return reply;
        }
        default:
            return {};
        }
    } catch (const std::bad_alloc&) {
        diagnostic_ = "Multimedia clock request allocation failed";
        return {};
    }
}

const char* Session::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::mm
