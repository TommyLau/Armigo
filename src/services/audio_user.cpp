// ============================================================================
// Armigo Services - Switch 2 Audio User Client
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/audio_user.h>
#include <armigo/services/audio_out.h>

#include <memory>

namespace armigo::services::audio_user {
namespace {
using namespace horizon;

constexpr std::uint32_t sfco = 0x4f434653;

std::uint32_t read(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(message[offset + i])) << (8 * i);
    return value;
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) message[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
} // namespace

UserClientSession::UserClientSession(std::uint64_t process_id, audren::OutputFactory output_factory)
    : process_id_(process_id), output_factory_(std::move(output_factory)) {}

IpcReply UserClientSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

const char* UserClientSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

std::uint32_t UserClientSession::manager_argument(unsigned command) const {
    const std::lock_guard lock(mutex_);
    return command == 1 || command == 2 ? arguments_[command - 1] : 0;
}

IpcReply UserClientSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Audio user client session belongs to another process");
    const auto located = locate_cmif(message);
    if (!located) return unsupported("Audio user client request shape is unsupported");
    const auto cmif = *located;
    const auto command = read(message, cmif + 8);
    if (command != 1 && command != 2) return unsupported("Audio user client command is not implemented");
    // One word of input, kept as given.
    arguments_[command - 1] = read(message, cmif + 16);
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 32; // HIPC header8 + one move-handle entry8 + CMIF header16
    write(reply.bytes, 4, (reply.size - 8) / 4);
    write(reply.bytes, 16, sfco);
    if (command == 1)
        reply.transferred[0] = std::make_shared<audout::ManagerSession>(process_id_, output_factory_);
    else
        reply.transferred[0] = std::make_shared<audren::ManagerSession>(process_id_, output_factory_);
    reply.transferred_count = 1;
    return reply;
}

} // namespace armigo::services::audio_user
