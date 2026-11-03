// ============================================================================
// Armigo Services - Initial Error Context Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/context_writer.h>

namespace armigo::services::err {
namespace {
std::uint32_t read(const horizon::IpcMessage& bytes, unsigned offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i != 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(bytes[offset + i])) << (8 * i);
    return value;
}
void write(horizon::IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
} // namespace

horizon::IpcReply ContextWriterSession::request(const horizon::IpcMessage& message,
                                               std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = "Error-context service method or packet is unsupported";
    if (sender_pid != process_id_) {
        diagnostic_ = "Error-context session belongs to a different process";
        return {};
    }
    const auto type = read(message, 0);
    const auto words = read(message, 4);
    // Exact type/header checks exclude buffer descriptors, PID and handles.
    if ((type != 5 && type != 7) || words < 8 || words > (message.size() - 8) / 4 ||
        read(message, 16) != 0x49434653)
        return {};
    const auto command = read(message, 24);
    if (command != 3 && (command != 0 || domain_)) return {};

    horizon::IpcReply reply{horizon::IpcDisposition::handled};
    reply.size = 44;
    write(reply.bytes, 4, 9);
    write(reply.bytes, 16, 0x4f434653);
    // The firmware framework writes its interface word, not a token echo.
    write(reply.bytes, 28, 0xec6be3ff);
    if (command == 0) {
        domain_ = true;
        write(reply.bytes, 32, 1);
    }
    // Control3 returns the actual zero-byte host pointer-buffer capacity.
    diagnostic_ = nullptr;
    return reply;
}

const char* ContextWriterSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::err
