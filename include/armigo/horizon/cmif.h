// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Words of a CMIF message as services read and write them: the request
// (SFCI) and reply (SFCO) magics, and little-endian values at byte offsets of
// the IPC message buffer.

#pragma once

#include <armigo/horizon/ipc.h>

#include <cstddef>
#include <cstdint>

namespace armigo::horizon::cmif {

inline constexpr std::uint32_t sfci = 0x49434653;
inline constexpr std::uint32_t sfco = 0x4f434653;

// The `width` bytes (1 to 4) at `offset`, little-endian.
inline std::uint32_t read(const IpcMessage& message, std::size_t offset, unsigned width = 4) noexcept {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < width; ++index)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + index])} << (8 * index);
    return value;
}

inline void write(IpcMessage& message, std::size_t offset, std::uint32_t value, unsigned width = 4) noexcept {
    for (unsigned index = 0; index < width; ++index)
        message[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
}

inline std::uint64_t read64(const IpcMessage& message, std::size_t offset) noexcept {
    return std::uint64_t{read(message, offset)} | (std::uint64_t{read(message, offset + 4)} << 32);
}

// The CMIF output header at `offset`: SFCO, then (Version and Result stay
// zero) the interface identifier this firmware writes in the fourth word in
// place of the input token.
inline void cmif_header(IpcMessage& message, std::size_t offset, std::uint32_t interface_id) noexcept {
    write(message, offset, sfco);
    write(message, offset + 12, interface_id);
}

// A non-domain reply with `output_words` of output after the CMIF header.
inline IpcReply reply_with(unsigned output_words) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 8 + 4 * (8 + output_words);
    write(reply.bytes, 4, 8 + output_words);
    write(reply.bytes, 16, sfco);
    return reply;
}

// The reply to a CMIF control request: nine words with the framework's
// interface word, and `value` as the output word (one byte wide replies end
// before it).
inline IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 28, 0xec6be3ff);
    write(reply.bytes, 32, value);
    if (width == 1) reply.size = 40;
    return reply;
}

// The control reply that always carries the output word: nine words, the
// framework's interface word, and only the low `width` bytes of `value`.
inline IpcReply control_reply_bytes(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    cmif_header(reply.bytes, 16, 0xec6be3ff);
    write(reply.bytes, 32, value, width);
    return reply;
}

} // namespace armigo::horizon::cmif
