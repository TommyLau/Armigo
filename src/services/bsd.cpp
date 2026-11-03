// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/bsd.h>

#include <armigo/horizon/startup_queries.h>

#include <cstring>
#include <new>

namespace armigo::services::bsd {
namespace {
using namespace horizon;
std::uint32_t read(const IpcMessage& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i != 4; ++i)
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
// A reply with `output_bytes` after the CMIF header, domain or not.
IpcReply reply_with(bool domain, unsigned output_bytes) {
    IpcReply reply{IpcDisposition::handled};
    const unsigned cmif = domain ? 32 : 16;
    reply.size = cmif + 16 + output_bytes + (domain ? 0 : 8);
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, cmif, 0x4f434653);
    return reply;
}
constexpr std::uint32_t special_pid = 1;
} // namespace

BsdSession::BsdSession(std::uint64_t process_id, horizon::HandleTable& handles)
    : BsdSession(process_id, handles, std::make_shared<State>()) {}

BsdSession::BsdSession(std::uint64_t process_id, horizon::HandleTable& handles, std::shared_ptr<State> state)
    : process_id_(process_id), handles_(handles), state_(std::move(state)) {}

horizon::IpcReply BsdSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(state_->mutex);
    auto& state = *state_;
    state.diagnostic = "Socket method or packet is unsupported";
    if (sender_pid != process_id_) {
        state.diagnostic = "Socket session belongs to a different process";
        return {};
    }
    try {
        const auto type = read(message, 0) & 0xffffu;
        if (type == 5 || type == 7) {
            if (read(message, 16) != 0x49434653) return {};
            const auto command = read(message, 24);
            auto reply = reply_with(false, 4);
            if (command == 3) {  // QueryPointerBufferSize: none
                write(reply.bytes, 32, 0);
            } else if (command == 0) {  // ConvertToDomain: the root is object 1
                if (state.domain) return {};
                state.domain = true;
                write(reply.bytes, 32, 1);
            } else if (command == 2) {  // CloneCurrentObject: another session on this state
                reply = reply_with(false, 0);
                reply.size = 32;
                reply.transferred[0] = std::shared_ptr<BsdSession>(new BsdSession(process_id_, handles_, state_));
                reply.transferred_count = 1;
            } else {
                return {};
            }
            state.diagnostic = nullptr;
            return reply;
        }
        if (type != 4 && type != 6) return {};
        // A domain message without a CMIF header closes the objects its domain
        // header names; the SDK sends it when it finalizes a socket client.
        if (state.domain && (read(message, 4) & 0x3ffu) >= 4 && (read(message, 16) & 0xff) == 2 &&
            (read(message, 16) >> 16) == 0) {
            IpcReply reply{IpcDisposition::handled};
            reply.size = 32;
            write(reply.bytes, 4, 6);
            write(reply.bytes, 16, 0x4f434653);
            state.diagnostic = nullptr;
            return reply;
        }
        const auto cmif = locate_cmif(message);
        if (!cmif || read(message, *cmif) != 0x49434653) return {};
        const bool domain = *cmif >= 32 && state.domain && (read(message, *cmif - 16) & 0xff) == 1;
        if (domain && read(message, *cmif - 12) != 1) {
            state.diagnostic = "Socket domain object is unsupported";
            return {};
        }
        const auto command = read(message, *cmif + 8);
        const auto header = read(message, 4);
        const bool pid = (header >> 31) && (read(message, 8) & special_pid);
        if (command == 0 && domain) {
            // RegisterClient(config[32], pid placeholder, work-memory size; copy
            // handle of the work memory) -> s32 status, 0 when admitted.
            if (!pid || state.registered) {
                state.diagnostic = "Socket RegisterClient needs the sender PID and a first registration";
                return {};
            }
            const auto copies = (read(message, 8) >> 1) & 0xfu;
            if (copies != 1) {
                state.diagnostic = "Socket RegisterClient needs one work-memory handle";
                return {};
            }
            auto memory = handles_.lookup_as<TransferMemoryRecord>(static_cast<Handle>(read(message, 20)));
            if (!memory) {
                state.diagnostic = "Socket RegisterClient work-memory handle is not open transfer memory";
                return {};
            }
            for (std::size_t i = 0; i != state.configuration.size(); ++i)
                state.configuration[i] = message[*cmif + 16 + i];
            state.work_memory_size = read64(message, *cmif + 56);
            state.work_memory = std::move(memory);
            state.registered = true;
            auto reply = reply_with(true, 4);
            write(reply.bytes, 48, 0);
            state.diagnostic = nullptr;
            return reply;
        }
        if (command == 1 && !domain) {
            // StartMonitoring(client pid) on the monitor session.
            if (!pid) {
                state.diagnostic = "Socket StartMonitoring needs the sender PID";
                return {};
            }
            state.monitoring = true;
            state.diagnostic = nullptr;
            return reply_with(false, 0);
        }
        return {};
    } catch (const std::bad_alloc&) {
        state.diagnostic = "Socket allocation failed";
    }
    return {};
}

const char* BsdSession::diagnostic() const {
    const std::lock_guard lock(state_->mutex);
    return state_->diagnostic;
}

bool BsdSession::registered() const {
    const std::lock_guard lock(state_->mutex);
    return state_->registered;
}

std::shared_ptr<horizon::KernelObject> BsdSession::work_memory() const {
    const std::lock_guard lock(state_->mutex);
    return state_->work_memory;
}

std::array<std::byte, 32> BsdSession::configuration() const {
    const std::lock_guard lock(state_->mutex);
    return state_->configuration;
}

} // namespace armigo::services::bsd
