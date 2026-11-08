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

#include <chrono>
#include <cstring>
#include <new>
#include <span>
#include <vector>

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

// Errno values of the socket service (FreeBSD's) and poll events.
constexpr std::uint32_t error_bad_descriptor = 9;
constexpr std::uint32_t error_invalid = 22;
constexpr std::uint32_t error_too_many = 24;
constexpr std::uint32_t error_again = 35;
constexpr std::uint32_t error_network_down = 50;
constexpr std::uint16_t poll_in = 0x1, poll_out = 0x4, poll_read_normal = 0x40, poll_invalid = 0x20;
constexpr std::uint32_t event_fd_semaphore = 1, event_fd_nonblocking = 6;
constexpr std::uint64_t event_fd_maximum = 0xffff'ffff'ffff'fffeull;
constexpr std::size_t maximum_descriptors = 128;
// An endless poll returns after this long with nothing ready.
constexpr auto endless_poll_limit = std::chrono::milliseconds(100);

struct PollDescriptor {
    std::int32_t fd;
    std::int16_t events;
    std::int16_t revents;
};
static_assert(sizeof(PollDescriptor) == 8);
} // namespace

BsdSession::BsdSession(std::uint64_t process_id, horizon::HandleTable& handles)
    : BsdSession(process_id, handles, std::make_shared<State>()) {}

BsdSession::BsdSession(std::uint64_t process_id, horizon::HandleTable& handles, std::shared_ptr<State> state)
    : process_id_(process_id), handles_(handles), state_(std::move(state)) {}

horizon::IpcReply BsdSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    return dispatch(message, sender_pid, nullptr);
}

horizon::IpcReply BsdSession::request_with_buffers(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                                                   const horizon::GuestBuffers& buffers) {
    return dispatch(message, sender_pid, &buffers);
}

horizon::IpcReply BsdSession::dispatch(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                                       const horizon::GuestBuffers* buffers) {
    std::unique_lock lock(state_->mutex);
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
        if (command == 36 && domain) {
            // Switch 2 RegisterClient (16 bytes and the process id) -> bsd errno.
            if (!pid) {
                state.diagnostic = "Socket RegisterClient needs the sender PID";
                return {};
            }
            state.registered = true;
            auto reply = reply_with(true, 4);
            write(reply.bytes, 48, 0);
            state.diagnostic = nullptr;
            return reply;
        }
        // Event descriptors: a return value and an errno after the CMIF header.
        const std::size_t out = domain ? 48 : 32;
        const auto result = [&](std::int32_t value, std::uint32_t error) {
            auto reply = reply_with(domain, 8);
            write(reply.bytes, out, static_cast<std::uint32_t>(value));
            write(reply.bytes, out + 4, value < 0 ? error : 0);
            return reply;
        };
        const auto fd_at = [&](std::size_t offset) { return static_cast<std::int32_t>(read(message, offset)); };
        auto& fds = state.event_fds;
        switch (command) {
        case 2:    // Socket(domain, type, protocol) -> descriptor
        case 3:    // SocketExempt(domain, type, protocol) -> descriptor
            // No network exists (nifm reports no connection), so a socket
            // fails as with the network down.
            state.diagnostic = nullptr;
            return result(-1, error_network_down);
        case 31: {  // EventFd(flags, initial value) -> descriptor
            const auto flags = read(message, *cmif + 16);
            const auto initial = read64(message, *cmif + 24);
            state.diagnostic = nullptr;
            lock.unlock();
            if (initial > event_fd_maximum) return result(-1, error_invalid);
            const std::lock_guard fd_lock(fds.mutex);
            std::size_t fd = 0;
            while (fd < fds.table.size() && fds.table[fd]) ++fd;
            if (fd == maximum_descriptors) return result(-1, error_too_many);
            if (fd == fds.table.size()) fds.table.emplace_back();
            fds.table[fd] = EventFd{initial, flags};
            return result(static_cast<std::int32_t>(fd), 0);
        }
        case 24:    // Write(descriptor; value buffer)
        case 25: {  // Read(descriptor; value buffer)
            const auto fd = fd_at(*cmif + 16);
            const bool writing = command == 24;
            if (!buffers || buffers->empty()) {
                state.diagnostic = "Socket read or write needs its buffer";
                return {};
            }
            const auto index = writing ? std::size_t{0} : buffers->size() - 1;
            if ((*buffers)[index].size < 8) {
                state.diagnostic = nullptr;
                return result(-1, error_invalid);
            }
            std::array<std::byte, 8> value{};
            if (writing && !buffers->read(index, value)) {
                state.diagnostic = "Socket write buffer is not readable";
                return {};
            }
            std::unique_lock fd_lock(fds.mutex);
            if (fd < 0 || static_cast<std::size_t>(fd) >= fds.table.size() || !fds.table[fd]) {
                state.diagnostic = nullptr;
                return result(-1, error_bad_descriptor);
            }
            auto& event = *fds.table[fd];
            if (writing) {
                std::uint64_t amount;
                std::memcpy(&amount, value.data(), sizeof amount);
                if (amount > event_fd_maximum) {
                    state.diagnostic = nullptr;
                    return result(-1, error_invalid);
                }
                if (amount > event_fd_maximum - event.counter) {
                    if (!(event.flags & event_fd_nonblocking)) {
                        state.diagnostic = "Socket write would wait on a full blocking event descriptor";
                        return {};
                    }
                    state.diagnostic = nullptr;
                    return result(-1, error_again);
                }
                event.counter += amount;
                fds.changed.notify_all();
                state.diagnostic = nullptr;
                return result(8, 0);
            }
            if (!event.counter) {
                if (!(event.flags & event_fd_nonblocking)) {
                    state.diagnostic = "Socket read would wait on an empty blocking event descriptor";
                    return {};
                }
                state.diagnostic = nullptr;
                return result(-1, error_again);
            }
            const std::uint64_t taken = (event.flags & event_fd_semaphore) ? 1 : event.counter;
            event.counter -= taken;
            fds.changed.notify_all();
            std::memcpy(value.data(), &taken, sizeof taken);
            fd_lock.unlock();
            if (!buffers->write(index, value)) {
                state.diagnostic = "Socket read buffer is not writable";
                return {};
            }
            state.diagnostic = nullptr;
            return result(8, 0);
        }
        case 26: {  // Close(descriptor)
            const auto fd = fd_at(*cmif + 16);
            const std::lock_guard fd_lock(fds.mutex);
            state.diagnostic = nullptr;
            if (fd < 0 || static_cast<std::size_t>(fd) >= fds.table.size() || !fds.table[fd])
                return result(-1, error_bad_descriptor);
            fds.table[fd].reset();
            fds.changed.notify_all();
            return result(0, 0);
        }
        case 6: {  // Poll(count, timeout ms; descriptors in, descriptors out)
            const auto count = read(message, *cmif + 16);
            const auto timeout = static_cast<std::int32_t>(read(message, *cmif + 20));
            if (!buffers || buffers->empty() || count > maximum_descriptors ||
                (*buffers)[0].size < count * sizeof(PollDescriptor) ||
                (*buffers)[buffers->size() - 1].size < count * sizeof(PollDescriptor)) {
                state.diagnostic = "Socket poll needs its descriptor buffers";
                return {};
            }
            std::vector<PollDescriptor> descriptors(count);
            if (count && !buffers->read(0, std::as_writable_bytes(std::span(descriptors)))) {
                state.diagnostic = "Socket poll descriptors are not readable";
                return {};
            }
            state.diagnostic = nullptr;
            lock.unlock();
            const auto ready = [&] {
                int any = 0;
                for (auto& descriptor : descriptors) {
                    descriptor.revents = 0;
                    if (descriptor.fd < 0) continue;
                    const auto fd = static_cast<std::size_t>(descriptor.fd);
                    if (fd >= fds.table.size() || !fds.table[fd]) {
                        descriptor.revents = poll_invalid;
                    } else {
                        if (fds.table[fd]->counter)
                            descriptor.revents |= descriptor.events & (poll_in | poll_read_normal);
                        if (fds.table[fd]->counter < event_fd_maximum) descriptor.revents |= descriptor.events & poll_out;
                    }
                    any += descriptor.revents != 0;
                }
                return any;
            };
            std::unique_lock fd_lock(fds.mutex);
            const auto deadline = std::chrono::steady_clock::now() +
                (timeout < 0 ? std::chrono::milliseconds(endless_poll_limit) : std::chrono::milliseconds(timeout));
            int found = ready();
            while (!found && fds.changed.wait_until(fd_lock, deadline) != std::cv_status::timeout) found = ready();
            if (!found) found = ready();
            fd_lock.unlock();
            if (count && !buffers->write(buffers->size() - 1, std::as_bytes(std::span(descriptors)))) {
                const std::lock_guard relock(state.mutex);
                state.diagnostic = "Socket poll descriptors are not writable";
                return {};
            }
            return result(found, 0);
        }
        default:
            break;
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
