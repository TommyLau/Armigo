// ============================================================================
// Armigo Services - Offline Play Report Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// SaveReportWithUser (10107) shares command 10105's report path.
// Validation and result propagation follow Horizon firmware;
// application reports return the validation result. Switch 2's
// SendReportWithUser (10500) carries its report in a transfer memory and
// returns an async context; the offline host validates it the same way.

#include <armigo/services/play_report.h>

#include "play_report_validation.h"

#include <algorithm>
#include <array>
#include <new>
#include <span>
#include <optional>
#include <vector>

namespace armigo::services::prepo {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
constexpr std::uint32_t service_interface = 0x2749f93d;
constexpr std::uint32_t control_interface = 0xec6be3ff;
constexpr std::uint32_t invalid_argument = 0x281;
constexpr std::uint32_t invalid_room = 0xa81;
constexpr std::uint32_t invalid_report = 0x1281;

std::uint32_t read(const IpcMessage& message, std::size_t offset, unsigned width = 4) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < width; ++index)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + index])}
                 << (8 * index);
    return value;
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index)
        message[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
}

IpcReply control_reply(std::uint32_t value) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 44;
    write(reply.bytes, 4, 9);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 28, control_interface);
    write(reply.bytes, 32, value);
    return reply;
}

IpcReply report_reply(bool domain, std::uint32_t result) {
    const std::size_t cmif = domain ? 32 : 16;
    IpcReply reply{IpcDisposition::handled};
    reply.size = cmif + 24;
    write(reply.bytes, 4, (reply.size - 8) / 4);
    write(reply.bytes, cmif, sfco);
    write(reply.bytes, cmif + 8, result);
    write(reply.bytes, cmif + 12, service_interface);
    return reply;
}

std::uint64_t address(std::uint32_t low, std::uint32_t high, std::uint32_t middle) {
    return (std::uint64_t{high} << 36) | (std::uint64_t{middle} << 32) | low;
}

struct InputBuffer {
    std::uint64_t address;
    std::uint64_t size;
};

InputBuffer pointer_buffer(const IpcMessage& message, std::size_t offset) {
    const auto packed = read(message, offset);
    return {address(read(message, offset + 4), (packed >> 6) & 7, (packed >> 12) & 15),
            packed >> 16};
}

InputBuffer map_alias_buffer(const IpcMessage& message, std::size_t offset) {
    const auto packed = read(message, offset + 8);
    return {address(read(message, offset + 4), (packed >> 2) & 7, packed >> 28),
            (std::uint64_t{(packed >> 24) & 15} << 32) | read(message, offset)};
}

struct ReportPacket {
    std::size_t cmif;
    std::size_t data_end;
    InputBuffer room;
    InputBuffer report;
};

// Parse the declared HIPC extent, not a scan for magic in descriptor or payload
// bytes. Keep empty descriptors and all 36 size bits for service validation.
std::optional<ReportPacket> report_packet(const IpcMessage& message, bool domain) {
    const auto header = read(message, 0);
    const auto second = read(message, 4);
    if ((header & 0xffff0000u) != 0x00110000u ||
        (second & ~0x3ffu) != 0x80000000u || read(message, 8) != 1)
        return {};
    constexpr std::size_t raw_start = 40;
    const auto words = second & 0x3ffu;
    if (words > (message.size() - raw_start) / 4) return {};
    const auto end = raw_start + 4 * words;
    const std::size_t cmif = domain ? 64 : 48;
    // HIPC raw words include sixteen bytes of alignment allowance: eight
    // precede this CMIF/domain header and eight follow the command data.
    if (end < cmif + 24 || read(message, cmif) != sfci) return {};
    std::size_t data_end = end - 8;
    if (domain) {
        const auto domain_header = read(message, 48);
        const auto data_size = domain_header >> 16;
        if ((domain_header & 0xffffu) != 1 || read(message, 52) != 1 ||
            data_size < 16 || data_size > data_end - cmif)
            return {};
        data_end = cmif + data_size;
    }
    return ReportPacket{cmif, data_end, pointer_buffer(message, 20),
                        map_alias_buffer(message, 28)};
}

// SendReportWithUser (10500), Switch 2 23.0.1: one X buffer (room), PID and
// one copied transfer memory, so the raw data starts at byte 32. Its 0x28
// bytes: opt-in check flag (u8, padded to four), transfer memory size, report
// size, padding, user ID and PID.
struct SendPacket {
    std::size_t cmif;
    InputBuffer room;
};
constexpr std::size_t send_data_size = 0x28;

std::optional<SendPacket> send_packet(const IpcMessage& message, bool domain) {
    const auto header = read(message, 0);
    const auto second = read(message, 4);
    if ((header & 0xffff0000u) != 0x00010000u || (second & ~0x3ffu) != 0x80000000u || read(message, 8) != 3)
        return {};
    constexpr std::size_t raw_start = 32;
    const auto words = second & 0x3ffu;
    if (words > (message.size() - raw_start) / 4) return {};
    const auto end = raw_start + 4 * words;
    const std::size_t cmif = domain ? 48 : 32;
    if (end < cmif + 16 + send_data_size || read(message, cmif) != sfci) return {};
    if (domain) {
        const auto domain_header = read(message, 32);
        if ((domain_header & 0xffffu) != 1 || read(message, 36) != 1 ||
            (domain_header >> 16) < 16 + send_data_size || (domain_header >> 16) > end - cmif)
            return {};
    }
    return SendPacket{cmif, pointer_buffer(message, 24)};
}

// A new domain object: the out header counts it and its id follows the result.
IpcReply object_reply(std::uint32_t object) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 52;
    write(reply.bytes, 4, (reply.size - 8) / 4);
    write(reply.bytes, 16, 1);
    write(reply.bytes, 32, sfco);
    write(reply.bytes, 44, service_interface);
    write(reply.bytes, 48, object);
    return reply;
}

bool valid_room(std::span<const std::byte> room) {
    if (room.front() == std::byte{0}) return false;
    for (const auto byte : room) {
        if (byte == std::byte{0}) return true;
        const auto value = std::to_integer<unsigned char>(byte);
        if ((value < 'a' || value > 'z') && (value < '0' || value > '9') && value != '_')
            return false;
    }
    return false;
}

} // namespace

IpcReply PlayReportSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply PlayReportSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return request_with_buffers(message, sender_pid, {});
}

IpcReply PlayReportSession::request_with_buffers(const IpcMessage& message,
                                                std::uint64_t sender_pid,
                                                const horizon::GuestBuffers& buffers) {
    return request_with_handles(message, sender_pid, buffers, {});
}

IpcReply PlayReportSession::send_report(const IpcMessage& message, const horizon::GuestBuffers& buffers,
    std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects) {
    const auto packet = send_packet(message, domain_);
    if (!packet)
        return unsupported("Play-report send needs a PID, one X buffer, one copied transfer memory and its data");
    if (!domain_) return unsupported("Play-report send returns an async context only to a domain session");
    const auto data = packet->cmif + 16;
    // The same validation order as SaveReportWithUser: room, user, report.
    const auto& room = packet->room;
    if (room.size == 0 || room.size > 32) return report_reply(domain_, invalid_argument);
    std::array<std::byte, 32> room_bytes{};
    const auto name = std::span{room_bytes}.first(static_cast<std::size_t>(room.size));
    if (!buffers.read_memory(room.address, name))
        return unsupported("Play-report room buffer is not readable");
    const auto uid = std::span{message}.subspan(data + 0x10, 16);
    if (name.back() != std::byte{0} ||
        std::all_of(uid.begin(), uid.end(), [](std::byte value) { return value == std::byte{0}; }))
        return report_reply(domain_, invalid_argument);
    if (!valid_room(name)) return report_reply(domain_, invalid_room);

    const auto memory = copied_objects.size() == 1
        ? std::dynamic_pointer_cast<horizon::TransferMemoryRecord>(copied_objects.front()) : nullptr;
    if (!memory) return unsupported("Play-report send needs one copied transfer memory");
    const auto memory_size = read(message, data + 4);
    const auto report_size = read(message, data + 8);
    if (memory_size != memory->range().size) return report_reply(domain_, invalid_argument);
    if (report_size == 0 || report_size > 0x10000 || report_size > memory_size)
        return report_reply(domain_, invalid_report);
    try {
        std::vector<std::byte> report(report_size);
        if (!buffers.read_memory(memory->range().address, report))
            return unsupported("Play-report transfer memory is not readable");
        if (!detail::valid_report(report)) return report_reply(domain_, invalid_report);
        // The offline host accepts and discards the report, as for a save, and
        // returns an async context that has already completed.
        contexts_.push_back(std::make_shared<horizon::Event>(true));
    } catch (const std::bad_alloc&) {
        return unsupported("Play-report send storage could not be allocated");
    } catch (const std::length_error&) {
        return unsupported("Play-report send storage could not be allocated");
    }
    return object_reply(static_cast<std::uint32_t>(contexts_.size() + 1));
}

// IAsyncContext, Switch 2 23.0.1: 0 GetSystemEvent (a copied event handle),
// 1 Cancel, 2 GetResult. An offline context completed when it was created.
IpcReply PlayReportSession::context_request(std::uint32_t object, std::uint32_t command) {
    if (object < 2 || object - 2 >= contexts_.size() || !contexts_[object - 2])
        return unsupported("Play-report request names no open async context");
    if (command > 2) return unsupported("Play-report async context command is not implemented");
    auto reply = report_reply(true, 0);
    if (command == 0) {
        reply.transferred[0] = contexts_[object - 2];
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
    }
    return reply;
}

IpcReply PlayReportSession::request_with_handles(const IpcMessage& message, std::uint64_t sender_pid,
    const horizon::GuestBuffers& buffers, std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_)
        return unsupported("Play-report session belongs to a different process");

    const auto type = read(message, 0, 2);
    if (type == 5 || type == 7) {
        const auto words = read(message, 4);
        if (read(message, 0) != type || words < 8 ||
            words > (message.size() - 8) / 4 || read(message, 16) != sfci)
            return unsupported("Play-report control packet is malformed");
        const auto command = read(message, 24);
        if (command == 3) return control_reply(0x80);
        if (command == 0 && !domain_) {
            domain_ = true;
            return control_reply(1);
        }
        return unsupported("Play-report control command is unsupported");
    }
    if (type != 4 && type != 6)
        return unsupported("Play-report HIPC message type is unsupported");
    if (domain_ && read(message, 4) >= 6 && read(message, 4) < 12 && (read(message, 16) & 0xff) == 2) {
        // A domain close releases one async context; its reply has no CMIF payload.
        const auto id = read(message, 20);
        if (id < 2 || id - 2 >= contexts_.size() || !contexts_[id - 2])
            return unsupported("Play-report domain close names no open async context");
        contexts_[id - 2].reset();
        IpcReply reply{IpcDisposition::handled};
        reply.size = 32;
        write(reply.bytes, 4, 6);
        write(reply.bytes, 16, sfco);
        return reply;
    }
    // A domain request names its object in the domain header before the CMIF
    // header. Object 1 is the service; later ones are async contexts.
    if (const auto cmif = horizon::locate_cmif(message); cmif && *cmif + 12 <= message.size()) {
        if (domain_ && *cmif >= 32 && read(message, *cmif - 12) != 1)
            return context_request(read(message, *cmif - 12), read(message, *cmif + 8));
        if (read(message, *cmif + 8) == 10500) return send_report(message, buffers, copied_objects);
    }
    const auto packet = report_packet(message, domain_);
    if (!packet)
        return unsupported("Play-report request needs a PID, one X buffer and one A buffer");
    const auto command = read(message, packet->cmif + 8);
    if (command != 10105 && command != 10107)
        return unsupported("Play-report command is not implemented; only SaveReportWithUser is supported");
    // 10107 carries the opt-in check flag first, padded to eight bytes, then
    // the same user ID and PID as 10105. The offline host discards both forms.
    const std::size_t uid_offset = command == 10107 ? 24 : 16;
    if (packet->data_end - packet->cmif < uid_offset + 24)
        return unsupported("Play-report user ID and PID payload is truncated");

    // Preserve the firmware's validation order. The kernel-supplied sender,
    // not either guest-controlled PID field, identifies this application.
    const auto& room = packet->room;
    if (room.size == 0 || room.size > 32) return report_reply(domain_, invalid_argument);
    std::array<std::byte, 32> room_bytes{};
    const auto name = std::span{room_bytes}.first(static_cast<std::size_t>(room.size));
    if (!buffers.read_memory(room.address, name))
        return unsupported("Play-report room buffer is not readable");
    const auto uid = std::span{message}.subspan(packet->cmif + uid_offset, 16);
    if (name.back() != std::byte{0} ||
        std::all_of(uid.begin(), uid.end(), [](std::byte value) { return value == std::byte{0}; }))
        return report_reply(domain_, invalid_argument);
    if (!valid_room(name)) return report_reply(domain_, invalid_room);

    const auto& report = packet->report;
    if (report.size == 0 || report.size > 0x10000)
        return report_reply(domain_, invalid_report);
    try {
        std::vector<std::byte> data(static_cast<std::size_t>(report.size));
        if (!buffers.read_memory(report.address, data))
            return unsupported("Play-report payload buffer is not readable");
        if (!detail::valid_report(data)) return report_reply(domain_, invalid_report);
    } catch (const std::bad_alloc&) {
        return unsupported("Play-report validation storage could not be allocated");
    }
    // The offline host accepts the validated report and discards it. It has
    // no persistent report store, transmission queue or telemetry side effects.
    return report_reply(domain_, 0);
}

const char* PlayReportSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::prepo
