// ============================================================================
// Armigo Services - Initial Time Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/time_session.h>

#include <limits>
#include <new>

namespace armigo::services::time {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
constexpr std::uint32_t control_interface = 0xec6be3ff;
constexpr std::uint32_t static_time_interface = 0x0920b7a6;

std::uint32_t read(const IpcMessage& bytes, std::size_t offset, unsigned count = 4) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i)
        value |= std::uint32_t{std::to_integer<unsigned char>(bytes[offset + i])} << (8 * i);
    return value;
}

void write(IpcMessage& bytes, std::size_t offset, std::uint32_t value, unsigned count = 4) {
    for (unsigned i = 0; i < count; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

bool no_descriptors(const IpcMessage& message, std::uint32_t minimum_words) {
    const auto header = read(message, 4);
    return (header & ~0x3ffU) == 0 && header >= minimum_words &&
           header <= (message.size() - 8) / 4;
}

void cmif_header(IpcMessage& message, std::size_t offset, std::uint32_t interface_id) {
    write(message, offset, sfco);
    write(message, offset + 12, interface_id);
}

IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    cmif_header(reply.bytes, 16, control_interface);
    write(reply.bytes, 32, value, width);
    return reply;
}

}  // namespace

TimeServiceSession::TimeServiceSession(
    std::uint64_t process_id, horizon::HandleTable& handles,
    std::shared_ptr<horizon::ReadOnlySharedMemory> shared_memory)
    : process_id_(process_id), handles_(handles),
      root_(std::make_shared<RootObject>(std::move(shared_memory))) {
    if (!root_->shared_memory || root_->shared_memory->bytes().size() != 0x1000)
        throw std::invalid_argument("Time service requires one initialized 4096-byte shared-memory page");
}

IpcReply TimeServiceSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply TimeServiceSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_)
        return unsupported("Time session belongs to a different configured process");
    try {
        const auto type = read(message, 0);
        if (type == 5 || type == 7) return control(message);
        if (type == 4 || type == 6) return domain_request(message);
        return unsupported("Time-service HIPC message type or descriptors are unsupported");
    } catch (const std::bad_alloc&) {
        return unsupported("Time-service host allocation failed; error reply is unsupported");
    } catch (const std::length_error&) {
        return unsupported("Time-service object storage exhausted; error reply is unsupported");
    }
}

IpcReply TimeServiceSession::control(const IpcMessage& message) {
    if (!no_descriptors(message, 8) || read(message, 16) != sfci)
        return unsupported("Time-service control packet is outside the supported wire format");
    // Context-bearing forms select the same methods. Their request context
    // does not alter these responses or replace the fixed interface tag.
    const auto command = read(message, 24);
    if (command == 3) return control_reply(0, 2);
    if (command != 0) return unsupported("Time-service control command is unsupported");
    if (!objects_.empty()) return unsupported("Repeated time-service domain conversion is unsupported");
    auto reply = control_reply(1, 4);
    objects_.push_back(root_);
    return reply;
}

IpcReply TimeServiceSession::domain_request(const IpcMessage& message) {
    if (objects_.empty()) return unsupported("Non-domain time-service requests are unsupported");
    if (!no_descriptors(message, 12))
        return unsupported("Time-service startup requires no PID, handles or buffer descriptors");
    if (read(message, 16, 1) != 1 || read(message, 17, 1) != 0)
        return unsupported("Time-service domain operation or input objects are unsupported");
    const auto data_size = read(message, 18, 2);
    const auto raw_end = 8 + 4 * read(message, 4);
    if (data_size < 16 || data_size > raw_end - 32)
        return unsupported("Time-service domain payload length is unsupported");
    const auto object_id = read(message, 20);
    if (object_id == 0 || object_id > objects_.size())
        return unsupported("Time-service domain object does not exist in this session");
    const auto is_child =
        std::dynamic_pointer_cast<TimeZoneObject>(objects_[object_id - 1]) != nullptr;
    if (!is_child && objects_[object_id - 1] != root_)
        return unsupported("Time-service domain object is not a known interface");
    if (read(message, 32) != sfci)
        return unsupported("Time-service CMIF magic is unsupported");

    const auto command = read(message, 40);
    if (is_child) return time_zone_command(command, message);
    if (command == 3) {
        if (objects_.size() >= std::numeric_limits<std::uint32_t>::max())
            return unsupported("Time-service domain object IDs are exhausted");
        const auto child = std::make_shared<TimeZoneObject>(root_);
        const auto id = static_cast<std::uint32_t>(objects_.size() + 1);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 60;
        write(reply.bytes, 4, 13);
        write(reply.bytes, 16, 1);
        cmif_header(reply.bytes, 32, static_time_interface);
        write(reply.bytes, 48, id);
        objects_.push_back(child);
        return reply;
    }
    if (command == 20) {
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 64;
        write(reply.bytes, 4, 0x8000000c);
        write(reply.bytes, 8, 2); // One copied handle; no PID or moved handles.
        cmif_header(reply.bytes, 32, static_time_interface);
        // Publication is last: no later operation can allocate or throw. The
        // root retains its reference, and the new caller handle pins the same
        // object independently of the time session and other copied handles.
        const auto handle = handles_.insert(root_->shared_memory);
        if (!handle) return unsupported("Caller handle table is full; transfer error reply unsupported");
        write(reply.bytes, 12, *handle);
        return reply;
    }
    return unsupported("Time-service command is outside the initial startup implementation");
}


namespace {
std::int64_t read_s64(const IpcMessage& message, std::size_t offset) {
    const auto low = std::uint64_t{read(message, offset)};
    const auto high = std::uint64_t{read(message, offset + 4)};
    return static_cast<std::int64_t>(low | (high << 32));
}

// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm), used for
// the UTC calendar the time zone interface reports while no console rule exists.
void civil_from_days(std::int64_t days, std::int64_t& year, unsigned& month, unsigned& day) {
    const auto z = days + 719468;
    const auto era = (z >= 0 ? z : z - 146096) / 146097;
    const auto doe = static_cast<unsigned>(z - era * 146097);
    const auto yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const auto y = static_cast<std::int64_t>(yoe) + era * 400;
    const auto doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const auto mp = (5 * doy + 2) / 153;
    day = doy - (153 * mp + 2) / 5 + 1;
    month = mp < 10 ? mp + 3 : mp - 9;
    year = y + (month <= 2);
}

unsigned day_of_year(std::int64_t year, unsigned month, unsigned day) {
    const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    const unsigned lengths[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    unsigned result = 0;
    for (unsigned index = 0; index + 1 < month; ++index)
        result += lengths[index] + (index == 1 && leap ? 1 : 0);
    return result + day - 1;
}
}  // namespace

IpcReply TimeServiceSession::time_zone_command(std::uint32_t command, const IpcMessage& message) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    switch (command) {
    case 0: {  // GetDeviceLocationName: one 0x24-byte location name.
        reply.size = 48 + 0x24;
        write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
        write(reply.bytes, 16, 1);
        cmif_header(reply.bytes, 32, static_time_interface);
        const char name[] = "UTC";
        for (std::size_t index = 0; index < sizeof(name); ++index)
            reply.bytes[48 + index] = static_cast<std::byte>(name[index]);
        return reply;
    }
    case 2:  // GetTotalLocationNameCount
        reply.size = 52;
        write(reply.bytes, 4, 11);
        write(reply.bytes, 16, 1);
        cmif_header(reply.bytes, 32, static_time_interface);
        write(reply.bytes, 48, 1);
        return reply;
    case 5: {  // GetTimeZoneRuleVersion
        reply.size = 48 + 0x10;
        write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
        write(reply.bytes, 16, 1);
        cmif_header(reply.bytes, 32, static_time_interface);
        const char version[] = "UTC";
        for (std::size_t index = 0; index < sizeof(version); ++index)
            reply.bytes[48 + index] = static_cast<std::byte>(version[index]);
        return reply;
    }
    case 100:  // ToCalendarTime
    case 101: {  // ToCalendarTimeWithMyRule
        const auto time = read_s64(message, 48);
        auto days = time / 86400;
        auto remainder = time % 86400;
        if (remainder < 0) {
            remainder += 86400;
            --days;
        }
        std::int64_t year = 0;
        unsigned month = 0, day = 0;
        civil_from_days(days, year, month, day);
        reply.size = 48 + 0x20;
        write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
        write(reply.bytes, 16, 1);
        cmif_header(reply.bytes, 32, static_time_interface);
        write(reply.bytes, 48, static_cast<std::uint32_t>(year), 2);
        reply.bytes[50] = static_cast<std::byte>(month);
        reply.bytes[51] = static_cast<std::byte>(day);
        reply.bytes[52] = static_cast<std::byte>(remainder / 3600);
        reply.bytes[53] = static_cast<std::byte>((remainder % 3600) / 60);
        reply.bytes[54] = static_cast<std::byte>(remainder % 60);
        const auto weekday = static_cast<std::uint32_t>((days % 7 + 11) % 7); // 0 = Sunday
        write(reply.bytes, 56, weekday);
        write(reply.bytes, 60, day_of_year(year, month, day));
        const char zone[] = "UTC";
        for (std::size_t index = 0; index < sizeof(zone); ++index)
            reply.bytes[64 + index] = static_cast<std::byte>(zone[index]);
        write(reply.bytes, 72, 0);  // not daylight saving time
        write(reply.bytes, 76, 0);  // UTC offset in seconds
        return reply;
    }
    default:
        return unsupported("Time-zone command is outside the initial startup implementation");
    }
}

std::size_t TimeServiceSession::domain_object_count() const {
    const std::lock_guard lock(mutex_);
    return objects_.size();
}

bool TimeServiceSession::is_time_zone_object(std::uint32_t id) const {
    const std::lock_guard lock(mutex_);
    return id != 0 && id <= objects_.size() &&
           std::dynamic_pointer_cast<TimeZoneObject>(objects_[id - 1]) != nullptr;
}

const char* TimeServiceSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

}  // namespace armigo::services::time
