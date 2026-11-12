// ============================================================================
// Armigo Services - Initial Time Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/services/time_session.h>

#include <algorithm>
#include <array>
#include <cstring>

#include <limits>
#include <new>

namespace armigo::services::time {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

using namespace horizon::cmif;
constexpr std::uint32_t static_time_interface = 0x0920b7a6;
// Large enough for every pointer buffer the served commands use (the 0xd0-byte
// clock snapshot); the firmware's own size is not established here.
constexpr std::uint16_t pointer_buffer_size = 0x1000;
constexpr std::size_t clock_snapshot_bytes = 0xd0;



bool no_descriptors(const IpcMessage& message, std::uint32_t minimum_words) {
    const auto header = read(message, 4);
    return (header & ~0x3ffU) == 0 && header >= minimum_words &&
           header <= (message.size() - 8) / 4;
}



}  // namespace

TimeServiceSession::TimeServiceSession(
    std::uint64_t process_id, horizon::HandleTable& handles,
    std::shared_ptr<horizon::ReadOnlySharedMemory> shared_memory,
    std::function<std::optional<std::int64_t>()> counter_nanoseconds)
    : process_id_(process_id), handles_(handles),
      root_(std::make_shared<RootObject>(std::move(shared_memory))),
      counter_nanoseconds_(std::move(counter_nanoseconds)) {
    if (!root_->shared_memory || root_->shared_memory->bytes().size() != 0x1000)
        throw std::invalid_argument("Time service requires one initialized 4096-byte shared-memory page");
}

IpcReply TimeServiceSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply TimeServiceSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return dispatch(message, sender_pid, nullptr);
}

IpcReply TimeServiceSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                                  const horizon::GuestBuffers& buffers) {
    return dispatch(message, sender_pid, &buffers);
}

IpcReply TimeServiceSession::dispatch(const IpcMessage& message, std::uint64_t sender_pid,
                                      const horizon::GuestBuffers* buffers) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_)
        return unsupported("Time session belongs to a different configured process");
    try {
        const auto type = read(message, 0);
        if (type == 5 || type == 7) return control(message);
        if (type == 4 || type == 6) return domain_request(message, buffers);
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
    if (command == 3) return control_reply_bytes(pointer_buffer_size, 2);
    if (command != 0) return unsupported("Time-service control command is unsupported");
    if (!objects_.empty()) return unsupported("Repeated time-service domain conversion is unsupported");
    auto reply = control_reply_bytes(1, 4);
    objects_.push_back(root_);
    return reply;
}

IpcReply TimeServiceSession::domain_request(const IpcMessage& message, const horizon::GuestBuffers* buffers) {
    if (objects_.empty()) return unsupported("Non-domain time-service requests are unsupported");
    // Only a receive list (an output pointer buffer) may accompany a request.
    const auto header = read(message, 4);
    const bool receive_list = buffers && (read(message, 0) & 0xffff0000U) == 0 && !(header >> 31) &&
                              ((header >> 10) & 0xfU) >= 2 && (header & 0x3ffU) >= 12;
    if (!receive_list && !no_descriptors(message, 12))
        return unsupported("Time-service startup requires no PID, handles or buffer descriptors");
    if (read(message, 16, 1) != 1 || read(message, 17, 1) != 0)
        return unsupported("Time-service domain operation or input objects are unsupported");
    const auto data_size = read(message, 18, 2);
    const auto raw_end = 8 + 4 * (read(message, 4) & 0x3ffU);
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
    if (command == 400) return clock_snapshot(message, buffers);
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

void put_bytes(std::span<std::byte> out, std::size_t offset, std::uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) out[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

// `time` as a UTC calendar time (8 bytes) and its additional information (24
// bytes): weekday, day of the year, zone name, no daylight saving, offset 0.
void utc_calendar(std::int64_t time, std::span<std::byte> calendar, std::span<std::byte> info) {
    auto days = time / 86400;
    auto remainder = time % 86400;
    if (remainder < 0) {
        remainder += 86400;
        --days;
    }
    std::int64_t year = 0;
    unsigned month = 0, day = 0;
    civil_from_days(days, year, month, day);
    put_bytes(calendar, 0, static_cast<std::uint64_t>(year), 2);
    calendar[2] = static_cast<std::byte>(month);
    calendar[3] = static_cast<std::byte>(day);
    calendar[4] = static_cast<std::byte>(remainder / 3600);
    calendar[5] = static_cast<std::byte>((remainder % 3600) / 60);
    calendar[6] = static_cast<std::byte>(remainder % 60);
    put_bytes(info, 0, static_cast<std::uint32_t>((days % 7 + 11) % 7), 4);  // 0 = Sunday
    put_bytes(info, 4, day_of_year(year, month, day), 4);
    const char zone[] = "UTC";
    for (std::size_t index = 0; index < sizeof(zone); ++index) info[8 + index] = static_cast<std::byte>(zone[index]);
    put_bytes(info, 16, 0, 4);  // not daylight saving time
    put_bytes(info, 20, 0, 4);  // UTC offset in seconds
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
        reply.size = 48 + 0x20;
        write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
        write(reply.bytes, 16, 1);
        cmif_header(reply.bytes, 32, static_time_interface);
        const std::span<std::byte> out(reply.bytes.data() + 48, 0x20);
        utc_calendar(time, out.first(8), out.subspan(8));
        return reply;
    }
    default:
        return unsupported("Time-zone command is outside the initial startup implementation");
    }
}

IpcReply TimeServiceSession::clock_snapshot(const IpcMessage& message, const horizon::GuestBuffers* buffers) {
    // GetClockSnapshot(type) -> ClockSnapshot in one fixed 0xd0-byte output
    // pointer buffer: user and network contexts, their current times, UTC
    // calendars and additional information, the steady time point, the
    // location name, automatic correction and the type.
    if (!buffers || buffers->size() != 1 || !(*buffers)[0].output || (*buffers)[0].size < clock_snapshot_bytes)
        return unsupported("Time GetClockSnapshot needs one 0xd0-byte output buffer");
    const auto counter = counter_nanoseconds_ ? counter_nanoseconds_() : std::nullopt;
    if (!counter) return unsupported("Time GetClockSnapshot has no guest counter");
    // The page's current slots: the steady clock {offset ns, source} at 0x08,
    // the user and network contexts {offset s, steady s, source} at 0x40 and
    // 0x88, and automatic correction at 0xcc.
    const auto page = root_->shared_memory->bytes();
    const auto s64_at = [&](std::size_t offset) {
        std::int64_t value = 0;
        std::memcpy(&value, page.data() + offset, sizeof value);
        return value;
    };
    const auto steady_seconds = (*counter + s64_at(0x08)) / 1'000'000'000;
    const auto follows_steady = [&](std::size_t context) {
        return std::equal(page.begin() + 0x10, page.begin() + 0x20, page.begin() + context + 16);
    };
    if (!follows_steady(0x40)) return unsupported("Time user clock does not follow the steady clock");
    const auto user = s64_at(0x40) + steady_seconds;
    // Without a network clock its time is zero, as the console reports it.
    const auto network = follows_steady(0x88) ? s64_at(0x88) + steady_seconds : 0;
    std::array<std::byte, clock_snapshot_bytes> snapshot{};
    const std::span<std::byte> out(snapshot);
    std::copy_n(page.begin() + 0x40, 32, out.begin());
    std::copy_n(page.begin() + 0x88, 32, out.begin() + 0x20);
    put_bytes(out, 0x40, static_cast<std::uint64_t>(user), 8);
    put_bytes(out, 0x48, static_cast<std::uint64_t>(network), 8);
    utc_calendar(user, out.subspan(0x50, 8), out.subspan(0x60, 0x18));
    utc_calendar(network, out.subspan(0x58, 8), out.subspan(0x78, 0x18));
    put_bytes(out, 0x90, static_cast<std::uint64_t>(steady_seconds), 8);
    std::copy_n(page.begin() + 0x10, 16, out.begin() + 0x98);
    const char name[] = "UTC";
    for (std::size_t index = 0; index < sizeof(name); ++index) out[0xa8 + index] = static_cast<std::byte>(name[index]);
    out[0xcc] = page[0xcc];
    out[0xcd] = message[48];
    if (!buffers->write(0, snapshot)) return unsupported("Time GetClockSnapshot buffer is not writable");
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 48;
    write(reply.bytes, 4, 10);
    cmif_header(reply.bytes, 32, static_time_interface);
    return reply;
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
