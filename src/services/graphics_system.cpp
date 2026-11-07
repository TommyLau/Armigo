// ============================================================================
// Armigo Services - Switch 2 Graphics System Session
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/services/graphics_system.h>
#include <armigo/horizon/startup_queries.h>

#include <algorithm>
#include <array>
#include <vector>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <system_error>

namespace armigo::services::gsv {
namespace {
std::uint32_t read(const horizon::IpcMessage& message, unsigned offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(message[offset + i])) << (i * 8);
    return value;
}
void write(horizon::IpcMessage& message, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
        message[offset + i] = static_cast<std::byte>((value >> (i * 8)) & 255);
}

// A TIPC reply without handles: the request type, the raw word count, the
// result, then the data words.
horizon::IpcReply tipc_reply(std::uint32_t type, std::uint32_t result, std::initializer_list<std::uint32_t> data) {
    horizon::IpcReply reply{horizon::IpcDisposition::handled};
    write(reply.bytes, 0, type);
    write(reply.bytes, 4, 1 + static_cast<std::uint32_t>(data.size()));
    write(reply.bytes, 8, result);
    unsigned offset = 12;
    for (const auto word : data) {
        write(reply.bytes, offset, word);
        offset += 4;
    }
    reply.size = offset;
    return reply;
}

// A TIPC reply carrying one copied handle: the handle at word 3, the result at word 4.
horizon::IpcReply tipc_handle_reply(std::uint32_t type, horizon::Handle handle) {
    horizon::IpcReply reply{horizon::IpcDisposition::handled};
    write(reply.bytes, 0, type);
    write(reply.bytes, 4, 0x80000001);
    write(reply.bytes, 8, 2);
    write(reply.bytes, 12, handle);
    write(reply.bytes, 16, 0);
    reply.size = 20;
    return reply;
}

// The command of a TIPC request and the byte offset of its raw data, or
// nothing for a request that is not TIPC.
struct TipcRequest { std::uint32_t type; std::uint32_t command; unsigned data; std::uint32_t raw_words; };
std::optional<TipcRequest> parse_tipc(const horizon::IpcMessage& message) {
    const auto type = read(message, 0) & 0xffff;
    const auto header = read(message, 4);
    if (type < 16 || (read(message, 0) >> 16) != 0) return std::nullopt;
    unsigned data = 8;
    if (header & 0x80000000u) {
        const auto special = read(message, 8);
        data = 12 + ((special & 1) ? 8 : 0) + 4 * (((special >> 1) & 15) + ((special >> 5) & 15));
    }
    return TipcRequest{type, type - 16, data, header & 0x3ff};
}

void warn_provisional(std::uint64_t& warned, std::size_t index, const char* session, const char* name,
                      std::uint32_t command) {
    if (warned & (std::uint64_t{1} << index)) return;
    warned |= std::uint64_t{1} << index;
    armigo::diag::write(armigo::diag::Level::warning,
        "Armigo %s: provisional handler %s (TIPC %u) answers with an unverified reply\n", session, name, command);
}

// Reference stubs adopted ahead of firmware verification: success with the
// listed number of zero data words, and a one-time warning per session. The
// table below lists them all.
struct Provisional {
    std::uint32_t command; std::uint32_t zero_words; const char* name;
};
constexpr Provisional provisional[] = {
    {4, 0, "Boot"}, {5, 0, "command 5"}, {101, 0, "command 101"}, {601, 0, "camera initialization"},
    {901, 0, "video initialization"}, {1001, 0, "second video initialization"}, {1201, 0, "chat initialization"},
    {113, 32, "command 113"}, {120, 32, "command 120"}, {121, 32, "command 121"}, {122, 32, "command 122"},
    {123, 32, "command 123"}, {124, 32, "command 124"}, {125, 32, "command 125"}, {131, 32, "command 131"},
    {133, 32, "command 133"}, {187, 32, "command 187"}, {211, 32, "command 211"}, {212, 32, "command 212"},
    {213, 32, "command 213"}, {214, 32, "command 214"}, {215, 32, "command 215"}, {303, 32, "command 303"},
    {424, 32, "command 424"}, {425, 32, "command 425"}, {436, 32, "command 436"}, {461, 32, "command 461"},
    {462, 32, "command 462"}, {463, 32, "command 463"}, {464, 32, "command 464"}, {465, 32, "command 465"},
    {466, 32, "command 466"}, {467, 32, "command 467"}, {468, 32, "command 468"}, {493, 32, "command 493"},
    {603, 32, "command 603"}, {613, 32, "command 613"}, {614, 32, "command 614"}, {802, 32, "command 802"},
    {803, 32, "command 803"}, {813, 32, "command 813"}, {814, 32, "command 814"}, {1003, 32, "command 1003"},
    {1011, 32, "command 1011"}, {1012, 32, "command 1012"}, {1091, 32, "command 1091"}, {1112, 32, "command 1112"},
    {1113, 32, "command 1113"}, {1203, 32, "command 1203"}, {1602, 32, "command 1602"}, {1603, 32, "command 1603"},
};
static_assert(std::size(provisional) <= 64);

// Feature permission checks. Each tests one feature the session was granted;
// every feature is granted here, so each succeeds without data.
constexpr std::uint32_t permission_checks[] = {201, 301, 401, 402, 403, 494, 801, 1101, 1102, 1501, 1502, 1503, 1601};
} // namespace

GraphicsSystemSession::GraphicsSystemSession(std::uint64_t process_id, horizon::HandleTable& handles,
                                             GraphicsSystemDependencies dependencies)
    : process_id_(process_id), handles_(&handles), dependencies_(std::move(dependencies)) {}

horizon::IpcReply GraphicsSystemSession::close_server() {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    closed_ = true;
    diagnostic_ = "Graphics system server session is closed";
    return {.disposition = horizon::IpcDisposition::handled};
}

horizon::IpcReply GraphicsSystemSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    return handle(message, sender_pid, nullptr);
}

horizon::IpcReply GraphicsSystemSession::request_with_buffers(const horizon::IpcMessage& message,
                                                              std::uint64_t sender_pid,
                                                              const horizon::GuestBuffers& buffers) {
    return handle(message, sender_pid, &buffers);
}

// Syncpoint memory layout written by command 204: offsets into the read-only
// memory for the u64 counter values and the acknowledge summary and bits, and
// into the read-write memory for the client's armed summary and bits.
constexpr std::uint32_t syncpoint_values_offset = 0x40;
constexpr std::uint32_t syncpoint_ro_summary_offset = 0x1040;
constexpr std::uint32_t syncpoint_ro_bits_offset = 0x1100;
constexpr std::uint32_t syncpoint_rw_summary_offset = 0x0;
constexpr std::uint32_t syncpoint_rw_bits_offset = 0x100;

horizon::IpcReply GraphicsSystemSession::attach_syncpoint_memory(const horizon::IpcMessage& message,
                                                                 std::uint32_t type,
                                                                 const horizon::GuestBuffers* buffers) {
    // AttachSyncptMemory(copied read-write and read-only transfer memories;
    // u32 version, u32 syncpoint count, ...). The read-only memory receives a
    // header: version 1, base id 0, the count, then the layout offsets.
    if (!(read(message, 4) & 0x80000000u) || read(message, 8) != 4 || (read(message, 4) & 0x3ff) < 2 || !buffers)
        return {};
    const auto writable = handles_->lookup_as<horizon::TransferMemoryRecord>(read(message, 12));
    const auto readable = handles_->lookup_as<horizon::TransferMemoryRecord>(read(message, 16));
    if (!writable || !readable || readable->range().size < 0x2000 || writable->range().size < 0x1000) {
        diagnostic_ = "Graphics system syncpoint memory is missing or too small";
        return {};
    }
    const auto count = std::min<std::uint32_t>(read(message, 24), (0x1000 - syncpoint_values_offset) / 8);
    std::array<std::byte, 32> header{};
    const std::uint32_t words[] = {1, 0, count, syncpoint_values_offset, syncpoint_rw_summary_offset,
                                   syncpoint_ro_summary_offset, syncpoint_rw_bits_offset, syncpoint_ro_bits_offset};
    for (unsigned i = 0; i != 8; ++i)
        for (unsigned b = 0; b != 4; ++b) header[i * 4 + b] = static_cast<std::byte>((words[i] >> (8 * b)) & 255);
    std::vector<std::byte> values(std::size_t{count} * 8);
    if (!buffers->write_memory(readable->range().address, header) ||
        !buffers->write_memory(readable->range().address + syncpoint_values_offset, values)) {
        diagnostic_ = "Graphics system syncpoint memory is not writable";
        return {};
    }
    syncpoint_request_event_ = std::make_shared<horizon::Event>(false);
    syncpoint_notify_event_ = std::make_shared<horizon::Event>(false);
    const auto request = handles_->insert(syncpoint_request_event_);
    if (!request) return {};
    const auto notify = handles_->insert(syncpoint_notify_event_);
    if (!notify) {
        handles_->close(*request);
        return {};
    }
    horizon::IpcReply reply{horizon::IpcDisposition::handled};
    write(reply.bytes, 0, type);
    write(reply.bytes, 4, 0x80000001);
    write(reply.bytes, 8, 4);
    write(reply.bytes, 12, *request);
    write(reply.bytes, 16, *notify);
    write(reply.bytes, 20, 0);
    reply.size = 24;
    diagnostic_ = nullptr;
    return reply;
}

horizon::IpcReply GraphicsSystemSession::handle(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                                                const horizon::GuestBuffers* buffers) {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    diagnostic_ = "Graphics system request is not implemented";
    if (sender_pid != process_id_) {
        diagnostic_ = "Graphics system session belongs to a different process";
        return {};
    }
    const auto type = read(message, 0) & 0xffff;
    const auto header = read(message, 4);
    if (type < 16 || (read(message, 0) >> 16) != 0) return {};
    const auto command = type - 16;
    // Raw data follows the two header words, or the special header and its
    // process ID and handle words when present.
    unsigned data = 8;
    if (header & 0x80000000u) {
        const auto special = read(message, 8);
        data = 12 + ((special & 1) ? 8 : 0) + 4 * (((special >> 1) & 15) + ((special >> 5) & 15));
    }
    const auto raw_words = header & 0x3ff;
    const auto argument = [&](unsigned index) { return read(message, data + 4 * index); };
    if (command == 0) {
        // Handshake: the client sends a 64-bit token and expects it back.
        if (raw_words < 2) return {};
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {argument(0), argument(1)});
    }
    if (command == 11) {
        // RegisterClientHandle: the client hands over one copied handle once per session.
        if (client_handle_registered_) {
            diagnostic_ = "Graphics system client handle is already registered";
            return {};
        }
        client_handle_registered_ = true;
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {});
    }
    if (command == 204) {
        try {
            return attach_syncpoint_memory(message, type, buffers);
        } catch (const std::bad_alloc&) {
            return {};
        }
    }
    if (command == 202) {
        // OpenSyncptSession: a copied handle to a new syncpoint session.
        std::shared_ptr<SyncpointSession> session;
        try {
            session = std::make_shared<SyncpointSession>(process_id_, *handles_);
        } catch (const std::bad_alloc&) {
            return {};
        }
        const auto handle = handles_->insert(session);
        if (!handle) return {};
        diagnostic_ = nullptr;
        return tipc_handle_reply(type, *handle);
    }
    if (command == 305) {
        // OpenViSession(u64): a copied handle to a new display session.
        std::shared_ptr<DisplaySession> session;
        try {
            session = std::make_shared<DisplaySession>(process_id_, *handles_, dependencies_.wait_refresh);
        } catch (const std::bad_alloc&) {
            return {};
        }
        const auto handle = handles_->insert(session);
        if (!handle) return {};
        diagnostic_ = nullptr;
        return tipc_handle_reply(type, *handle);
    }
    if (command == 497) {
        // OpenGpuSession(u64): a copied handle to a new GPU session.
        std::shared_ptr<GpuSession> session;
        try {
            session = std::make_shared<GpuSession>(process_id_, *handles_);
        } catch (const std::bad_alloc&) {
            return {};
        }
        const auto handle = handles_->insert(session);
        if (!handle) return {};
        diagnostic_ = nullptr;
        return tipc_handle_reply(type, *handle);
    }
    if (command == 1402) {
        // OpenNvsrvSession: a copied handle to a new NVIDIA driver session,
        // the handle at word 3 and the result at word 4.
        std::shared_ptr<horizon::ClientSession> driver;
        try {
            if (dependencies_.open_driver) driver = dependencies_.open_driver();
        } catch (const std::bad_alloc&) {
            return {};
        }
        if (!driver) {
            diagnostic_ = "Graphics system driver session is unavailable";
            return {};
        }
        const auto handle = handles_->insert(driver);
        if (!handle) return {};
        horizon::IpcReply reply{horizon::IpcDisposition::handled};
        write(reply.bytes, 0, type);
        write(reply.bytes, 4, 0x80000001);
        write(reply.bytes, 8, 2);
        write(reply.bytes, 12, *handle);
        write(reply.bytes, 16, 0);
        reply.size = 20;
        diagnostic_ = nullptr;
        return reply;
    }
    if (command == 115) {
        // QueryGrantedFeatures(u64 requested): every requested feature is granted.
        if (raw_words < 2) return {};
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {argument(0), argument(1)});
    }
    if (command == 126) {
        // GetSessionFlag: one session state bit, clear.
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {0});
    }
    if (command == 434 || command == 1111) {
        // AttachGpuAddressSpace(u32 id) and SetFeatureVersion(u32): recorded.
        if (raw_words < 1) return {};
        if (command == 434) attached_address_spaces_.insert(argument(0));
        else feature_version_ = argument(0);
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {});
    }
    if (std::find(std::begin(permission_checks), std::end(permission_checks), command) != std::end(permission_checks)) {
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {});
    }
    for (std::size_t index = 0; index != std::size(provisional); ++index) {
        const auto& entry = provisional[index];
        if (entry.command != command) continue;
        if (!(provisional_warned_ & (std::uint64_t{1} << index))) {
            provisional_warned_ |= std::uint64_t{1} << index;
            armigo::diag::write(armigo::diag::Level::warning,
                "Armigo gsv: provisional handler %s (TIPC %u) answers with an unverified reply\n", entry.name, command);
        }
        auto reply = tipc_reply(type, 0, {});
        write(reply.bytes, 4, 1 + entry.zero_words);
        reply.size = 12 + 4 * entry.zero_words;
        diagnostic_ = nullptr;
        return reply;
    }
    armigo::diag::write(armigo::diag::Level::warning,
        "Armigo gsv: TIPC %u is not implemented (raw words %u: %08x %08x %08x %08x)\n", command, raw_words,
        argument(0), argument(1), argument(2), argument(3));
    return {};
}

const char* GraphicsSystemSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

SyncpointSession::SyncpointSession(std::uint64_t process_id, horizon::HandleTable& handles)
    : process_id_(process_id), handles_(&handles) {}

horizon::IpcReply SyncpointSession::close_server() {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    closed_ = true;
    events_.clear();
    pending_.clear();
    diagnostic_ = "Syncpoint server session is closed";
    return {.disposition = horizon::IpcDisposition::handled};
}

std::shared_ptr<horizon::Event> SyncpointSession::slot_event(std::uint32_t slot) {
    auto& event = events_[slot];
    if (!event) event = std::make_shared<horizon::Event>(false);
    return event;
}

horizon::IpcReply SyncpointSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    diagnostic_ = "Syncpoint request is not implemented";
    if (sender_pid != process_id_) {
        diagnostic_ = "Syncpoint session belongs to a different process";
        return {};
    }
    const auto parsed = parse_tipc(message);
    if (!parsed) return {};
    const auto [type, command, data, raw_words] = *parsed;
    const auto argument = [&](unsigned index) { return read(message, data + 4 * index); };
    try {
        switch (command) {
        case 0:
            // Handshake: the client sends a 64-bit token and expects it back.
            if (raw_words < 2) return {};
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {argument(0), argument(1)});
        case 1:
            // The syncpoint count; the client requires it below 0x400.
            warn_provisional(provisional_warned_, 0, "syncpt", "count", command);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {32});
        case 21: {
            warn_provisional(provisional_warned_, 1, "syncpt", "command 21", command);
            auto reply = tipc_reply(type, 0, {});
            write(reply.bytes, 4, 33);
            reply.size = 12 + 4 * 32;
            diagnostic_ = nullptr;
            return reply;
        }
        case 22: {
            // The event of one slot, which the client multi-waits on.
            if (raw_words < 1) return {};
            const auto handle = handles_->insert(slot_event(argument(0)));
            if (!handle) return {};
            diagnostic_ = nullptr;
            return tipc_handle_reply(type, *handle);
        }
        case 103:
            // QueryCounter(~syncpoint id): the counter value, then a zero word.
            if (raw_words < 1) return {};
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {0, 0, 0});
        case 201: {
            // RequestSlotEvent(slot, ~syncpoint id, u64 value, mode): signal the
            // slot once the counter reaches the value. Mode 0 compares 32-bit
            // wrapping values, mode 1 64-bit values; every counter is zero.
            if (raw_words < 5) return {};
            const auto slot = argument(0);
            const auto value = std::uint64_t{argument(2)} | (std::uint64_t{argument(3)} << 32);
            const bool reached = argument(4) == 0 ? static_cast<std::int32_t>(0u - static_cast<std::uint32_t>(value)) >= 0
                                                  : value == 0;
            auto event = slot_event(slot);
            event->reset();
            if (reached) {
                pending_.erase(slot);
                event->signal();
            } else {
                pending_[slot] = value;
            }
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        }
        case 202:
            // CancelSlotEvent(slot): the client always waits on the slot after a
            // cancel, so the event is signaled whether or not a request was pending.
            if (raw_words < 1) return {};
            pending_.erase(argument(0));
            slot_event(argument(0))->signal();
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        default:
            break;
        }
    } catch (const std::bad_alloc&) {
        return {};
    }
    armigo::diag::write(armigo::diag::Level::warning,
        "Armigo syncpt: TIPC %u is not implemented (raw words %u: %08x %08x %08x %08x)\n", command, raw_words,
        argument(0), argument(1), argument(2), argument(3));
    return {};
}

const char* SyncpointSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

GpuSession::GpuSession(std::uint64_t process_id, horizon::HandleTable& handles)
    : process_id_(process_id), handles_(&handles) {}

horizon::IpcReply GpuSession::close_server() {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    closed_ = true;
    events_ = {};
    diagnostic_ = "GPU server session is closed";
    return {.disposition = horizon::IpcDisposition::handled};
}

std::shared_ptr<horizon::Event> GpuSession::id_event(std::uint32_t id) {
    auto& event = events_[id % events_.size()];
    if (!event) event = std::make_shared<horizon::Event>(false);
    return event;
}

horizon::IpcReply GpuSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    diagnostic_ = "GPU session request is not implemented";
    if (sender_pid != process_id_) {
        diagnostic_ = "GPU session belongs to a different process";
        return {};
    }
    const auto parsed = parse_tipc(message);
    if (!parsed) return {};
    const auto [type, command, data, raw_words] = *parsed;
    const auto argument = [&](unsigned index) { return read(message, data + 4 * index); };
    // The GPU virtual range the SDK's address-space allocator serves: NVN
    // reserves fixed addresses from 128 MiB up to 768 GiB.
    constexpr std::uint64_t va_start = 0x8000000, va_size = 0xc000000000 - va_start;
    try {
        switch (command) {
        case 0:
            if (raw_words < 2) return {};
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {argument(0), argument(1)});
        case 11:
            // The id range [first, last] whose events the SDK uses.
            warn_provisional(provisional_warned_, 0, "gpu", "id range", command);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {0, static_cast<std::uint32_t>(events_.size() - 1)});
        case 12: {
            if (raw_words < 1) return {};
            const auto handle = handles_->insert(id_event(argument(0)));
            if (!handle) return {};
            diagnostic_ = nullptr;
            return tipc_handle_reply(type, *handle);
        }
        case 21:
            // Start(id): the SDK then waits on the id's event; nothing runs, so
            // it completes at once.
            if (raw_words < 1) return {};
            warn_provisional(provisional_warned_, 1, "gpu", "start", command);
            id_event(argument(0))->signal();
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        case 22: {
            // OpenSubSession: another handle to this session for memory and jobs.
            const auto handle = handles_->insert(shared_from_this());
            if (!handle) return {};
            diagnostic_ = nullptr;
            return tipc_handle_reply(type, *handle);
        }
        case 33:
            warn_provisional(provisional_warned_, 2, "gpu", "count 33", command);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {1});
        case 41: {
            // CreateGraphicsDevice(u64 name, u64 flags, u64 label): a nonzero device id.
            for (std::uint32_t id = 1; id != devices_.size(); ++id) {
                if (devices_[id]) continue;
                devices_[id] = true;
                diagnostic_ = nullptr;
                return tipc_reply(type, 0, {id});
            }
            diagnostic_ = "GPU session graphics device ids are exhausted";
            return {};
        }
        case 42: {
            if (raw_words < 1) return {};
            const auto id = argument(0) & 0xff;
            if (!id || !devices_[id]) {
                diagnostic_ = "GPU session graphics device is not open";
                return {};
            }
            devices_[id] = false;
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        }
        case 201:
            // GetUserPartition(id): the address space's user partition.
            warn_provisional(provisional_warned_, 3, "gpu", "user partition", command);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {static_cast<std::uint32_t>(va_start), static_cast<std::uint32_t>(va_start >> 32),
                                        static_cast<std::uint32_t>(va_size), static_cast<std::uint32_t>(va_size >> 32)});
        case 202:
            // GetGsvPartition(id): empty, minimum above maximum.
            warn_provisional(provisional_warned_, 4, "gpu", "graphics system partition", command);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {~0u, ~0u, 0, 0});
        case 203:
            // GetVaRange: kind 1, then the start and size of the range.
            warn_provisional(provisional_warned_, 5, "gpu", "address range", command);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {1, 0, static_cast<std::uint32_t>(va_start), static_cast<std::uint32_t>(va_start >> 32),
                                        static_cast<std::uint32_t>(va_size), static_cast<std::uint32_t>(va_size >> 32)});
        case 23: case 24: case 25: case 34: case 51: case 8001: case 8002: case 8003: case 8004: {
            warn_provisional(provisional_warned_, 6 + (command % 32), "gpu", "zero reply", command);
            auto reply = tipc_reply(type, 0, {});
            write(reply.bytes, 4, 33);
            reply.size = 12 + 4 * 32;
            diagnostic_ = nullptr;
            return reply;
        }
        default:
            break;
        }
    } catch (const std::bad_alloc&) {
        return {};
    }
    armigo::diag::write(armigo::diag::Level::warning,
        "Armigo gpu: TIPC %u is not implemented (raw words %u: %08x %08x %08x %08x)\n", command, raw_words,
        argument(0), argument(1), argument(2), argument(3));
    return {};
}

const char* GpuSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

DisplaySession::DisplaySession(std::uint64_t process_id, horizon::HandleTable& handles,
    std::function<std::optional<std::uint64_t>(std::uint64_t, std::stop_token)> wait_refresh)
    : process_id_(process_id), handles_(&handles), wait_refresh_(std::move(wait_refresh)) {}

DisplaySession::~DisplaySession() {
    refresh_.request_stop();
    if (refresh_.joinable()) refresh_.join();
}

horizon::IpcReply DisplaySession::close_server() {
    refresh_.request_stop();
    if (refresh_.joinable()) refresh_.join();
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    closed_ = true;
    slots_ = {};
    diagnostic_ = "Display server session is closed";
    return {.disposition = horizon::IpcDisposition::handled};
}

void DisplaySession::signal_refresh(std::stop_token stop) {
    std::uint64_t sequence = 0;
    while (!stop.stop_requested()) {
        const auto next = wait_refresh_(sequence, stop);
        if (!next) return;
        sequence = *next;
        const std::lock_guard lock(mutex_);
        for (auto& slot : slots_)
            if (slot.event && slot.refresh) slot.event->signal();
    }
}

horizon::IpcReply DisplaySession::slot_reply(std::uint32_t type, std::uint32_t index) {
    if (index >= slots_.size()) {
        diagnostic_ = "Display native event index is out of range";
        return {};
    }
    auto& slot = slots_[index];
    if (!slot.event) {
        slot.event = std::make_shared<horizon::Event>(false);
        slot.refresh = true;
        if (wait_refresh_ && !refresh_.joinable())
            refresh_ = std::jthread([this](std::stop_token stop) { signal_refresh(stop); });
    }
    const auto handle = handles_->insert(slot.event);
    if (!handle) return {};
    diagnostic_ = nullptr;
    return tipc_handle_reply(type, *handle);
}

horizon::IpcReply DisplaySession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    diagnostic_ = "Display session request is not implemented";
    if (sender_pid != process_id_) {
        diagnostic_ = "Display session belongs to a different process";
        return {};
    }
    const auto parsed = parse_tipc(message);
    if (!parsed) return {};
    const auto [type, command, data, raw_words] = *parsed;
    const auto argument = [&](unsigned index) { return read(message, data + 4 * index); };
    const auto zero_reply = [&](std::size_t index, const char* name) {
        warn_provisional(provisional_warned_, index, "vi2", name, command);
        auto reply = tipc_reply(type, 0, {});
        write(reply.bytes, 4, 33);
        reply.size = 12 + 4 * 32;
        diagnostic_ = nullptr;
        return reply;
    };
    try {
        switch (command) {
        case 0:
            if (raw_words < 2) return {};
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {argument(0), argument(1)});
        case 1: case 3:
            // Command 3 is the display count; the meaning of 1 is unknown.
            warn_provisional(provisional_warned_, command, "vi2", command == 1 ? "command 1" : "display count", command);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {1});
        case 5:
            // Timestamp recorder count, below 8: seven per layer history.
            warn_provisional(provisional_warned_, 5, "vi2", "timestamp recorder count", command);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {7});
        case 11: {
            warn_provisional(provisional_warned_, 11, "vi2", "event 11", command);
            const auto handle = handles_->insert(std::make_shared<horizon::Event>(false));
            if (!handle) return {};
            diagnostic_ = nullptr;
            return tipc_handle_reply(type, *handle);
        }
        case 31:
            // AcquireNativeEvent(u32 index < 8).
            if (raw_words < 1) return {};
            return slot_reply(type, argument(0));
        case 32:
            // DuplicateNativeEvent(copied handle, u32 index): the slot's event again.
            if (raw_words < 1) return {};
            return slot_reply(type, argument(0));
        case 103:
            // QueryDisplayRefreshRate(display name): 60 Hz.
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {60, 0});
        case 104:
            // QueryDisplayHdrStatus(display name): output format 1 (SDR), then
            // the peak luminance in nits.
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {1, 0, 400});
        case 105:
            // QueryDisplayHdrPaperWhite(display name): the reference white in nits.
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {203});
        case 106: {
            // GetBufferMaxResolution(display name): the highest buffer resolution
            // mode of the default display, 4 (2560x1440) as on a television.
            if (raw_words < 2) return {};
            constexpr std::uint64_t default_display = 0x00746c7561666564; // "default"
            const auto name = std::uint64_t{argument(0)} | (std::uint64_t{argument(1)} << 32);
            diagnostic_ = nullptr;
            return name == default_display ? tipc_reply(type, 0, {4, 0}) : tipc_reply(type, 0xe72, {0, 0});
        }
        case 228:
            // CancelClientBuffer: the ready buffer slots, then four zero words.
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {0x7fff, 0, 0, 0, 0, 0});
        case 212:
            // GetLayerBufferBits(u64 layer): the ready buffer slots, which the
            // client polls until some are set.
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {0x7fff, 0});
        case 201:
            // Sent by nn::vi::Layer::OnOpen with a u64 and the two layer ids the
            // client allocated; the client reads only the result. Recorded.
            if (raw_words < 4) return {};
            warn_provisional(provisional_warned_, 63, "vi2", "layer open", command);
            layers_.push_back({std::uint64_t{argument(0)} | (std::uint64_t{argument(1)} << 32), argument(2), argument(3)});
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        case 261: case 262: case 263: case 264: case 265: case 266:
        case 267: case 268: case 269: case 270: case 271: case 272: {
            // (u64 display name, u32 index): 261 binds the slot to the display
            // refresh; the other odd commands bind a change notification, which
            // fires once because nothing changes here; even commands unbind.
            if (raw_words < 3) return {};
            const auto index = argument(2);
            if (index < slots_.size() && slots_[index].event) {
                slots_[index].refresh = command == 261;
                if (command != 261 && (command & 1)) slots_[index].event->signal();
            }
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        }
        case 13: case 51: case 101: case 102: case 107: case 108: case 111: case 112: case 113: case 202:
        case 230: case 231: case 232: case 251: case 252: case 301: case 302: case 351: case 352:
        case 353: case 354: case 355: case 356: case 358: case 359:
            return zero_reply(12 + command % 50, "zero reply");
        default:
            break;
        }
    } catch (const std::bad_alloc&) {
        return {};
    } catch (const std::system_error&) {
        diagnostic_ = "Display refresh thread could not start";
        return {};
    }
    armigo::diag::write(armigo::diag::Level::warning,
        "Armigo vi2: TIPC %u is not implemented (raw words %u: %08x %08x %08x %08x)\n", command, raw_words,
        argument(0), argument(1), argument(2), argument(3));
    return {};
}

const char* DisplaySession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::gsv
