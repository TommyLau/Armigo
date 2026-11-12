// ============================================================================
// Armigo Services - Switch 2 Graphics System Session
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/common/diagnostics.h>
#include <armigo/services/graphics_system.h>
#include <armigo/horizon/startup_queries.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <system_error>

namespace armigo::services::gsv {
namespace {
using namespace horizon::cmif;

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
// A decoder output image as this server lays it out. The preference holds the
// output code at word 1 (1 and 4 are NV12, 2 and 5 BGRA, others RGBA) and
// (width - 1) | (height - 1) << 16 at word 2. Rows are pitch-linear with a
// 64-byte aligned pitch; NV12's half-height chroma plane follows the luma
// plane, each padded to 128 rows. Clients read images through the descriptor
// command 961 returns, so the layout is this server's choice.
struct DecoderImageLayout {
    bool nv12 = true;
    bool bgra = false;
    std::uint64_t pitch = 0;
    std::uint64_t chroma_offset = 0;
    std::uint64_t size = 0;
};
DecoderImageLayout decoder_image_layout(std::uint32_t code, std::uint32_t extent) {
    DecoderImageLayout layout;
    layout.nv12 = code == 1 || code == 4;
    layout.bgra = code == 2 || code == 5;
    const std::uint64_t width = (extent & 0xffff) + 1, height = (extent >> 16) + 1;
    const auto align = [](std::uint64_t value, std::uint64_t to) { return (value + to - 1) / to * to; };
    layout.pitch = align(width * (layout.nv12 ? 1 : 4), 64);
    layout.chroma_offset = layout.pitch * align(height, 128);
    layout.size = layout.chroma_offset + (layout.nv12 ? layout.pitch * align((height + 1) / 2, 128) : 0);
    return layout;
}
constexpr std::uint64_t decoder_image_pool_alignment = 64 * 128;

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
    {901, 0, "video initialization"}, {903, 0, "video finalization"}, {1001, 0, "second video initialization"},
    {1201, 0, "chat initialization"},
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
    // The client holds the request event's writable side and signals it after
    // arming waits; serving the request stands in for the server's wait on it.
    counters_->attach({writable->range().address, readable->range().address, count, dependencies_.read_memory,
                       dependencies_.write_memory, syncpoint_notify_event_});
    auto writable_request = std::make_shared<horizon::WritableEvent>(syncpoint_request_event_,
        [counters = std::weak_ptr<SyncpointCounters>(counters_),
         event = std::weak_ptr<horizon::Event>(syncpoint_request_event_)] {
            if (const auto request = event.lock()) (void)request->reset();
            if (const auto shared = counters.lock()) shared->serve_requests();
        });
    const auto request = handles_->insert(std::move(writable_request));
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
            session = std::make_shared<SyncpointSession>(process_id_, *handles_, counters_);
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
    if (command == 111) {
        // CreateNativeMemory(u8 mode, u64 size) -> u32 native memory id.
        if (raw_words != 3) return {};
        const std::lock_guard lock(objects_->mutex);
        const auto id = objects_->next_memory++;
        objects_->memories[id] = {std::uint64_t{argument(1)} | std::uint64_t{argument(2)} << 32, 0};
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {id});
    }
    if (command == 112) {
        // DestroyNativeMemory(u32 id).
        if (raw_words != 1) return {};
        const std::lock_guard lock(objects_->mutex);
        objects_->memories.erase(argument(0));
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {});
    }
    if (command == 114) {
        // BindNativeMemory(u32 id, u64 address, u64 size, u64 flags, u64
        // offset): back a native memory with process memory.
        if (raw_words != 9) return {};
        const std::lock_guard lock(objects_->mutex);
        const auto found = objects_->memories.find(argument(0));
        if (found == objects_->memories.end()) return {};
        found->second.address = std::uint64_t{argument(1)} | std::uint64_t{argument(2)} << 32;
        if (!found->second.size) found->second.size = std::uint64_t{argument(3)} | std::uint64_t{argument(4)} << 32;
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {});
    }
    if (command == 911) {
        // InitializeDecoderImagePool(preference[16], u32 count, u32 memory
        // id) -> the id of the first of `count` output images, one image
        // size apart in the memory.
        if (raw_words != 18) return {};
        const auto count = std::max<std::uint32_t>(argument(16), 1);
        const auto layout = decoder_image_layout(argument(1), argument(2));
        const std::lock_guard lock(objects_->mutex);
        if (!objects_->memories.contains(argument(17))) return {};
        const auto first = objects_->next_image;
        objects_->next_image += count;
        for (std::uint32_t index = 0; index < count; ++index)
            objects_->images[first + index] = {argument(17), layout.size * index, argument(1), argument(2)};
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {first});
    }
    if (command == 912) {
        // DestroyDecoderImage(u32 id).
        if (raw_words != 1) return {};
        const std::lock_guard lock(objects_->mutex);
        objects_->images.erase(argument(0));
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {});
    }
    if (command == 961) {
        // GetCpuUserImageDescriptor(u32 image id) -> the 128-byte descriptor
        // a client copies an image out with: kind at +0 (10 NV12, 9 BGRA, 8
        // RGBA, pitch-linear), the plane's offset into the memory at +8 and
        // its pitch at +0x10, NV12's chroma plane at +0x18 and +0x20.
        if (raw_words != 1) return {};
        std::array<std::uint32_t, 32> words{};
        {
            const std::lock_guard lock(objects_->mutex);
            const auto found = objects_->images.find(argument(0));
            if (found == objects_->images.end()) return {};
            const auto layout = decoder_image_layout(found->second.code, found->second.extent);
            words[0] = layout.nv12 ? 10 : layout.bgra ? 9 : 8;
            words[2] = static_cast<std::uint32_t>(found->second.offset);
            words[4] = static_cast<std::uint32_t>(layout.pitch);
            if (layout.nv12) {
                words[6] = static_cast<std::uint32_t>(found->second.offset + layout.chroma_offset);
                words[8] = static_cast<std::uint32_t>(layout.pitch);
            }
        }
        horizon::IpcReply reply{horizon::IpcDisposition::handled};
        write(reply.bytes, 0, type);
        write(reply.bytes, 4, 1 + 32);
        write(reply.bytes, 8, 0);
        for (unsigned index = 0; index < words.size(); ++index) write(reply.bytes, 12 + 4 * index, words[index]);
        reply.size = 12 + 4 * 32;
        diagnostic_ = nullptr;
        return reply;
    }
    if (command == 991) {
        // CalculateDecoderImagePoolBuffer(preference[16], u32 count) -> u64
        // size, u64 alignment of a pool of `count` output images. The 23.0.1
        // program takes 17 raw words.
        if (raw_words != 17) return {};
        const auto layout = decoder_image_layout(argument(1), argument(2));
        const std::uint64_t size = layout.size * std::max<std::uint32_t>(argument(16), 1);
        diagnostic_ = nullptr;
        return tipc_reply(type, 0, {static_cast<std::uint32_t>(size), static_cast<std::uint32_t>(size >> 32),
                                    static_cast<std::uint32_t>(decoder_image_pool_alignment), 0});
    }
    if (command == 906 || command == 1002) {
        // OpenVideoDecoderSessionWithVersion(u8, u64 decoder library
        // version), which the 23.0.1 program accepts with exactly three raw
        // words, and the second video session the SDK opens after 1001 (no
        // input): a copied handle to a new video session each.
        if (raw_words != (command == 906 ? 3u : 0u)) return {};
        std::shared_ptr<VideoSession> session;
        try {
            session = std::make_shared<VideoSession>(process_id_, *handles_, objects_, counters_,
                                                     dependencies_.write_memory, dependencies_.clock);
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

std::uint32_t SyncpointCounters::allocate() {
    const std::lock_guard lock(mutex_);
    const auto id = next_++;
    counters_[id] = 0;
    return id;
}

std::uint64_t SyncpointCounters::value(std::uint32_t id) const {
    const std::lock_guard lock(mutex_);
    const auto found = counters_.find(id);
    return found == counters_.end() ? 0 : found->second;
}

bool SyncpointCounters::reached(std::uint64_t counter, std::uint64_t value, std::uint32_t mode) {
    return mode == 0 ? static_cast<std::int32_t>(static_cast<std::uint32_t>(counter) - static_cast<std::uint32_t>(value)) >= 0
                     : counter >= value;
}

void SyncpointCounters::increment(std::uint32_t id) {
    std::vector<std::shared_ptr<horizon::Event>> signal;
    {
        const std::lock_guard lock(mutex_);
        const auto counter = ++counters_[id];
        std::erase_if(waiters_, [&](const Waiter& waiter) {
            if (waiter.id != id || !reached(counter, waiter.value, waiter.mode)) return waiter.event.expired();
            if (auto event = waiter.event.lock()) signal.push_back(std::move(event));
            return true;
        });
        if (memory_ && id < memory_->count) {
            std::array<std::byte, 8> bytes;
            for (unsigned b = 0; b != 8; ++b) bytes[b] = static_cast<std::byte>((counter >> (8 * b)) & 255);
            (void)memory_->write(memory_->readable + syncpoint_values_offset + std::uint64_t{id} * 8, bytes);
            if (acknowledge()) signal.push_back(memory_->notify);
        }
    }
    for (const auto& event : signal) event->signal();
}

void SyncpointCounters::attach(SharedMemory memory) {
    const std::lock_guard lock(mutex_);
    acknowledged_.clear();
    memory_ = std::move(memory);
    for (const auto& [id, counter] : counters_) {
        if (id >= memory_->count) continue;
        std::array<std::byte, 8> bytes;
        for (unsigned b = 0; b != 8; ++b) bytes[b] = static_cast<std::byte>((counter >> (8 * b)) & 255);
        (void)memory_->write(memory_->readable + syncpoint_values_offset + std::uint64_t{id} * 8, bytes);
    }
}

void SyncpointCounters::serve_requests() {
    std::shared_ptr<horizon::Event> notify;
    {
        const std::lock_guard lock(mutex_);
        if (memory_ && acknowledge()) notify = memory_->notify;
    }
    if (notify) notify->signal();
}

// The client arms a wait by making its read-write bit for a syncpoint differ
// from the read-only one, rereads the value once the bits match again and
// rearms if it still has to wait. An armed bit is acknowledged once its
// counter has moved since that syncpoint's last acknowledgement, or at once
// for one never acknowledged, so a rearm with nothing new waits for the next
// increment. The acknowledge summary follows the client's armed summary.
// True when any bit was acknowledged. Called with the mutex held.
bool SyncpointCounters::acknowledge() {
    const auto read64 = [&](std::uint64_t address, std::uint64_t& value) {
        std::array<std::byte, 8> bytes{};
        if (!memory_->read(address, bytes)) return false;
        value = 0;
        for (unsigned b = 0; b != 8; ++b) value |= std::uint64_t{std::to_integer<std::uint8_t>(bytes[b])} << (8 * b);
        return true;
    };
    const auto write64 = [&](std::uint64_t address, std::uint64_t value) {
        std::array<std::byte, 8> bytes;
        for (unsigned b = 0; b != 8; ++b) bytes[b] = static_cast<std::byte>((value >> (8 * b)) & 255);
        return memory_->write(address, bytes);
    };
    if (!memory_->read || !memory_->write) return false;
    bool acknowledged = false;
    for (std::uint32_t word = 0; word != (memory_->count + 63) / 64; ++word) {
        std::uint64_t armed = 0, current = 0;
        if (!read64(memory_->writable + syncpoint_rw_bits_offset + word * 8, armed) ||
            !read64(memory_->readable + syncpoint_ro_bits_offset + word * 8, current))
            return acknowledged;
        std::uint64_t take = 0;
        for (auto pending = armed ^ current; pending; pending &= pending - 1) {
            const auto bit = static_cast<std::uint32_t>(std::countr_zero(pending));
            const auto id = word * 64 + bit;
            const auto found = counters_.find(id);
            const auto counter = found == counters_.end() ? 0 : found->second;
            const auto last = acknowledged_.find(id);
            if (last != acknowledged_.end() && last->second == counter) continue;
            acknowledged_[id] = counter;
            take |= std::uint64_t{1} << bit;
        }
        if (!take) continue;
        (void)write64(memory_->readable + syncpoint_ro_bits_offset + word * 8, (current & ~take) | (armed & take));
        acknowledged = true;
    }
    std::uint64_t summary = 0;
    if (acknowledged && read64(memory_->writable + syncpoint_rw_summary_offset, summary))
        (void)write64(memory_->readable + syncpoint_ro_summary_offset, summary);
    return acknowledged;
}

bool SyncpointCounters::wait(std::uint32_t id, std::uint64_t value, std::uint32_t mode,
                             std::shared_ptr<horizon::Event> event) {
    const std::lock_guard lock(mutex_);
    const auto found = counters_.find(id);
    if (reached(found == counters_.end() ? 0 : found->second, value, mode)) return true;
    waiters_.push_back({id, value, mode, event});
    return false;
}

void SyncpointCounters::cancel(const std::shared_ptr<horizon::Event>& event) {
    const std::lock_guard lock(mutex_);
    std::erase_if(waiters_, [&](const Waiter& waiter) { return waiter.event.lock() == event; });
}

SyncpointSession::SyncpointSession(std::uint64_t process_id, horizon::HandleTable& handles,
                                   std::shared_ptr<SyncpointCounters> counters)
    : process_id_(process_id), handles_(&handles), counters_(std::move(counters)) {}

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
    static const bool trace = std::getenv("ARMIGO_TRACE_VIDEO") != nullptr;
    if (trace)
        armigo::diag::write(armigo::diag::Level::info, "Armigo syncpt: command %u words %08x %08x %08x %08x %08x\n",
                            command, argument(0), argument(1), argument(2), argument(3), argument(4));
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
        case 103: {
            // QueryCounter(~syncpoint id): the counter value, then a zero word.
            if (raw_words < 1) return {};
            const auto value = counters_->value(~argument(0));
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32), 0});
        }
        case 201: {
            // RequestSlotEvent(slot, ~syncpoint id, u64 value, mode): signal the
            // slot once the counter reaches the value. Mode 0 compares 32-bit
            // wrapping values, mode 1 64-bit values. Only video decoders
            // advance counters here; every other counter stays zero.
            if (raw_words < 5) return {};
            const auto slot = argument(0);
            const auto value = std::uint64_t{argument(2)} | (std::uint64_t{argument(3)} << 32);
            auto event = slot_event(slot);
            event->reset();
            counters_->cancel(event);
            const bool reached = counters_->wait(~argument(1), value, argument(4), event);
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
            counters_->cancel(slot_event(argument(0)));
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

VideoSession::VideoSession(std::uint64_t process_id, horizon::HandleTable& handles,
                           std::shared_ptr<NativeObjects> objects, std::shared_ptr<SyncpointCounters> counters,
                           std::function<bool(std::uint64_t, std::span<const std::byte>)> write_memory,
                           std::function<std::chrono::steady_clock::time_point()> clock)
    : process_id_(process_id), handles_(&handles), objects_(std::move(objects)), counters_(std::move(counters)),
      write_memory_(std::move(write_memory)), clock_(std::move(clock)) {}

VideoSession::~VideoSession() = default;

// Queues an output image for the next decoded frame and returns its
// NativeSyncHandle {~syncpoint, mode 0, u64 value}: the decoder's counter
// reaches the value once that frame is in the image. Images fill in the order
// they are handed out.
std::array<std::uint32_t, 4> VideoSession::hand_out(std::uint32_t image) {
    if (!syncpoint_) syncpoint_ = counters_->allocate();
    free_images_.push_back(image);
    const auto value = ++handed_;
    return {~*syncpoint_, 0, static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32)};
}

// Advances the decoder's counter to the last value handed out, so every
// NativeSyncHandle of this decoder reads as reached.
void VideoSession::complete_handed() {
    if (!syncpoint_) return;
    while (counters_->value(*syncpoint_) < handed_) counters_->increment(*syncpoint_);
}

std::int64_t VideoSession::now_us() const {
    const auto now = clock_ ? clock_() : std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
}

// Starts or corrects the movie clock from a frame the client just acquired.
// The clock never runs behind a frame the client already shows.
void VideoSession::anchor_clock(std::uint64_t pts_us) {
    const auto now = now_us();
    const auto offset = now - static_cast<std::int64_t>(pts_us);
    if (!clock_offset_us_ || now - last_acquire_us_ > pause_gap_us || pts_us < last_acquired_pts_us_ ||
        offset < *clock_offset_us_ || offset - *clock_offset_us_ > max_catch_up_us)
        clock_offset_us_ = offset;
    last_acquire_us_ = now;
    last_acquired_pts_us_ = pts_us;
}

// The pts the movie's sound has reached, once the client shows frames and
// while it keeps acquiring them.
std::optional<std::uint64_t> VideoSession::due_pts_us() const {
    if (!clock_offset_us_) return std::nullopt;
    const auto now = now_us();
    const auto due = now - *clock_offset_us_;
    if (due < 0 || now - last_acquire_us_ > pause_gap_us) return std::nullopt;
    return static_cast<std::uint64_t>(due);
}

// A movie player that shows one decoded frame per frame it draws and never
// skips one falls behind its sound, which plays in real time, whenever it
// draws below the movie's frame rate. Decoded frames that a later waiting
// frame has already overtaken are dropped, at most thirty in a row.
void VideoSession::drop_overtaken_frames() {
    const auto due = due_pts_us();
    if (!due) {
        dropped_in_a_row_ = 0;
        return;
    }
    while (pending_frames_.size() >= 2 && dropped_in_a_row_ < max_dropped_in_a_row) {
        const auto& [frame, pts_ns, end] = pending_frames_[0];
        const auto next_pts_us = std::get<1>(pending_frames_[1]) / 1000;
        if (end || !frame || next_pts_us <= pts_ns / 1000 || next_pts_us > *due) break;
        pending_frames_.pop_front();
        ++dropped_;
        ++dropped_in_a_row_;
    }
}

// Writes decoded frames into free images in hand-out order, advances the
// decoder's counter for each and queues its output information.
bool VideoSession::place_frames(const char*& failure) {
    while (!pending_frames_.empty() && !free_images_.empty()) {
        drop_overtaken_frames();
        dropped_in_a_row_ = 0;
        const auto& [frame, pts_ns, end] = pending_frames_.front();
        const auto image = free_images_.front();
        // A stream whose last input held no picture ends with an empty frame:
        // it takes an image, unwritten, and reports the configured size.
        if (!frame) {
            outputs_.push_back({image, width_, height_, pts_ns, end});
            armigo::diag::write(armigo::diag::Level::info,
                                "Armigo video: end of stream without a picture (pts %llu us) in image %u\n",
                                static_cast<unsigned long long>(pts_ns / 1000), image);
            free_images_.pop_front();
            pending_frames_.pop_front();
            counters_->increment(*syncpoint_);
            continue;
        }
        std::uint64_t address = 0;
        DecoderImageLayout layout;
        {
            const std::lock_guard lock(objects_->mutex);
            const auto found = objects_->images.find(image);
            if (found == objects_->images.end()) {
                failure = "Video decoder output image is not a decoder image";
                return false;
            }
            const auto memory = objects_->memories.find(found->second.memory);
            if (memory == objects_->memories.end() || !memory->second.address) {
                failure = "Video decoder output image memory is not bound";
                return false;
            }
            layout = decoder_image_layout(found->second.code, found->second.extent);
            address = memory->second.address + found->second.offset;
        }
        if (!layout.nv12) {
            failure = "Video decoder output images other than NV12 are not implemented";
            return false;
        }
        // The picture's planes, row by row, in the image's pitch-linear layout.
        std::vector<std::byte> bytes(layout.size);
        const auto rows = std::min<std::uint64_t>(frame->height, layout.chroma_offset / layout.pitch);
        const auto width = std::min<std::uint64_t>(frame->width, layout.pitch);
        for (std::uint64_t row = 0; row < rows && (row + 1) * frame->luma_stride <= frame->luma.size(); ++row)
            std::memcpy(bytes.data() + row * layout.pitch, frame->luma.data() + row * frame->luma_stride, width);
        const auto chroma_rows = std::min<std::uint64_t>((frame->height + 1) / 2,
                                                         (layout.size - layout.chroma_offset) / layout.pitch);
        for (std::uint64_t row = 0; row < chroma_rows && (row + 1) * frame->chroma_stride <= frame->chroma.size(); ++row)
            std::memcpy(bytes.data() + layout.chroma_offset + row * layout.pitch,
                        frame->chroma.data() + row * frame->chroma_stride, width);
        if (!write_memory_ || !write_memory_(address, bytes)) {
            failure = "Video decoder output image is not writable";
            return false;
        }
        outputs_.push_back({image, frame->width, frame->height, pts_ns, end});
        // Progress for pacing checks: the first frame, every 240th and the
        // stream's end, with the time since the first.
        const auto now = std::chrono::steady_clock::now();
        if (!placed_++) first_placed_ = now;
        if (placed_ == 1 || placed_ % 240 == 0 || end)
            armigo::diag::write(armigo::diag::Level::info,
                                "Armigo video: frame %llu (%ux%u, pts %llu us%s) in image %u at %lld ms, %llu dropped\n",
                                static_cast<unsigned long long>(placed_), frame->width, frame->height,
                                static_cast<unsigned long long>(pts_ns / 1000), end ? ", end of stream" : "", image,
                                static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now - first_placed_).count()),
                                static_cast<unsigned long long>(dropped_));
        free_images_.pop_front();
        pending_frames_.pop_front();
        counters_->increment(*syncpoint_);
    }
    return true;
}

// The ready event tells the client the decoder takes another input: while it
// runs and fewer than four decoded frames wait for an output image.
void VideoSession::update_ready() {
    if (!ready_event_) return;
    if (running_ && pending_frames_.size() < 4) ready_event_->signal();
    else ready_event_->reset();
}

horizon::IpcReply VideoSession::close_server() {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    closed_ = true;
    diagnostic_ = "Video decoder server session is closed";
    return {.disposition = horizon::IpcDisposition::handled};
}

horizon::IpcReply VideoSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    return logged(handle(message, sender_pid, nullptr));
}

horizon::IpcReply VideoSession::request_with_buffers(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                                                     const horizon::GuestBuffers& buffers) {
    return logged(handle(message, sender_pid, &buffers));
}

// A refused request names its reason in the log too: the launch failure names
// only the session and the message type.
horizon::IpcReply VideoSession::logged(horizon::IpcReply reply) const {
    if (reply.disposition == horizon::IpcDisposition::unsupported)
        if (const char* text = diagnostic())
            armigo::diag::write(armigo::diag::Level::warning, "Armigo video: %s\n", text);
    return reply;
}

const char* VideoSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

horizon::IpcReply VideoSession::unsupported_command(std::uint32_t command, const horizon::IpcMessage& message) {
    std::snprintf(diagnostic_text_, sizeof diagnostic_text_,
                  "Video decoder command %u (header %08x %08x) is not implemented", command, read(message, 0),
                  read(message, 4));
    diagnostic_ = diagnostic_text_;
    return {};
}

horizon::IpcReply VideoSession::handle(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                                       const horizon::GuestBuffers* buffers) {
    const std::lock_guard lock(mutex_);
    if (closed_) return {.disposition = horizon::IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    if (sender_pid != process_id_) {
        diagnostic_ = "Video decoder session belongs to a different process";
        return {};
    }
    // TIPC with buffer descriptors: raw data follows the header, the special
    // header and its words, and every pointer (2 words) and send, receive or
    // exchange descriptor (3 words).
    const auto first = read(message, 0), header = read(message, 4);
    const auto type = first & 0xffff;
    if (type < 16) return unsupported_command(type, message);
    const auto command = type - 16;
    unsigned data = 8;
    if (header & 0x80000000u) {
        const auto special = read(message, 8);
        data = 12 + ((special & 1) ? 8 : 0) + 4 * (((special >> 1) & 15) + ((special >> 5) & 15));
    }
    data += 8 * ((first >> 16) & 15) + 12 * (((first >> 20) & 15) + ((first >> 24) & 15) + ((first >> 28) & 15));
    const auto raw_words = header & 0x3ff;
    const auto argument = [&](unsigned index) { return read(message, data + 4 * index); };
    // ARMIGO_TRACE_VIDEO=1 prints every request with its first raw words.
    static const bool trace = std::getenv("ARMIGO_TRACE_VIDEO") != nullptr;
    if (trace)
        armigo::diag::write(armigo::diag::Level::info,
                            "Armigo video: session %p command %u words %08x %08x %08x %08x (frames %zu waiting, "
                            "%zu images free, %zu outputs)\n",
                            static_cast<const void*>(this), command, argument(0), argument(1), argument(2),
                            argument(3), pending_frames_.size(), free_images_.size(), outputs_.size());
    if (!warned_) {
        warned_ = true;
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo video: provisional video decoder session follows the practical reference\n");
    }
    try {
        switch (command) {
        case 1: {  // CreateVideoDecoderService: a copied handle to a new session of this class.
            auto session =
                std::make_shared<VideoSession>(process_id_, *handles_, objects_, counters_, write_memory_, clock_);
            const auto handle = handles_->insert(session);
            if (!handle) return {};
            diagnostic_ = nullptr;
            return tipc_handle_reply(type, *handle);
        }
        case 3: {  // The submission-ready event.
            if (!ready_event_) ready_event_ = std::make_shared<horizon::Event>(false);
            const auto handle = handles_->insert(ready_event_);
            if (!handle) return {};
            diagnostic_ = nullptr;
            return tipc_handle_reply(type, *handle);
        }
        case 200: case 201: case 202: case 210:
            // Initialization with the client's work-buffer transfer memory and
            // its parameters; the decoder keeps its state in host memory.
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        case 101: {  // Configure(codec name; u32 width, u32 height, u32 rate, u32)
            if (!buffers || buffers->empty()) return unsupported_command(command, message);
            std::array<std::byte, 64> name{};
            const auto size = std::min<std::size_t>((*buffers)[0].size, name.size());
            if (!buffers->read(0, std::span(name).first(size))) return unsupported_command(command, message);
            codec_.assign(reinterpret_cast<const char*>(name.data()),
                          strnlen(reinterpret_cast<const char*>(name.data()), size));
            if (raw_words >= 4) {
                width_ = argument(0);
                height_ = argument(1);
            }
            armigo::diag::write(armigo::diag::Level::info, "Armigo video: decoder configured for %s %ux%u\n",
                                codec_.c_str(), width_, height_);
            decoder_.reset();
            decoder_failure_.clear();
            clock_offset_us_.reset();
            if (codec_ == "video/av1") {
                std::string failure;
                decoder_ = host1x::make_av1_decoder(failure);
                if (!decoder_) decoder_failure_ = failure;
            } else {
                decoder_failure_ = "Video codec " + codec_ + " is not implemented";
            }
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        }
        case 6:  // Start: the decoder takes input.
            running_ = true;
            update_ready();
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        case 207: case 8: case 115:  // Stop, flush, finalize: queued input and frames are dropped.
            if (command != 8) running_ = false;
            pending_frames_.clear();
            // Stop and finalize also complete every fence handed out: the
            // client waits for the images it released after the stream's end
            // before it closes the movie, and no picture will fill them. A
            // flush keeps them, as a player may flush between movies and take
            // each completed fence as a decoded frame.
            if (command != 8) complete_handed();
            clock_offset_us_.reset();
            dropped_in_a_row_ = 0;
            update_ready();
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        case 311: case 314: {
            // RegisterBuffer(u32 image id) and ReleaseBuffer(u32 image id, u32,
            // NativeSyncHandle of the client's reads): the image takes a later
            // decoded frame; the reply is its NativeSyncHandle. The client's
            // read fence is not waited for, so a frame may land in an image
            // its GPU work still reads.
            if (raw_words < 1) return unsupported_command(command, message);
            const auto handle = hand_out(argument(0));
            const char* failure = nullptr;
            if (!place_frames(failure)) {
                diagnostic_ = failure;
                return {};
            }
            update_ready();
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {handle[0], handle[1], handle[2], handle[3]});
        }
        case 212:  // UnregisterBuffer(u32 image id)
            if (raw_words < 1) return unsupported_command(command, message);
            std::erase(free_images_, argument(0));
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {});
        case 109: case 400: {
            // SubmitInputBuffer(one temporal unit; u64 pts in microseconds, u8
            // end of stream) -> u64 input id; 400 adds a frame information
            // buffer the decoder does not need. Decoded frames wait for
            // output images.
            if (raw_words < 3) return unsupported_command(command, message);
            if (!decoder_) {
                std::snprintf(diagnostic_text_, sizeof diagnostic_text_, "%s", decoder_failure_.c_str());
                diagnostic_ = diagnostic_text_;
                return {};
            }
            // A player may submit while the ready event is down; with twelve
            // decoded frames waiting for images the unit is refused as busy,
            // which the client retries after the ready event.
            if (pending_frames_.size() >= busy_frames) {
                if (!busy_logged_) {
                    busy_logged_ = true;
                    armigo::diag::write(armigo::diag::Level::info,
                                        "Armigo video: decoder busy, a unit refused with %zu frames waiting\n",
                                        pending_frames_.size());
                }
                diagnostic_ = nullptr;
                return tipc_reply(type, decoder_busy_result, {});
            }
            // A unit without a buffer is empty, as an end-of-stream unit may be.
            std::vector<std::uint8_t> unit(buffers && !buffers->empty() ? (*buffers)[0].size : 0);
            if (!unit.empty() && !buffers->read(0, std::as_writable_bytes(std::span(unit))))
                return unsupported_command(command, message);
            const auto pts_us = std::uint64_t{argument(0)} | std::uint64_t{argument(1)} << 32;
            const bool end = (argument(2) & 0xff) != 0;
            std::vector<std::shared_ptr<const host1x::DecodedFrame>> shown;
            std::string decode_failure;
            if (!unit.empty() && !decoder_->decode(unit, shown, decode_failure)) {
                // The movie player ends a stream with time -1 and a unit that
                // holds no picture: the stream ends without one. Any other
                // unit that does not decode stops the session.
                if (!end) {
                    std::snprintf(diagnostic_text_, sizeof diagnostic_text_, "%s", decode_failure.c_str());
                    diagnostic_ = diagnostic_text_;
                    return {};
                }
                armigo::diag::write(armigo::diag::Level::info,
                                    "Armigo video: end-of-stream unit of %zu bytes holds no picture (%s)\n", unit.size(),
                                    decode_failure.c_str());
                shown.clear();
            }
            for (std::size_t index = 0; index < shown.size(); ++index)
                pending_frames_.emplace_back(shown[index], pts_us * 1000, end && index + 1 == shown.size());
            // The client stops at the frame marked end of stream: the last
            // waiting frame carries the mark, or an empty frame when none waits.
            if (end && shown.empty()) {
                if (pending_frames_.empty()) pending_frames_.emplace_back(nullptr, pts_us * 1000, true);
                else std::get<2>(pending_frames_.back()) = true;
            }
            const char* failure = nullptr;
            if (!place_frames(failure)) {
                diagnostic_ = failure;
                return {};
            }
            update_ready();
            const auto id = ++submitted_;
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {static_cast<std::uint32_t>(id), static_cast<std::uint32_t>(id >> 32)});
        }
        case 213: {
            // GetOutputInfo: the next finished frame (image id, width, height,
            // width, height, colour space, pts in ns, end of stream), or
            // 0x1c286f while none is finished.
            if (outputs_.empty()) {
                diagnostic_ = nullptr;
                return tipc_reply(type, 0x1c286f, {});
            }
            const auto output = outputs_.front();
            outputs_.pop_front();
            anchor_clock(output.pts_ns / 1000);
            diagnostic_ = nullptr;
            return tipc_reply(type, 0, {output.image, output.width, output.height, output.width, output.height, 0,
                                        static_cast<std::uint32_t>(output.pts_ns),
                                        static_cast<std::uint32_t>(output.pts_ns >> 32), output.end ? 1u : 0u, 0, 0,
                                        0, 0, 0, 0, 0});
        }
        default:
            return unsupported_command(command, message);
        }
    } catch (const std::bad_alloc&) {
        return {};
    }
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
