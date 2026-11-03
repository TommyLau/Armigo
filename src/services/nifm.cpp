// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/nifm.h>

#include <array>
#include <new>
#include <optional>
#include <vector>

namespace armigo::services::nifm {
namespace {
using namespace horizon;
constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
// The pointer-buffer capacity each nifm session reports.
constexpr std::uint32_t pointer_capacity = 0x1000;
// A request's state: free, or on hold while it is evaluated.
constexpr std::uint32_t state_free = 1;
// Submit's evaluation found no network profile to connect with.
constexpr std::uint32_t result_no_candidate = 0x8986e;
// The network-connection library applet and its mode for that result.
constexpr std::uint32_t network_connection_applet = 15;
constexpr std::uint32_t applet_mode_all_foreground = 0;

std::uint32_t read(const IpcMessage& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i != 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(bytes[offset + i])) << (8 * i);
    return value;
}
void write(IpcMessage& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

// What one method produced: a result, output words, a new object, copied objects.
struct Outcome {
    std::uint32_t result = 0;
    std::vector<std::uint32_t> output;
    std::shared_ptr<Object> created;
    std::array<std::shared_ptr<KernelObject>, 2> copies{};
    std::size_t copy_count = 0;
};

// Runs `command` on `object`; nullopt with `diagnostic` set when unsupported.
// `raw` is the request's raw input after the CMIF header and `raw_bytes` its size.
std::optional<Outcome> run(Object& object, ProcessState& process, std::uint32_t command, const IpcMessage& message,
                           std::size_t raw, std::size_t raw_bytes, const GuestBuffers* buffers,
                           const char*& diagnostic) {
    Outcome outcome;
    switch (object.kind) {
    case Object::Kind::static_service:
        if (command != 4 && command != 5) return std::nullopt;
        // CreateGeneralServiceOld (4) or CreateGeneralService (5, reserved u64
        // and the sender PID, which the transport already carries).
        outcome.created = std::make_shared<Object>();
        outcome.created->kind = Object::Kind::general_service;
        {
            const std::lock_guard lock(process.mutex);
            outcome.created->client_id = process.next_client_id++;
        }
        return outcome;
    case Object::Kind::general_service:
        if (command == 1) {  // GetClientId: the client id into the output buffer
            if (!buffers || buffers->size() != 1 || !(*buffers)[0].output || (*buffers)[0].size < 4) {
                diagnostic = "Network GetClientId needs one output buffer of at least 4 bytes";
                return std::nullopt;
            }
            std::array<std::byte, 4> id{};
            for (unsigned i = 0; i != 4; ++i) id[i] = static_cast<std::byte>((object.client_id >> (8 * i)) & 255);
            if (!buffers->write(0, id)) {
                diagnostic = "Network GetClientId buffer is not writable";
                return std::nullopt;
            }
            return outcome;
        }
        if (command == 4) {  // CreateRequest(requirement preset)
            if (raw_bytes < 4) {
                diagnostic = "Network CreateRequest needs its requirement preset";
                return std::nullopt;
            }
            outcome.created = std::make_shared<Object>();
            outcome.created->kind = Object::Kind::request;
            outcome.created->requirement_preset = static_cast<std::int32_t>(read(message, raw));
            outcome.created->events[0] = std::make_shared<Event>(false);
            outcome.created->events[1] = std::make_shared<Event>(false);
            return outcome;
        }
        return std::nullopt;
    case Object::Kind::request:
        if (command == 0) {  // GetRequestState
            outcome.output.push_back(object.state);
            return outcome;
        }
        if (command == 2) {  // GetSystemEventReadableHandles: two copied readable events
            outcome.copies = {object.events[0], object.events[1]};
            outcome.copy_count = 2;
            return outcome;
        }
        if (command == 4) {
            // Submit: the request is evaluated at once. With no network profile
            // there is no candidate, so it fails, returns to free and signals the
            // state-change event (the first one).
            object.submitted = true;
            object.state = state_free;
            object.result = result_no_candidate;
            object.events[0]->signal();
            return outcome;
        }
        if (command == 1) {  // GetResult: the last evaluation's result
            if (!object.submitted) {
                diagnostic = "Network request result before a submission is not implemented";
                return std::nullopt;
            }
            outcome.result = object.result;
            return outcome;
        }
        if (command == 21) {
            // GetAppletInfo(theme color): the applet that handles the result, its
            // mode, and its argument records [u32 size][data] up to 0xffffffff.
            if (!object.submitted || object.result != result_no_candidate) {
                diagnostic = "Network applet information exists only for a failed request";
                return std::nullopt;
            }
            if (raw_bytes < 4 || !buffers || buffers->size() != 1 || !(*buffers)[0].output) {
                diagnostic = "Network GetAppletInfo needs its theme color and one output buffer";
                return std::nullopt;
            }
            std::vector<std::byte> records;
            const auto put = [&](std::uint32_t value) {
                for (unsigned i = 0; i != 4; ++i) records.push_back(static_cast<std::byte>((value >> (8 * i)) & 255));
            };
            // Common library-applet arguments: version 1, size 0x20, API version 0,
            // the theme color, no startup sound, and a zero system tick.
            put(0x20);
            put(1);
            put(0x20);
            put(0);
            put(read(message, raw));
            for (unsigned i = 0; i != 4; ++i) put(0);
            put(4);
            put(3);
            put(0xffffffff);
            if ((*buffers)[0].size < records.size() || !buffers->write(0, records)) {
                diagnostic = "Network GetAppletInfo buffer is too small or not writable";
                return std::nullopt;
            }
            outcome.output = {network_connection_applet, applet_mode_all_foreground,
                              static_cast<std::uint32_t>(records.size())};
            return outcome;
        }
        return std::nullopt;
    }
    return std::nullopt;
}
} // namespace

Session::Session(std::uint64_t process_id)
    : process_(std::make_shared<ProcessState>(process_id)), object_(std::make_shared<Object>()) {}

Session::Session(std::shared_ptr<ProcessState> process, std::shared_ptr<Object> object)
    : process_(std::move(process)), object_(std::move(object)) {}

horizon::IpcReply Session::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    return handle(message, sender_pid, nullptr);
}

horizon::IpcReply Session::request_with_buffers(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                                                const horizon::GuestBuffers& buffers) {
    return handle(message, sender_pid, buffers.empty() ? nullptr : &buffers);
}

horizon::IpcReply Session::handle(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                                  const horizon::GuestBuffers* buffers) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = "Network service method or packet is unsupported";
    if (sender_pid != process_->process_id) {
        diagnostic_ = "Network service session belongs to a different process";
        return {};
    }
    try {
        const auto type = read(message, 0) & 0xffffu;
        if (type == 5 || type == 7) {
            if (read(message, 16) != sfci) return {};
            const auto command = read(message, 24);
            IpcReply reply{IpcDisposition::handled};
            reply.size = 8 + 4 * 9;
            write(reply.bytes, 4, 9);
            write(reply.bytes, 16, sfco);
            if (command == 3) {  // QueryPointerBufferSize
                write(reply.bytes, 32, pointer_capacity);
            } else if (command == 0 && !domain_) {  // ConvertToDomain: the port object is object 1
                domain_ = true;
                objects_.emplace(1, object_);
                write(reply.bytes, 32, 1);
            } else {
                return {};
            }
            diagnostic_ = nullptr;
            return reply;
        }
        if (type != 4 && type != 6) return {};
        if (domain_ && (read(message, 16) & 0xff) == 2 && read(message, 32) != sfci) {
            // A domain close releases one handed-out object; the reply has no CMIF payload.
            const auto id = read(message, 20);
            if (id == 1 || !objects_.erase(id)) {
                diagnostic_ = "Network domain close names no handed-out object";
                return {};
            }
            IpcReply reply{IpcDisposition::handled};
            reply.size = 32;
            write(reply.bytes, 4, 6);
            write(reply.bytes, 16, sfco);
            diagnostic_ = nullptr;
            return reply;
        }
        const auto located = locate_cmif(message);
        if (!located || read(message, *located) != sfci) return {};
        const auto cmif = *located;
        std::shared_ptr<Object> target = object_;
        std::size_t raw_bytes = 0;
        if (domain_) {
            if (cmif < 32 || (read(message, cmif - 16) & 0xff) != 1) return {};
            const auto found = objects_.find(read(message, cmif - 12));
            if (found == objects_.end()) {
                diagnostic_ = "Network domain request names no object";
                return {};
            }
            target = found->second;
            const auto size = read(message, cmif - 16) >> 16;
            raw_bytes = size >= 16 ? size - 16 : 0;
        } else {
            // Raw words are 4 of alignment padding, the 4-word CMIF header, then the inputs.
            const auto words = read(message, 4) & 0x3ffu;
            raw_bytes = words >= 8 ? (words - 8) * 4 : 0;
        }
        const auto command = read(message, cmif + 8);
        auto outcome = run(*target, *process_, command, message, cmif + 16, raw_bytes, buffers, diagnostic_);
        if (!outcome) return {};
        IpcReply reply{IpcDisposition::handled};
        const std::size_t header = domain_ ? 32 : 16;
        std::size_t raw = header + 16;
        write(reply.bytes, header, sfco);
        write(reply.bytes, header + 8, outcome->result);
        for (const auto value : outcome->output) {
            write(reply.bytes, raw, value);
            raw += 4;
        }
        if (outcome->created) {
            if (domain_) {
                const auto id = next_object_++;
                objects_.emplace(id, outcome->created);
                write(reply.bytes, 16, 1);  // one output object
                write(reply.bytes, raw, id);
                raw += 4;
            } else {
                reply.transferred[0] = std::make_shared<Session>(process_, outcome->created);
                reply.transferred_count = 1;
            }
        }
        for (std::size_t i = 0; i != outcome->copy_count; ++i) reply.transferred[i] = outcome->copies[i];
        if (outcome->copy_count) {
            reply.transferred_count = outcome->copy_count;
            reply.transferred_as_copy = true;
        }
        // A non-domain reply keeps two padding words, as does a domain reply
        // carrying an object id.
        reply.size = domain_ && !outcome->created ? raw : raw + 8;
        write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
        diagnostic_ = nullptr;
        return reply;
    } catch (const std::bad_alloc&) {
        diagnostic_ = "Network service allocation failed";
        return {};
    }
}

const char* Session::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

std::shared_ptr<const Object> Session::object(std::uint32_t id) const {
    const std::lock_guard lock(mutex_);
    if (!domain_) return id == 1 ? object_ : nullptr;
    const auto found = objects_.find(id);
    return found == objects_.end() ? nullptr : found->second;
}

std::size_t Session::domain_object_count() const {
    const std::lock_guard lock(mutex_);
    return objects_.size();
}

} // namespace armigo::services::nifm
