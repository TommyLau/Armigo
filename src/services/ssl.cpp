// ============================================================================
// Armigo Services - SSL
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/services/ssl.h>

namespace armigo::services::ssl {
namespace {
using namespace horizon;

using namespace horizon::cmif;
// The pointer buffer size the service answers, the process default the
// practical reference answers (as for hwopus and the sound mixer).
constexpr std::uint16_t pointer_buffer_size = 0x8000;



// A reply whose CMIF header is at `out` (16, or 32 after a domain header)
// with `data` bytes after it.
IpcReply reply_with(std::size_t data, std::size_t out = 16) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = out + 16 + data;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, out, sfco);
    return reply;
}

bool control(const IpcMessage& message) {
    const auto type = read(message, 0) & 0xffffu;
    return type == 5 || type == 7;
}

IpcReply control_value(std::uint32_t value, unsigned width) {
    auto reply = reply_with(4);
    write(reply.bytes, 32, width == 2 ? value & 0xffffu : value);
    return reply;
}
} // namespace

Context::Context(std::uint64_t process_id, std::uint32_t version) : process_id_(process_id), version_(version) {}

const char* Context::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

IpcReply Context::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return request_with_buffers(message, sender_pid, GuestBuffers{});
}

IpcReply Context::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                       const GuestBuffers& buffers) {
    if (sender_pid != process_id_) {
        const std::lock_guard lock(mutex_);
        diagnostic_ = "SSL context belongs to another process";
        return {};
    }
    const auto cmif = locate_cmif(message);
    const char* detail = "SSL context request shape is unsupported";
    std::optional<IpcReply> reply;
    if (cmif && !control(message)) reply = command(message, *cmif, 16, &buffers, detail);
    const std::lock_guard lock(mutex_);
    diagnostic_ = reply ? nullptr : detail;
    return reply ? std::move(*reply) : IpcReply{};
}

std::optional<IpcReply> Context::command(const IpcMessage& message, std::size_t cmif, std::size_t out,
                                         const GuestBuffers* buffers, const char*& detail) {
    const std::lock_guard lock(mutex_);
    switch (read(message, cmif + 8)) {
    case 0: {  // SetOption(option, value)
        options_[read(message, cmif + 16)] = static_cast<std::int32_t>(read(message, cmif + 20));
        return reply_with(0, out);
    }
    case 1: {  // GetOption(option) -> value
        const auto found = options_.find(read(message, cmif + 16));
        auto reply = reply_with(4, out);
        write(reply.bytes, out + 16, found == options_.end() ? 0u : static_cast<std::uint32_t>(found->second));
        return reply;
    }
    case 3: {  // GetConnectionCount -> 0: no connection exists offline
        auto reply = reply_with(4, out);
        write(reply.bytes, out + 16, 0);
        return reply;
    }
    case 4:    // ImportServerPki(format; certificates) -> id
    case 5: {  // ImportClientPki(PKCS #12 data, password) -> id
        if (!buffers || buffers->empty()) {
            detail = "SSL PKI import needs its certificate buffer";
            return std::nullopt;
        }
        const auto id = next_pki_++;
        auto reply = reply_with(8, out);
        write(reply.bytes, out + 16, static_cast<std::uint32_t>(id));
        write(reply.bytes, out + 20, static_cast<std::uint32_t>(id >> 32));
        return reply;
    }
    case 2:
    case 100:
        detail = "SSL connections need a network, which does not exist";
        return std::nullopt;
    default:
        detail = "SSL context command is not implemented";
        return std::nullopt;
    }
}

SslSession::SslSession(std::uint64_t process_id) : SslSession(process_id, std::make_shared<Domain>()) {}

SslSession::SslSession(std::uint64_t process_id, std::shared_ptr<Domain> domain)
    : process_id_(process_id), domain_(std::move(domain)) {}

IpcReply SslSession::unsupported(const char* detail) {
    domain_->diagnostic = detail;
    return {};
}

const char* SslSession::diagnostic() const {
    const std::lock_guard lock(domain_->mutex);
    return domain_->diagnostic;
}

std::optional<std::uint32_t> SslSession::interface_version() const {
    const std::lock_guard lock(domain_->mutex);
    return domain_->interface_version;
}

IpcReply SslSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return serve(message, sender_pid, nullptr);
}

IpcReply SslSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                          const GuestBuffers& buffers) {
    return serve(message, sender_pid, &buffers);
}

std::optional<IpcReply> SslSession::command(const IpcMessage& message, std::size_t cmif, std::size_t out,
                                            std::shared_ptr<Context>& opened) {
    switch (read(message, cmif + 8)) {
    case 0:  // CreateContext(SSL version, process id placeholder) -> context
        opened = std::make_shared<Context>(process_id_, read(message, cmif + 16));
        return reply_with(0, out);
    case 5:  // SetInterfaceVersion(version)
        domain_->interface_version = read(message, cmif + 16);
        return reply_with(0, out);
    default:
        domain_->diagnostic = "SSL service command is not implemented";
        return std::nullopt;
    }
}

IpcReply SslSession::serve(const IpcMessage& message, std::uint64_t sender_pid, const GuestBuffers* buffers) {
    const std::lock_guard lock(domain_->mutex);
    domain_->diagnostic = nullptr;
    if (sender_pid != process_id_) return unsupported("SSL session belongs to another process");
    if (control(message)) {
        // QueryPointerBufferSize; ConvertCurrentObjectToDomain (the service is
        // object 1); CloneCurrentObject, another session of the same domain.
        if (read(message, 16) != sfci) return unsupported("SSL control request is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_value(pointer_buffer_size, 2);
        if (command == 0 && !domain_->converted) {
            domain_->converted = true;
            return control_value(1, 4);
        }
        if (command == 2 && domain_->converted) {
            auto reply = control_value(0, 4);
            reply.transferred[0].reset(new SslSession(process_id_, domain_));
            reply.transferred_count = 1;
            return reply;
        }
        return unsupported("SSL control request is unsupported");
    }
    if (!domain_->converted) {
        const auto cmif = locate_cmif(message);
        if (!cmif || read(message, *cmif) != sfci) return unsupported("SSL request shape is unsupported");
        std::shared_ptr<Context> opened;
        auto reply = command(message, *cmif, 16, opened);
        if (!reply) return unsupported(domain_->diagnostic);
        if (opened) {
            reply->transferred[0] = std::move(opened);
            reply->transferred_count = 1;
        }
        return std::move(*reply);
    }
    // A domain message: {type, input objects, data size} then the object id.
    // Type 2 without data releases the object (the service, object 1, last);
    // type 1 carries a CMIF request.
    const auto header = read(message, 16);
    if ((header & 0xff) == 2 && (header >> 16) == 0) {
        const auto id = read(message, 20);
        if (id == 1 && !domain_->root_released) domain_->root_released = true;
        else if (!domain_->objects.erase(id)) return unsupported("SSL domain object is not open");
        return reply_with(0);
    }
    std::optional<std::size_t> cmif;
    for (const std::size_t at : {32u, 48u, 64u})
        if (read(message, at) == sfci) {
            cmif = at;
            break;
        }
    if (!cmif || (read(message, *cmif - 16) & 0xff) != 1) return unsupported("SSL domain request shape is unsupported");
    const auto id = read(message, *cmif - 12);
    if (id == 1) {
        if (domain_->root_released) return unsupported("SSL domain object is not open");
        std::shared_ptr<Context> opened;
        auto reply = command(message, *cmif, 32, opened);
        if (!reply) return unsupported(domain_->diagnostic);
        if (opened) {
            // The context becomes an object of the domain: the reply's domain
            // header counts one output object, whose id follows the data.
            const auto object = domain_->next_object++;
            domain_->objects.emplace(object, std::move(opened));
            write(reply->bytes, 16, 1);
            const auto at = reply->size;
            reply->size += 4;
            write(reply->bytes, 4, static_cast<std::uint32_t>((reply->size - 8) / 4));
            write(reply->bytes, at, object);
        }
        return std::move(*reply);
    }
    const auto found = domain_->objects.find(id);
    if (found == domain_->objects.end()) return unsupported("SSL domain object is not open");
    const char* detail = nullptr;
    auto reply = found->second->command(message, *cmif, 32, buffers, detail);
    return reply ? std::move(*reply) : unsupported(detail);
}

} // namespace armigo::services::ssl
