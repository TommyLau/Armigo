// ============================================================================
// Armigo Services - Initial Add-on Content Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/addon_content.h>

#include <array>
#include <utility>

namespace armigo::services::aoc {
namespace {
using horizon::IpcReply;
using horizon::IpcMessage;
using horizon::IpcDisposition;

std::uint32_t read(const IpcMessage& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::uint32_t{std::to_integer<unsigned char>(bytes[offset + i])} << (8 * i);
    return value;
}
void write(IpcMessage& bytes, std::size_t offset, std::uint32_t value, unsigned count = 4) {
    for (unsigned i = 0; i < count; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    write(reply.bytes, 16, 0x4f434653);
    write(reply.bytes, 28, 0xec6be3ff);
    write(reply.bytes, 32, value, width);
    return reply;
}
}  // namespace

AddOnContentSession::AddOnContentSession(std::uint64_t process_id, horizon::HandleTable& handles,
                                         std::shared_ptr<const AddOnContentCatalog> catalog)
    : handles_(handles), root_(std::make_shared<RootObject>(process_id)), catalog_(std::move(catalog)) {}

IpcReply AddOnContentSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply AddOnContentSession::method(const IpcMessage& message,
                                    const horizon::GuestBuffers* buffers) {
    std::size_t cmif = 0;
    if (read(message, 16) == 0x49434653) cmif = 16;
    else if (read(message, 32) == 0x49434653) cmif = 32;
    else if (read(message, 48) == 0x49434653) cmif = 48;
    else return unsupported("Add-on content domain request shape is unsupported");
    const auto command = read(message, cmif + 8);
    // A domain request keeps its response CMIF at byte 32; a non-domain type-6 request
    // carries descriptors before the request CMIF, and its reply has no descriptors, so the
    // response CMIF is at byte 16.
    bool domain = false;
    if (cmif >= 32) {
        const auto header = read(message, cmif - 16);
        const auto objects = header & 0xffffu;
        const auto size = header >> 16;
        domain = objects == 1 && size >= 16;
    }
    const auto response = domain ? std::size_t{32} : std::size_t{16};
    // A successful reply with `payload` output bytes after the response header.
    const auto reply_with = [&](std::size_t payload) {
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = response + 16 + payload;
        write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
        write(reply.bytes, response, 0x4f434653);
        write(reply.bytes, response + 4, 1);
        return reply;
    };
    const auto installed = catalog_ ? catalog_->items.size() : std::size_t{0};
    switch (command) {
    case 2: {  // CountAddOnContent(pid): the number of installed items
        auto reply = reply_with(4);
        write(reply.bytes, response + 16, static_cast<std::uint32_t>(installed));
        return reply;
    }
    case 3: {  // ListAddOnContent(offset, count, pid): their indices, ascending, into the out buffer
        const auto offset = read(message, cmif + 16);
        const auto count = read(message, cmif + 20);
        std::uint32_t written = 0;
        if (offset < installed && count != 0) {
            if (buffers == nullptr || buffers->empty())
                return unsupported("Add-on content list needs its output buffer");
            const auto capacity = (*buffers)[0].size / 4;
            for (auto position = std::size_t{offset};
                 position < installed && written < count && written < capacity; ++position) {
                const auto index = static_cast<std::uint32_t>(catalog_->items[position].data_id - catalog_->base_id);
                std::array<std::byte, 4> bytes{};
                for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<std::byte>((index >> (8 * i)) & 255);
                if (!buffers->write_at(0, std::size_t{written} * 4, bytes))
                    return unsupported("Add-on content list buffer is not writable");
                ++written;
            }
        }
        auto reply = reply_with(4);
        write(reply.bytes, response + 16, written);
        return reply;
    }
    case 5: {  // GetAddOnContentBaseId(pid)
        if (!catalog_) return unsupported("No add-on content base id is configured");
        auto reply = reply_with(8);
        write(reply.bytes, response + 16, static_cast<std::uint32_t>(catalog_->base_id));
        write(reply.bytes, response + 20, static_cast<std::uint32_t>(catalog_->base_id >> 32));
        return reply;
    }
    case 7: {  // PrepareAddOnContent(index, pid): the item's data is already on the host
        const auto index = read(message, cmif + 16);
        if (!catalog_ || index == 0 || !catalog_->find(catalog_->base_id + index))
            return unsupported("Add-on content to prepare is not installed");
        return reply_with(0);
    }
    case 11: {  // NotifyMountAddOnContent(data id): the item's storage has been mounted
        std::uint64_t data_id = read(message, cmif + 16);
        data_id |= std::uint64_t{read(message, cmif + 20)} << 32;
        if (!catalog_ || !catalog_->find(data_id))
            return unsupported("Add-on content reported as mounted is not installed");
        return reply_with(0);
    }
    case 50:  // CheckAddOnContentMountStatus: installed items are never removed while running
        return reply_with(0);
    default:
        return unsupported("Add-on content method is not implemented");
    }
}

IpcReply AddOnContentSession::request_with_buffers(const IpcMessage& message,
                                                  std::uint64_t sender_pid,
                                                  const horizon::GuestBuffers& buffers) {
    // Descriptor-free control and inline requests use the ordinary parser.
    if (buffers.empty()) return request(message, sender_pid);
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != root_->process_id)
        return unsupported("Add-on content session belongs to a different process");
    const auto type = read(message, 0) & 0xffffu;
    if (type != 4 && type != 6)
        return unsupported("Add-on content method or HIPC format is not implemented");
    return method(message, &buffers);
}

IpcReply AddOnContentSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != root_->process_id)
        return unsupported("Add-on content session belongs to a different process");
    const auto type = read(message, 0) & 0xffffu;
    if (type == 4 || type == 6) return method(message, nullptr);
    if (type != 5 && type != 7)
        return unsupported("Add-on content method or HIPC format is not implemented");
    const auto header = read(message, 4);
    if ((header & ~0x3ffU) != 0 || header < 8 || header > (message.size() - 8) / 4 ||
        read(message, 16) != 0x49434653)
        return unsupported("Add-on content control packet is outside the supported format");
    const auto command = read(message, 24);
    // Zero is the actual capacity of this host session, which has no pointer backing.
    if (command == 3) return control_reply(0, 2);
    if (command != 0) return unsupported("Add-on content control command is unsupported");
    if (domain_) return unsupported("Repeated add-on content domain conversion is unsupported");
    const auto reply = control_reply(1, 4);
    domain_ = true; // The sole domain ID names the already-owned root object.
    return reply;
}

std::size_t AddOnContentSession::domain_object_count() const {
    const std::lock_guard lock(mutex_);
    return domain_ ? 1 : 0;
}

const char* AddOnContentSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

}  // namespace armigo::services::aoc
