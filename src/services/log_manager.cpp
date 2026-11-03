// ============================================================================
// Armigo Services - Initial Log Manager Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/log_manager.h>

#include <limits>
#include <new>
#include <stdexcept>

namespace armigo::services::lm {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
constexpr std::uint32_t control_interface = 0xec6be3ff;
constexpr std::uint32_t log_service_interface = 0xe1d7f748;

std::uint32_t read(const IpcMessage& message, std::size_t offset, unsigned size = 4) {
    std::uint32_t value = 0;
    for (unsigned byte = 0; byte < size; ++byte)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + byte])} << (byte * 8);
    return value;
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value, unsigned size = 4) {
    for (unsigned byte = 0; byte < size; ++byte)
        message[offset + byte] = static_cast<std::byte>((value >> (byte * 8)) & 255);
}

// Supported packets have no descriptors or transferred handles. The raw
// count includes the firmware's sixteen-byte CMIF alignment allowance.
// Request4/6 and control5/7 share their respective firmware dispatch arms.
bool packet_shape(const IpcMessage& message, std::uint32_t type, bool pid,
                  std::uint32_t minimum_words) {
    const auto second = read(message, 4);
    const auto words = second & 0x3ffu;
    const std::size_t raw_start = pid ? 20 : 8;
    // The wire type is the leading halfword; the following halfword carries
    // the HIPC message length, which startup requests do not constrain.
    const auto actual_type = read(message, 0, 2);
    return (actual_type == type || actual_type == type + 2) &&
           (second & ~0x3ffu) == (pid ? 0x80000000u : 0u) &&
           (!pid || read(message, 8) == 1) && words >= minimum_words &&
           words <= (message.size() - raw_start) / 4;
}

void cmif_header(IpcMessage& message, std::size_t offset, std::uint32_t interface_id) {
    write(message, offset, sfco);
    // Version and Result are zero from value initialization. This firmware
    // writes an interface identifier in the fourth word, not the input token.
    write(message, offset + 12, interface_id);
}

IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44; // HIPC header8 + raw data36 (9 words).
    write(reply.bytes, 4, 9);
    cmif_header(reply.bytes, 16, control_interface);
    write(reply.bytes, 32, value, width);
    return reply;
}
} // namespace

LogManagerSession::LogManagerSession(std::uint64_t process_id)
    : process_id_(process_id),
      root_(std::make_shared<DomainObject>(DomainObject{ObjectKind::root, process_id, {}})) {}

horizon::IpcReply LogManagerSession::unsupported(const char* message) {
    diagnostic_ = message;
    return {};
}

horizon::IpcReply LogManagerSession::request(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_)
        return unsupported("Log-manager session belongs to a different process");
    try {
        const auto type = read(message, 0, 2);
        if (type == 5 || type == 7) return control(message);
        if (type == 4 || type == 6) return domain(message, sender_pid);
        return unsupported("Log-manager HIPC message type or descriptors are unsupported");
    } catch (const std::bad_alloc&) {
        return unsupported("Log-manager object allocation failed; wire failure unsupported");
    } catch (const std::length_error&) {
        return unsupported("Log-manager object storage exhausted; wire failure unsupported");
    }
}

horizon::IpcReply LogManagerSession::control(const horizon::IpcMessage& message) {
    if (!packet_shape(message, 5, false, 8) || read(message, 16) != sfci)
        return unsupported("Log-manager control packet is outside the supported wire format");
    // The firmware parses the same CMIF header for types5/7. Type7 carries
    // a temporary request context in its fourth word; controls0/3 do not use
    // it and the response writes the fixed interface word instead. The input
    // version does not select a different control implementation.
    const auto command = read(message, 24);
    if (command == 3) return control_reply(0x400, 2);
    if (command != 0) return unsupported("Log-manager control command is unsupported");
    if (!objects_.empty()) return unsupported("Repeated domain conversion is unsupported");
    auto reply = control_reply(1, 4);
    objects_.push_back(root_);
    return reply;
}

horizon::IpcReply LogManagerSession::domain(const horizon::IpcMessage& message,
                                            std::uint64_t sender_pid) {
    // The domain header is 16 bytes at byte32: request type1, input-object
    // count, CMIF data size, live object ID, then two context words. Type4/6
    // requests share this route.
    if (read(message, 32, 1) != 1)
        return unsupported("Log-manager domain request type is unsupported");
    const auto input_objects = read(message, 33, 1);
    const auto data_size = read(message, 34, 2);
    const auto object_id = read(message, 36);
    // Bit31 is the receive-PID descriptor; no other descriptor or handle is
    // accepted. The raw extent still starts after the four header words and
    // the special header word the SDK reserves in both variants.
    const auto second = read(message, 4);
    if ((second & ~0x800003ffu) != 0)
        return unsupported("Log-manager descriptors are unsupported");
    const auto raw_end = 20 + 4 * (second & 0x3ffu);
    // The leading halfword after the type carries the HIPC message length in
    // words when the SDK sets it; zero means unset. Any other value would be a
    // descriptor or a length inconsistent with the admitted raw extent.
    const auto length = read(message, 2, 2);
    if (length != 0 && length * 4 != raw_end)
        return unsupported("Log-manager HIPC length or descriptors are unsupported");
    if (data_size < 16 || raw_end < 64 || raw_end > message.size())
        return unsupported("Log-manager domain packet is outside the supported wire format");
    // The firmware dispatcher reads the magic and command without choosing
    // behavior by the input version, so the version word is not constrained.
    if (read(message, 48) != sfci)
        return unsupported("Log-manager CMIF header is unsupported");
    const auto command = read(message, 56);
    if (object_id == 0 || object_id > objects_.size())
        return unsupported("Log-manager domain object does not exist in this session");
    const auto& object = objects_[object_id - 1];
    if (input_objects != 0)
        return unsupported("Log-manager input objects are unsupported");
    if (object->kind == ObjectKind::root) return open_logger(message, sender_pid);
    if (object->process_id != sender_pid)
        return unsupported("Logger object belongs to a different process");
    if (command != 0) return unsupported("Log-manager logger command is unsupported");
    return log(message, sender_pid);
}

horizon::IpcReply LogManagerSession::log(const horizon::IpcMessage& message,
                                         std::uint64_t) {
    // Firmware 20.5.0 completes ILogger command0 without consuming the input
    // buffer, so the reply carries no returned objects and no output data.
    const auto data_size = read(message, 34, 2);
    if (data_size != 16)
        return unsupported("Log-manager logger payload size is unsupported");
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 48; // HIPC header8 + domain output header16 + CMIF header16 + 8.
    write(reply.bytes, 4, 10);
    cmif_header(reply.bytes, 32, log_service_interface);
    return reply;
}

horizon::IpcReply LogManagerSession::open_logger(const horizon::IpcMessage& message,
                                               std::uint64_t sender_pid) {
    if (objects_.empty()) return unsupported("Non-domain OpenLogger handle transfer is unsupported");
    if (!packet_shape(message, 4, true, 14))
        return unsupported("OpenLogger requires a PID descriptor and no handles or buffers");
    if (read(message, 32, 1) != 1 || read(message, 33, 1) != 0)
        return unsupported("Log-manager domain operation or input objects are unsupported");
    const auto data_size = read(message, 34, 2);
    const auto raw_end = 20 + 4 * (read(message, 4) & 0x3ffu);
    if (data_size < 24 || data_size > raw_end - 48)
        return unsupported("Log-manager domain payload length is unsupported");
    const auto id = read(message, 36);
    if (id == 0 || id > objects_.size())
        return unsupported("Log-manager domain object does not exist in this session");
    const auto& object = objects_[id - 1];
    if (object->kind != ObjectKind::root)
        return unsupported("Logger commands are not implemented");
    if (read(message, 48) != sfci || read(message, 56) != 0)
        return unsupported("Log-manager CMIF header or command is unsupported");
    if (objects_.size() >= std::numeric_limits<std::uint32_t>::max())
        return unsupported("Log-manager domain IDs exhausted; wire failure unsupported");

    // Types4/6 share this domain dispatch. For type6 the request context is
    // domain-header+12, not nested CMIF-header+12. OpenLogger does not use or
    // echo it; the firmware response below is identical for both wire types.
    // Neither the HIPC PID bytes nor the inline PID placeholder are trusted.
    // The kernel-facing dispatcher supplies the actual sender identity.
    const auto logger = std::make_shared<DomainObject>(
        DomainObject{ObjectKind::logger, sender_pid, object});
    const auto logger_id = static_cast<std::uint32_t>(objects_.size() + 1);
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 60; // HIPC header8 + raw data52 (13 words).
    write(reply.bytes, 4, 13);
    write(reply.bytes, 16, 1); // One returned domain object, then12 zero bytes.
    cmif_header(reply.bytes, 32, log_service_interface);
    write(reply.bytes, 48, logger_id);
    objects_.push_back(logger);
    return reply;
}

std::optional<std::uint64_t> LogManagerSession::logger_process_id(std::uint32_t id) const {
    const std::lock_guard lock(mutex_);
    if (id == 0 || id > objects_.size() || objects_[id - 1]->kind != ObjectKind::logger) return {};
    return objects_[id - 1]->process_id;
}

std::size_t LogManagerSession::domain_object_count() const {
    const std::lock_guard lock(mutex_);
    return objects_.size();
}

const char* LogManagerSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::lm
