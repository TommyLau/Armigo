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
constexpr std::uint32_t switch2_logger_interface = 0x3e81dad2;

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
    // The leading halfword is the wire type. Preserve the legacy checks here;
    // Switch 2 callers validate the upper descriptor counts separately.
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

LogManagerSession::LogManagerSession(std::uint64_t process_id, LogManagerContract contract)
    : process_id_(process_id), contract_(contract),
      root_(std::make_shared<DomainObject>(DomainObject{ObjectKind::root, process_id, {}})) {}

horizon::IpcReply LogManagerSession::close_server() {
    const std::lock_guard lock(mutex_);
    if (contract_ == LogManagerContract::horizon_20_5_0) return ClientSession::close_server();
    if (contract_ != LogManagerContract::switch2_23_0_1)
        return unsupported("Log-manager contract is unsupported");
    if (closed_) return {.disposition = IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    closed_ = true;
    std::vector<std::shared_ptr<DomainObject>>{}.swap(objects_);
    root_.reset();
    diagnostic_ = "Log-manager server session is closed";
    return {.disposition = IpcDisposition::handled};
}

horizon::IpcReply LogManagerSession::unsupported(const char* message) {
    diagnostic_ = message;
    return {};
}

horizon::IpcReply LogManagerSession::request(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid) {
    return dispatch(message, sender_pid, nullptr);
}

horizon::IpcReply LogManagerSession::request_with_buffers(const horizon::IpcMessage& message,
    std::uint64_t sender_pid, const horizon::GuestBuffers& buffers) {
    return dispatch(message, sender_pid, &buffers);
}

horizon::IpcReply LogManagerSession::dispatch(const horizon::IpcMessage& message,
    std::uint64_t sender_pid, const horizon::GuestBuffers* buffers) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (contract_ != LogManagerContract::horizon_20_5_0 &&
        contract_ != LogManagerContract::switch2_23_0_1)
        return unsupported("Log-manager contract is unsupported");
    if (closed_) return {.disposition = IpcDisposition::handled,
                         .transport_result = horizon::session_closed_result};
    if (sender_pid != process_id_)
        return unsupported("Log-manager session belongs to a different process");
    try {
        const auto type = read(message, 0, 2);
        if (type == 5 || type == 7) return control(message);
        if (type == 4 || type == 6) return domain(message, sender_pid, buffers);
        return unsupported("Log-manager HIPC message type or descriptors are unsupported");
    } catch (const std::bad_alloc&) {
        return unsupported("Log-manager object allocation failed; wire failure unsupported");
    } catch (const std::length_error&) {
        return unsupported("Log-manager object storage exhausted; wire failure unsupported");
    }
}

horizon::IpcReply LogManagerSession::control(const horizon::IpcMessage& message) {
    if (contract_ == LogManagerContract::switch2_23_0_1 && read(message, 2, 2) != 0)
        return unsupported("Switch 2 log-manager controls cannot carry buffer descriptors");
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
    std::uint64_t sender_pid, const horizon::GuestBuffers* buffers) {
    // The domain header is 16 bytes at byte32: request type1, input-object
    // count, CMIF data size, live object ID, then two context words. Type4/6
    // requests share this route.
    if (read(message, 32, 1) != 1)
        return unsupported("Log-manager domain request type is unsupported");
    const auto input_objects = read(message, 33, 1);
    const auto data_size = read(message, 34, 2);
    const auto object_id = read(message, 36);
    // The second word admits raw-data size and PID, but no receive list. The
    // supported PID shape has a 20-byte header area; the X/A shape has 28 bytes.
    const auto second = read(message, 4);
    if ((second & ~0x800003ffu) != 0)
        return unsupported("Log-manager descriptors are unsupported");
    const auto descriptor_counts = read(message, 2, 2);
    const bool switch2_buffer = contract_ == LogManagerContract::switch2_23_0_1 &&
                               descriptor_counts == 0x11 && !(second >> 31);
    const auto raw_end = (switch2_buffer ? 28 : 20) + 4 * (second & 0x3ffu);
    // Preserve the legacy admission predicate. Switch 2 interprets the upper
    // halfword as descriptor counts and validates the X/A pair explicitly.
    if (contract_ == LogManagerContract::horizon_20_5_0) {
        const auto length = descriptor_counts;
        if (length != 0 && length * 4 != raw_end)
            return unsupported("Log-manager HIPC length or descriptors are unsupported");
    } else if (!switch2_buffer && (descriptor_counts != 0 || !(second >> 31))) {
        return unsupported("Switch 2 log-manager descriptor shape is unsupported");
    }
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
    return log(message, buffers);
}

horizon::IpcReply LogManagerSession::log(const horizon::IpcMessage& message,
    const horizon::GuestBuffers* buffers) {
    // Both verified retail implementations return success without interpreting
    // the log payload. Transport validation precedes the contract-specific reply.
    const auto data_size = read(message, 34, 2);
    if (data_size != 16)
        return unsupported("Log-manager logger payload size is unsupported");
    const bool switch2 = contract_ == LogManagerContract::switch2_23_0_1;
    if (switch2) {
        // The verified metadata declares one input auto-select buffer (0x21).
        // Admit one X/A pair with exactly one active descriptor, or both empty.
        if (read(message, 2, 2) != 0x11 || (read(message, 4) & ~0x3ffu) != 0 ||
            (read(message, 4) & 0x3ffu) < 12)
            return unsupported("Switch 2 logger requires one input auto-select buffer");
        const auto x = read(message, 8), x_low = read(message, 12);
        const auto a_size = read(message, 16), a_low = read(message, 20), a = read(message, 24);
        if ((x & 0x3fu) != 0 || (a & 0x00ffff03u) != 0)
            return unsupported("Switch 2 logger buffer index or mode is unsupported");
        const std::uint64_t x_size = x >> 16;
        const std::uint64_t alias_size = std::uint64_t{a_size} | (std::uint64_t{(a >> 24) & 15u} << 32);
        const auto wire_address = [](std::uint32_t low, unsigned high, unsigned mid) {
            return std::uint64_t{low} | (std::uint64_t{mid} << 32) | (std::uint64_t{high} << 36);
        };
        const auto x_address = wire_address(x_low, (x >> 6) & 0x3fu, (x >> 12) & 15u);
        const auto a_address = wire_address(a_low, (a >> 2) & 0x3fu, (a >> 28) & 15u);
        if ((x_size && alias_size) || x_size > 0x400 || alias_size > 0x10000 ||
            (!x_size && (x || x_low)) || (!alias_size && (a_size || a_low || a)))
            return unsupported("Switch 2 logger buffer shape exceeds the supported bounds");
        const auto size = x_size ? x_size : alias_size;
        const auto address = x_size ? x_address : a_address;
        if (size) {
            if (!address || !buffers || buffers->size() != 1 ||
                !(*buffers)[0].input || (*buffers)[0].address != address || (*buffers)[0].size != size)
                return unsupported("Switch 2 logger buffer does not match the validated transport");
            // Bound and validate the input copy before the retail no-op method.
            // This temporary buffer is host overhead, never application backing.
            std::vector<std::byte> input(static_cast<std::size_t>(size));
            if (!buffers->read(0, input)) return unsupported("Switch 2 logger input buffer is unreadable");
        } else if (buffers && !buffers->empty()) {
            return unsupported("Switch 2 logger received an unexpected buffer");
        }
    }
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = switch2 ? 56 : 48;
    write(reply.bytes, 4, switch2 ? 12 : 10);
    cmif_header(reply.bytes, 32, switch2 ? switch2_logger_interface : log_service_interface);
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
