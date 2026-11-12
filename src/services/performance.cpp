// ============================================================================
// Armigo Services - Initial Performance Manager Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/services/performance.h>

#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

namespace armigo::services::apm {
namespace {
using namespace horizon;

using namespace horizon::cmif;

bool shape(const IpcMessage& message, unsigned minimum_words) {
    const auto words = read(message, 4);
    return words >= minimum_words && words <= (message.size() - 8) / 4;
}

void sfco(IpcMessage& message, std::size_t offset, std::uint32_t interface_id) {
    write(message, offset, 0x4f434653);
    write(message, offset + 12, interface_id);
}

IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 44;
    write(reply.bytes, 4, 9);
    sfco(reply.bytes, 16, 0xec6be3ff);
    write(reply.bytes, 32, value, width);
    return reply;
}
} // namespace

PerformanceSession::PerformanceSession(std::uint64_t process_id, std::uint16_t capacity,
                                       Role role, OperationMode operation_mode)
    : process_id_(process_id), capacity_(capacity), role_(role),
      operation_mode_(operation_mode), pointer_buffer_(capacity),
      root_(std::make_shared<Object>(Object{Kind::root, {}})) {}

horizon::IpcReply PerformanceSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

horizon::IpcReply PerformanceSession::request(const horizon::IpcMessage& message,
                                             std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("APM session belongs to a different process");
    try {
        const auto type = read(message, 0);
        if (role_ == Role::opened) {
            // ISession command0: SetPerformanceConfiguration, two inline u32
            // arguments. The host records the request; it does not change Apple
            // hardware clocks. Other session commands stay explicit unsupported.
            if ((type != 4 && type != 6) || read(message, 4) != 10 ||
                read(message, 16) != 0x49434653 || read(message, 24) != 0)
                return unsupported("APM session command is not implemented");
            requested_.mode = read(message, 32);
            requested_.configuration = read(message, 36);
            requested_.set = true;
            IpcReply reply{IpcDisposition::handled};
            reply.size = 44;
            write(reply.bytes, 4, 9);
            sfco(reply.bytes, 16, 0xbc2665da);
            diagnostic_ = nullptr;
            return reply;
        }
        if (type == 5 || type == 7) {
            if (!shape(message, 8) || read(message, 16) != 0x49434653)
                return unsupported("APM control packet is unsupported");
            const auto command = read(message, 24);
            if (command == 3) return control_reply(pointer_buffer_.size(), 2);
            if (command != 0 || !objects_.empty())
                return unsupported("APM control command or repeated conversion is unsupported");
            const auto reply = control_reply(1, 4);
            objects_.push_back(root_);
            return reply;
        }
        // Non-domain OpenSession: type4/6, eight raw words, no descriptors and
        // the CMIF header at byte16. The original SDK sends this shape through
        // its generic no-input command helper before any domain conversion.
        // The alignment padding before the header (bytes 8-15) is not part of
        // the request; newer SDKs leave stale values there.
        if ((type == 4 || type == 6) && read(message, 4) == 8 && read(message, 16) == 0x49434653 &&
            read(message, 24) == 0 && read(message, 28) == 0)
            return open_session(message);
        if ((type == 4 || type == 6) && read(message, 4) == 8 &&
            read(message, 16) == 0x49434653 && read(message, 24) == 1) {
            IpcReply reply{IpcDisposition::handled};
            reply.size = 44;
            write(reply.bytes, 4, 9);
            sfco(reply.bytes, 16, 0xbc2665da);
            write(reply.bytes, 32, operation_mode_profile(operation_mode_).performance_mode);
            return reply;
        }
        if ((type != 4 && type != 6) || !shape(message, 12) || objects_.empty())
            return unsupported("APM requires a supported domain request without descriptors");
        if (read(message, 16, 1) != 1 || read(message, 17, 1) != 0)
            return unsupported("APM domain operation or input objects are unsupported");
        const auto data_size = read(message, 18, 2);
        if (data_size < 16 || data_size > 8 + 4 * read(message, 4) - 32)
            return unsupported("APM domain payload length is unsupported");
        const auto id = read(message, 20);
        if (!id || id > objects_.size() || objects_[id - 1]->kind != Kind::root)
            return unsupported("APM object or child command is unsupported");
        if (read(message, 32) != 0x49434653)
            return unsupported("APM manager command is unsupported");
        if (read(message, 40) == 1) {
            horizon::IpcReply reply{IpcDisposition::handled};
            reply.size = 60;
            write(reply.bytes, 4, 13);
            sfco(reply.bytes, 32, 0xbc2665da);
            write(reply.bytes, 48, operation_mode_profile(operation_mode_).performance_mode);
            return reply;
        }
        if (read(message, 40) != 0)
            return unsupported("APM manager command is unsupported");
        if (objects_.size() >= std::numeric_limits<std::uint32_t>::max())
            return unsupported("APM domain IDs exhausted");
        const auto child = std::make_shared<Object>(Object{Kind::session, root_});
        const auto child_id = static_cast<std::uint32_t>(objects_.size() + 1);
        IpcReply reply{IpcDisposition::handled};
        reply.size = 60;
        write(reply.bytes, 4, 13);
        write(reply.bytes, 16, 1);
        sfco(reply.bytes, 32, 0xbc2665da);
        write(reply.bytes, 48, child_id);
        objects_.push_back(child);
        return reply;
    } catch (const std::bad_alloc&) {
        return unsupported("APM allocation failed; guest failure framing is unsupported");
    } catch (const std::length_error&) {
        return unsupported("APM object storage exhausted; guest failure framing is unsupported");
    }
}

horizon::IpcReply PerformanceSession::open_session(const IpcMessage& message) {
    // Input version is not part of the firmware decision; the SDK sends 1.
    if (read(message, 20) != 1)
        return unsupported("APM OpenSession input version is unsupported");
    IpcReply reply{IpcDisposition::handled};
    reply.size = 32; // HIPC header8 + one move-handle entry8 + CMIF header16
    write(reply.bytes, 4, 4);
    sfco(reply.bytes, 16, 0xbc2665da);
    reply.transferred[0] =
        std::make_shared<PerformanceSession>(process_id_, capacity_, Role::opened, operation_mode_);
    reply.transferred_count = 1;
    return reply;
}

PerformanceSession::RequestedConfiguration PerformanceSession::requested_configuration() const {
    const std::lock_guard lock(mutex_);
    return requested_;
}

std::size_t PerformanceSession::domain_object_count() const {
    const std::lock_guard lock(mutex_);
    return objects_.size();
}

const char* PerformanceSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::apm
