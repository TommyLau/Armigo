// ============================================================================
// Armigo Services - Initial Parental Control Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/services/parental_control.h>

#include <limits>
#include <new>
#include <stdexcept>

namespace armigo::services::pctl {
namespace {
using namespace horizon;
using namespace horizon::cmif;
void response_header(IpcMessage& bytes, unsigned offset, std::uint32_t interface_id) {
    write(bytes, offset, 0x4f434653);
    write(bytes, offset + 12, interface_id);
}
IpcReply control_reply(std::uint32_t value) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 44;
    write(reply.bytes, 4, 9);
    response_header(reply.bytes, 16, 0xec6be3ff);
    write(reply.bytes, 32, value);
    return reply;
}
} // namespace

PctlSession::PctlSession(std::uint64_t process_id)
    : process_id_(process_id), root_(std::make_shared<Object>(Object{process_id, false, {}})) {}

horizon::IpcReply PctlSession::request(const horizon::IpcMessage& message,
                                      std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = "Parental-control method or packet is unsupported";
    if (sender_pid != process_id_) {
        diagnostic_ = "Parental-control session belongs to a different process";
        return {};
    }
    try {
        const auto type = read(message, 0);
        const auto header = read(message, 4);
        if (type == 5 || type == 7) {
            if (header < 8 || header > (message.size() - 8) / 4 || read(message, 16) != 0x49434653)
                return {};
            const auto command = read(message, 24);
            if (command == 3) {
                diagnostic_ = nullptr;
                return control_reply(0); // Actual host pointer-buffer capacity.
            }
            if (command != 0 || !objects_.empty()) return {};
            objects_.push_back(root_);
            diagnostic_ = nullptr;
            return control_reply(1);
        }
        if ((type != 4 && type != 6) || objects_.empty()) return {};
        const bool pid = (header & 0x80000000u) != 0;
        const auto words = header & 0x3ffu;
        const unsigned raw_start = pid ? 20 : 8;
        const unsigned domain = pid ? 32 : 16;
        const unsigned cmif = domain + 16;
        const unsigned minimum_data = pid ? 24 : 16;
        if ((header & ~0x3ffu) != (pid ? 0x80000000u : 0u) ||
            words < (pid ? 14u : 12u) || words > (message.size() - raw_start) / 4 ||
            (pid && read(message, 8) != 1) || read(message, domain, 2) != 1)
            return {};
        const auto data_size = read(message, domain + 2, 2);
        const auto id = read(message, domain + 4);
        if (data_size < minimum_data || data_size > raw_start + 4 * words - cmif ||
            id == 0 || id > objects_.size() || read(message, cmif) != 0x49434653 ||
            read(message, cmif + 8) != 1)
            return {};
        const auto object = objects_[id - 1];
        if (object == root_) {
            if (!pid || objects_.size() >= std::numeric_limits<std::uint32_t>::max()) return {};
            // The PID descriptor and inline placeholder are not identity sources.
            // Domain children belong to the actual sender authorized for this session.
            auto child = std::make_shared<Object>(Object{sender_pid, false, root_});
            const auto child_id = static_cast<std::uint32_t>(objects_.size() + 1);
            objects_.push_back(std::move(child));
            IpcReply reply{IpcDisposition::handled};
            reply.size = 60;
            write(reply.bytes, 4, 13);
            write(reply.bytes, 16, 1);
            response_header(reply.bytes, 32, 0x64f05bf6);
            write(reply.bytes, 48, child_id);
            diagnostic_ = nullptr;
            return reply;
        }
        if (pid || object->process_id != sender_pid) return {};
        // This launch-authorized application profile initializes local service
        // state only. It does not answer or bypass later restriction checks.
        object->initialized = true;
        IpcReply reply{IpcDisposition::handled};
        reply.size = 56;
        write(reply.bytes, 4, 12);
        response_header(reply.bytes, 32, 0x81a6fff1);
        diagnostic_ = nullptr;
        return reply;
    } catch (const std::bad_alloc&) {
        diagnostic_ = "Parental-control allocation failed; guest error framing is unsupported";
    } catch (const std::length_error&) {
        diagnostic_ = "Parental-control object storage exhausted; guest error framing is unsupported";
    }
    return {};
}

const char* PctlSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::pctl
