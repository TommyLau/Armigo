// ============================================================================
// Armigo Services - Initial Runtime-Loader (ro) Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/services/loader.h>

namespace armigo::services::ro {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

using namespace horizon::cmif;


// A domain command that carries no output: the CMIF header is authored where the
// dispatcher expects it and nothing follows.
IpcReply empty_reply(std::size_t cmif) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = cmif + 16;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, cmif, sfco);
    write(reply.bytes, cmif + 4, 1);
    return reply;
}
}  // namespace

LoaderSession::LoaderSession(std::uint64_t process_id) : process_id_(process_id) {}

IpcReply LoaderSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply LoaderSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_)
        return unsupported("Runtime-loader session belongs to a different process");
    const auto type = read(message, 0) & 0xffffu;
    if (type == 5 || type == 7) {
        if (read(message, 16) != sfci) return unsupported("Runtime-loader control magic is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_reply(0, 2);
        if (command == 0) {
            if (converted_) return unsupported("Repeated runtime-loader domain conversion is unsupported");
            converted_ = true;
            return control_reply(1, 4);
        }
        if (command == 2) {
            IpcReply reply = control_reply(0, 4);
            reply.transferred[0] = std::make_shared<LoaderSession>(process_id_);
            reply.transferred_count = 1;
            return reply;
        }
        return unsupported("Runtime-loader control command is unsupported");
    }
    if (type != 4 && type != 6) return unsupported("Runtime-loader request shape is unsupported");
    std::size_t cmif = 0;
    if (read(message, 48) == sfci) cmif = 48;
    else if (read(message, 32) == sfci) cmif = 32;
    else if (read(message, 16) == sfci) cmif = 16;
    else return unsupported("Runtime-loader request shape is unsupported");
    switch (read(message, cmif + 8)) {
    case 2: return register_module(message, cmif);
    case 3: return unregister_module(message, cmif);
    default: return unsupported("Runtime-loader command is not implemented");
    }
}

IpcReply LoaderSession::register_module(const IpcMessage& message, std::size_t cmif) {
    // RegisterModuleInfo(process_id, nrr_address, nrr_size): the client hands over the range
    // of the module it loaded, and the service records it for later exception reports.
    const auto process = read(message, cmif + 16);
    const auto address = read64(message, cmif + 24);
    const auto size = read64(message, cmif + 32);
    if (process != process_id_)
        return unsupported("Runtime-loader registration names a different process");
    if (address == 0 || size == 0)
        return unsupported("Runtime-loader registration needs a non-empty range");
    modules_[address] = ModuleRange{.address = address, .size = size};
    return empty_reply(cmif);
}

IpcReply LoaderSession::unregister_module(const IpcMessage& message, std::size_t cmif) {
    const auto address = read64(message, cmif + 16);
    const auto found = modules_.find(address);
    if (found == modules_.end())
        return unsupported("Runtime-loader unregistration names an unknown range");
    modules_.erase(found);
    return empty_reply(cmif);
}

std::size_t LoaderSession::domain_object_count() const {
    const std::lock_guard lock(mutex_);
    return converted_ ? 1 : 0;
}

const char* LoaderSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

std::size_t LoaderSession::module_count() const {
    const std::lock_guard lock(mutex_);
    return modules_.size();
}

}  // namespace armigo::services::ro
