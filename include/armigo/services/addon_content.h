// ============================================================================
// Armigo Services - Initial Add-on Content Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>
#include <armigo/services/addon_content_catalog.h>

#include <memory>
#include <mutex>

namespace armigo::services::aoc {

// Native aoc:u session over the application's installed add-on content. It
// counts and lists the catalog's items by index, reports the add-on content
// base id and prepares an installed item; an absent catalog has no items and
// no base id. The host provides no pointer buffer. Other methods, and preparing
// an index that is not installed, are explicitly unsupported.
class AddOnContentSession final : public horizon::ClientSession {
public:
    AddOnContentSession(std::uint64_t process_id, horizon::HandleTable& handles,
                        std::shared_ptr<const AddOnContentCatalog> catalog = {});
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    std::size_t domain_object_count() const;
    const char* diagnostic() const;

private:
    struct RootObject {
        explicit RootObject(std::uint64_t pid)
            : process_id(pid), changed(std::make_shared<horizon::Event>()) {}
        const std::uint64_t process_id;
        const std::shared_ptr<horizon::Event> changed;
    };
    horizon::IpcReply unsupported(const char* detail);
    horizon::IpcReply method(const horizon::IpcMessage& message, const horizon::GuestBuffers* buffers);
    mutable std::mutex mutex_;
    [[maybe_unused]] horizon::HandleTable& handles_;
    const std::shared_ptr<RootObject> root_;
    const std::shared_ptr<const AddOnContentCatalog> catalog_;
    bool domain_ = false;
    const char* diagnostic_ = nullptr;
};

}  // namespace armigo::services::aoc
