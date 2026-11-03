// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/channel.h>
#include <memory>
#include <string>
#include <utility>
#include <variant>

namespace armigo::graphics {

// Async errors own the existing CommandResult diagnostic. command() borrows
// this object's text: a sticky channel must retain the error or copy its text.
// Copy/move never retain a string_view into another error's storage.
class RendererError {
public:
    RendererError() : RendererError({CommandStatus::Unsupported,
                                     "renderer operation is not implemented"}) {}
    explicit RendererError(CommandResult command)
        : command_(command), detail_(command.detail) {
        command_.detail = {};
        if (command_.ok()) {
            command_.status = CommandStatus::Unsupported;
            detail_ = "renderer error cannot contain successful completion";
        }
    }
    [[nodiscard]] CommandResult command() const noexcept {
        auto result = command_;
        result.detail = detail_;
        return result;
    }
private:
    CommandResult command_;
    std::string detail_;
};

// Default construction fails closed. A value means this operation succeeded,
// never that GPU work or guest-visible completion happened implicitly.
template <class T> using RendererResult = std::variant<RendererError, T>;
using RendererStatus = RendererResult<std::monostate>;
inline RendererStatus renderer_success() { return std::monostate{}; }

// Concrete owners retain allocations, pins, staging slices, native objects or
// callback contexts. Destruction must not call guest code or reenter nvdrv.
class ResourceLease {
public:
    virtual ~ResourceLease() = default;
};
using RetainedResource = std::shared_ptr<const ResourceLease>;

struct SubmissionId {
    std::uint64_t queue = 0;
    std::uint64_t sequence = 0;
    bool operator==(const SubmissionId&) const = default;
};

enum class ResourceAccess : std::uint8_t { Read, Write, ReadWrite };

} // namespace armigo::graphics
