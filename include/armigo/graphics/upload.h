// ============================================================================
// Armigo Graphics - Inline Memory Upload
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <optional>
#include <utility>

#include <armigo/graphics/memory.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace armigo::graphics {

enum class UploadStatus : std::uint8_t {
    Complete,
    Pending,
    InvalidRegister,
    InvalidLaunch,
    Unsupported,
    BudgetExceeded,
    DataWithoutLaunch,
    LaunchWhilePending,
    MemoryFault,
};

struct UploadResult {
    UploadStatus status = UploadStatus::Complete;
    std::string_view detail{};
    std::uint64_t destination = 0;
    std::uint64_t expected_bytes = 0;
    std::uint64_t received_bytes = 0;

    [[nodiscard]] constexpr bool accepted() const noexcept {
        return status == UploadStatus::Complete || status == UploadStatus::Pending;
    }
};

struct StagedUploadWrite {
    std::uint64_t destination = 0;
    std::span<const std::byte> payload{};
};

// Buffered inline-to-memory state. Register indices are relative to method
// 0x60: indices 0 through 11 are methods 0x60 through 0x6b.
class InlineUpload {
public:
    static constexpr std::size_t register_count = 12;
    static constexpr std::uint64_t max_upload_bytes = 64u * 1024u * 1024u;

    [[nodiscard]] UploadResult set_register(std::uint32_t relative_method,
                                            std::uint32_t value) noexcept;
    [[nodiscard]] UploadResult launch(std::uint32_t flags) noexcept;
    [[nodiscard]] UploadResult push_word(
        std::uint32_t value, const GpuAddressSpace& memory) noexcept;
    // Stages bytes without publishing them to guest memory. Once the final
    // word arrives, staged_write() remains valid until commit_write().
    [[nodiscard]] UploadResult push_word(std::uint32_t value) noexcept;
    [[nodiscard]] std::optional<StagedUploadWrite> staged_write() const noexcept;
    [[nodiscard]] UploadResult commit_write() noexcept;

    // Pending transfers and failed destination writes both block later ordered
    // side effects. Packet boundaries do not change this state.
    [[nodiscard]] bool blocked() const noexcept { return active_ || failed_; }
    // Destination range written by push_word() when the next word completes
    // the active transfer; nothing otherwise.
    [[nodiscard]] std::optional<std::pair<std::uint64_t, std::uint64_t>>
    write_on_next_word() const noexcept {
        if (!active_ || failed_ || received_bytes_ + 4 < expected_bytes_) return std::nullopt;
        return std::pair{destination_, expected_bytes_};
    }
    [[nodiscard]] UploadStatus status() const noexcept;

private:
    [[nodiscard]] UploadResult current(UploadStatus status,
                                       std::string_view detail = {}) const noexcept;

    std::array<std::uint32_t, register_count> registers_{};
    std::vector<std::byte> payload_;
    std::uint64_t destination_ = 0;
    std::uint64_t expected_bytes_ = 0;
    std::uint64_t received_bytes_ = 0;
    bool active_ = false;
    bool failed_ = false;
    UploadResult failure_{};
};

}  // namespace armigo::graphics
