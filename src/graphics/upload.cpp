// ============================================================================
// Armigo Graphics - Inline Memory Upload
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/upload.h>

#include <algorithm>
#include <new>
#include <utility>

namespace armigo::graphics {
namespace {

constexpr std::uint32_t launch_layout_pitch = 1u << 0;
constexpr std::uint32_t launch_reduction_enable = 1u << 1;
constexpr std::uint32_t launch_reduction_format = 3u << 2;
constexpr std::uint32_t launch_completion = 3u << 4;
constexpr std::uint32_t launch_sysmembar_disable = 1u << 6;
constexpr std::uint32_t launch_interrupt = 3u << 8;
constexpr std::uint32_t launch_semaphore_size = 1u << 12;
constexpr std::uint32_t launch_reduction_operation = 7u << 13;
constexpr std::uint32_t known_launch_bits =
    launch_layout_pitch | launch_reduction_enable | launch_reduction_format |
    launch_completion | launch_sysmembar_disable | launch_interrupt |
    launch_semaphore_size | launch_reduction_operation;
constexpr std::uint64_t gpu_address_limit = std::uint64_t{1} << 40;

constexpr UploadResult result(UploadStatus status, std::string_view detail,
                              std::uint64_t destination = 0,
                              std::uint64_t expected = 0,
                              std::uint64_t received = 0) noexcept {
    return {status, detail, destination, expected, received};
}

}  // namespace

UploadResult InlineUpload::current(UploadStatus upload_status,
                                   std::string_view detail) const noexcept {
    return result(upload_status, detail, destination_, expected_bytes_,
                  received_bytes_);
}

UploadStatus InlineUpload::status() const noexcept {
    if (failed_) {
        return failure_.status;
    }
    return active_ ? UploadStatus::Pending : UploadStatus::Complete;
}

UploadResult InlineUpload::set_register(std::uint32_t relative_method,
                                        std::uint32_t value) noexcept {
    if (failed_) {
        return failure_;
    }
    if (relative_method >= registers_.size()) {
        return current(UploadStatus::InvalidRegister,
                       "inline upload register is outside methods 0x60 through 0x6b");
    }
    registers_[relative_method] = value;
    return current(active_ ? UploadStatus::Pending : UploadStatus::Complete);
}

UploadResult InlineUpload::launch(std::uint32_t flags) noexcept {
    if (failed_) {
        return failure_;
    }
    if (active_) {
        return current(UploadStatus::LaunchWhilePending,
                       "inline upload launch arrived before the prior payload completed");
    }
    if ((flags & ~known_launch_bits) != 0) {
        return result(UploadStatus::Unsupported,
                      "inline upload launch contains unknown control bits");
    }
    if ((flags & launch_layout_pitch) == 0) {
        return result(UploadStatus::Unsupported,
                      "block-linear inline upload is not implemented");
    }
    if ((flags & (launch_reduction_enable | launch_reduction_format |
                  launch_reduction_operation)) != 0) {
        return result(UploadStatus::Unsupported,
                      "inline upload reduction is not implemented");
    }
    if ((flags & launch_completion) != (1u << 4)) {
        return result(UploadStatus::Unsupported,
                      "inline upload requires the established flush-only completion form");
    }
    if ((flags & launch_interrupt) != 0) {
        return result(UploadStatus::Unsupported,
                      "inline upload interrupt completion is not implemented");
    }
    if ((flags & launch_sysmembar_disable) != 0) {
        return result(UploadStatus::Unsupported,
                      "inline upload with disabled system-memory barrier is not implemented");
    }
    const std::uint32_t line_length = registers_[0];
    const std::uint32_t line_count = registers_[1];
    if (line_count == 0) {
        return result(UploadStatus::InvalidLaunch,
                      "inline upload line count must be nonzero");
    }
    if (line_count != 1) {
        return result(UploadStatus::Unsupported,
                      "multi-line inline upload padding is not established");
    }
    if (line_length == 0) {
        payload_.clear();
        destination_ = 0;
        expected_bytes_ = 0;
        received_bytes_ = 0;
        return current(UploadStatus::Complete);
    }
    if (registers_[10] != 0 || registers_[11] != 0) {
        return result(UploadStatus::Unsupported,
                      "inline upload destination origins are not implemented");
    }
    if ((registers_[2] & ~0xffu) != 0) {
        return result(UploadStatus::InvalidLaunch,
                      "inline upload destination exceeds the 40-bit GPU address space");
    }
    if (line_length > max_upload_bytes) {
        return result(UploadStatus::BudgetExceeded,
                      "inline upload exceeds the byte budget");
    }

    const std::uint64_t destination =
        (static_cast<std::uint64_t>(registers_[2]) << 32) | registers_[3];
    if (destination >= gpu_address_limit ||
        line_length > gpu_address_limit - destination) {
        return result(UploadStatus::InvalidLaunch,
                      "inline upload destination range exceeds the 40-bit GPU address space",
                      destination, line_length, 0);
    }
    std::vector<std::byte> payload;
    try {
        payload.resize(line_length);
    } catch (const std::bad_alloc&) {
        return result(UploadStatus::BudgetExceeded,
                      "inline upload payload allocation failed", destination,
                      line_length, 0);
    }

    payload_ = std::move(payload);
    destination_ = destination;
    expected_bytes_ = line_length;
    received_bytes_ = 0;
    active_ = true;
    return current(UploadStatus::Pending);
}

UploadResult InlineUpload::push_word(
    std::uint32_t value, const GpuAddressSpace& memory) noexcept {
    UploadResult staged = push_word(value);
    if (!staged.accepted()) return staged;
    const auto write_request = staged_write();
    if (!write_request) return staged;

    const MemoryResult write = memory.write(write_request->destination, write_request->payload);
    if (!write) {
        failed_ = true;
        active_ = false;
        failure_ = current(
            UploadStatus::MemoryFault,
            write.detail.empty() ? "inline upload destination write failed"
                                 : write.detail);
        return failure_;
    }
    return commit_write();
}

UploadResult InlineUpload::push_word(std::uint32_t value) noexcept {
    if (failed_) {
        return failure_;
    }
    if (!active_) {
        return current(UploadStatus::DataWithoutLaunch,
                       "inline upload data arrived without an active launch");
    }

    const std::uint64_t remaining = expected_bytes_ - received_bytes_;
    const auto count = static_cast<unsigned>(std::min<std::uint64_t>(4, remaining));
    for (unsigned index = 0; index < count; ++index) {
        payload_[static_cast<std::size_t>(received_bytes_ + index)] =
            static_cast<std::byte>(value >> (index * 8));
    }
    received_bytes_ += count;
    if (received_bytes_ != expected_bytes_) {
        return current(UploadStatus::Pending);
    }

    return current(UploadStatus::Pending);
}

std::optional<StagedUploadWrite> InlineUpload::staged_write() const noexcept {
    if (!active_ || failed_ || received_bytes_ != expected_bytes_) return std::nullopt;
    return StagedUploadWrite{destination_, payload_};
}

UploadResult InlineUpload::commit_write() noexcept {
    if (failed_) return failure_;
    if (!staged_write()) {
        return current(UploadStatus::DataWithoutLaunch,
                       "inline upload has no completed write to commit");
    }
    active_ = false;
    payload_.clear();
    return current(UploadStatus::Complete);
}

}  // namespace armigo::graphics
