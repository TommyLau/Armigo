// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/horizon/read_only_shared_memory.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <variant>

namespace armigo::services::pl {

inline constexpr std::size_t shared_font_count = 6;
inline constexpr std::size_t shared_font_memory_size = 0x1100000;

struct SharedFont {
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
};

struct SharedFonts {
    std::array<SharedFont, shared_font_count> fonts{};
    std::shared_ptr<horizon::ReadOnlySharedMemory> memory;
};

// Loads six user-supplied firmware BFTTF files from system/<SystemDataID> into
// the firmware shared layout. The order is standard, simplified, simplified
// extension, traditional, Korean, then Nintendo extended.
std::variant<SharedFonts, std::string> load_shared_fonts(
    const std::filesystem::path& directory);

class SharedFontSession final : public horizon::ClientSession {
public:
    SharedFontSession(std::uint64_t process_id, SharedFonts fonts);

    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;

private:
    horizon::IpcReply method(const horizon::IpcMessage&, const horizon::GuestBuffers*);

    const std::uint64_t process_id_;
    const SharedFonts fonts_;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::pl
