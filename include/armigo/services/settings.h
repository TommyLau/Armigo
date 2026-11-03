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

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace armigo::services::set {

// Independent launch-time console settings shared by set:GetLanguageCode/GetRegionCode.
bool configure_console_language(std::string_view language) noexcept;
bool configure_console_region(std::uint32_t region) noexcept;
std::uint64_t console_language_code() noexcept;
std::uint32_t console_region_code() noexcept;

// The title's NACP SupportedLanguageFlag selects AM's desired language.
void configure_supported_languages(std::uint32_t supported_language_flag) noexcept;
// The application language the applet manager reports: the firmware's
// selection from the system language and the title's supported-language
// mask; nullopt when the system language is not an application language or
// the mask allows none of its order.
std::optional<std::uint64_t> desired_console_language() noexcept;
// The Switch system language a host locale (BCP 47, such as "zh-Hant-TW" or
// "en_AU") stands for, with an explicit script deciding Chinese; empty when
// it has none.
std::string host_system_language(std::string_view locale);
// The title's language for a system language: the first language in the
// firmware's priority order for it that the supported-language mask allows
// (0 allows every language).
std::optional<std::string> title_language(std::string_view system_language, std::uint32_t supported_language_flag);
// Automatic choice: the first preferred host locale that stands for a
// system language chooses through title_language(); with none, American
// English's order.
std::optional<std::string> automatic_language(std::span<const std::string_view> preferred,
                                              std::uint32_t supported_language_flag);
// The region of a language's market: Japan for Japanese; the Americas for
// American English, Canadian French, Latin American Spanish and Brazilian
// Portuguese; Europe for British English and the European languages; Hong
// Kong, Taiwan and Korea for Chinese and Korean. Never China.
std::uint32_t market_region(std::string_view language) noexcept;

class SettingsSession final : public horizon::ClientSession {
public:
    explicit SettingsSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;

private:
    horizon::IpcReply method(const horizon::IpcMessage&, const horizon::GuestBuffers*);
    const std::uint64_t process_id_;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::set
