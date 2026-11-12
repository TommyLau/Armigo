// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/services/settings.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::services::set {
namespace {
using namespace horizon;
using namespace horizon::cmif;

// A language code is its name in at most eight bytes, zero padded.
constexpr std::uint64_t code(std::string_view name) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i != name.size(); ++i) value |= std::uint64_t(static_cast<unsigned char>(name[i])) << (8 * i);
    return value;
}
// The firmware's available languages in its order; the original list is the
// first 15.
constexpr std::array<std::uint64_t, 18> available_languages{
    code("ja"), code("en-US"), code("fr"), code("de"), code("it"), code("es"),
    code("zh-CN"), code("ko"), code("nl"), code("pt"), code("ru"), code("zh-TW"),
    code("en-GB"), code("fr-CA"), code("es-419"), code("zh-Hans"), code("zh-Hant"), code("pt-BR")};
constexpr std::array<std::uint64_t, 16> nacp_language_codes{
    code("en-US"), code("en-GB"), code("ja"), code("fr"), code("de"), code("es-419"),
    code("es"), code("it"), code("nl"), code("fr-CA"), code("pt"), code("ru"),
    code("ko"), code("zh-Hant"), code("zh-Hans"), code("pt-BR")};
constexpr std::uint8_t nacp_priority[16][16] = {
    {0, 1, 5, 9, 15, 3, 4, 6, 7, 8, 10, 11, 2, 14, 13, 12},
    {1, 0, 3, 4, 6, 7, 8, 10, 11, 5, 9, 15, 2, 14, 13, 12},
    {2, 0, 1, 5, 9, 15, 3, 4, 6, 7, 8, 10, 11, 14, 13, 12},
    {3, 9, 1, 0, 4, 6, 7, 8, 10, 11, 5, 15, 2, 14, 13, 12},
    {4, 1, 0, 3, 6, 7, 8, 10, 11, 5, 9, 15, 2, 14, 13, 12},
    {5, 6, 0, 1, 15, 10, 9, 3, 7, 4, 8, 11, 2, 14, 13, 12},
    {6, 5, 1, 0, 3, 4, 7, 8, 10, 11, 9, 15, 2, 14, 13, 12},
    {7, 1, 0, 3, 4, 6, 8, 10, 11, 5, 9, 15, 2, 14, 13, 12},
    {8, 1, 0, 3, 4, 6, 7, 10, 11, 5, 9, 15, 2, 14, 13, 12},
    {9, 3, 0, 1, 5, 6, 4, 7, 8, 15, 10, 11, 2, 14, 13, 12},
    {10, 15, 1, 0, 3, 4, 6, 7, 8, 11, 5, 9, 2, 14, 13, 12},
    {11, 1, 0, 3, 4, 6, 7, 8, 10, 5, 9, 15, 2, 14, 13, 12},
    {12, 0, 1, 5, 9, 15, 3, 4, 6, 7, 8, 10, 11, 2, 14, 13},
    {13, 14, 0, 1, 2, 5, 9, 15, 3, 4, 6, 7, 8, 10, 11, 12},
    {14, 13, 0, 1, 2, 5, 9, 15, 3, 4, 6, 7, 8, 10, 11, 12},
    {15, 10, 0, 1, 5, 6, 9, 3, 4, 7, 8, 11, 2, 14, 13, 12},
};
constexpr std::size_t original_language_count = 15;
std::atomic<std::uint64_t> selected_language{code("en-US")};
std::atomic<std::uint32_t> selected_region{1};  // USA
std::atomic<std::uint32_t> supported_languages{0xffff};

// A non-domain reply with `output_words` of output after the CMIF header. Both
// 23.0.1 settings programs write the interface word 0x7f7baf0a after a
// method's result; control replies leave it zero.
IpcReply reply_with(unsigned output_words, bool method = true) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 8 + 4 * (8 + output_words);
    write(reply.bytes, 4, 8 + output_words);
    write(reply.bytes, 16, 0x4f434653);
    if (method) write(reply.bytes, 28, 0x7f7baf0a);
    return reply;
}
} // namespace

bool configure_console_language(std::string_view language) noexcept {
    if (language.size() > 8) return false;
    const std::uint64_t selected = code(language);
    const auto known = std::find(available_languages.begin(), available_languages.end(), selected);
    if (known == available_languages.end()) return false;
    selected_language.store(selected, std::memory_order_relaxed);
    return true;
}

bool configure_console_region(std::uint32_t region) noexcept {
    if (region > 5) return false;
    selected_region.store(region, std::memory_order_relaxed);
    return true;
}

std::uint64_t console_language_code() noexcept {
    return selected_language.load(std::memory_order_relaxed);
}

std::uint32_t console_region_code() noexcept {
    return selected_region.load(std::memory_order_relaxed);
}

void configure_supported_languages(std::uint32_t supported_language_flag) noexcept {
    supported_languages.store(supported_language_flag, std::memory_order_relaxed);
}

namespace {
// The firmware's desired-language selection: the system language's NACP
// index, then the first language in its priority row that the supported-
// language mask allows. A zero mask allows every language.
std::optional<std::size_t> select_title_language(std::uint64_t system_language, std::uint32_t mask) noexcept {
    const auto language = std::find(nacp_language_codes.begin(), nacp_language_codes.end(), system_language);
    if (language == nacp_language_codes.end()) return std::nullopt;
    const auto selected_index = static_cast<std::size_t>(language - nacp_language_codes.begin());
    for (const std::uint8_t candidate : nacp_priority[selected_index]) {
        if (mask == 0 || (mask & (std::uint32_t{1} << candidate)) != 0) return candidate;
    }
    return std::nullopt;
}

std::string code_text(std::uint64_t value) {
    std::string text;
    for (unsigned i = 0; i != 8 && ((value >> (8 * i)) & 0xff); ++i)
        text += static_cast<char>((value >> (8 * i)) & 0xff);
    return text;
}

bool is_one_of(std::string_view value, std::initializer_list<std::string_view> set) {
    return std::find(set.begin(), set.end(), value) != set.end();
}
} // namespace

std::optional<std::uint64_t> desired_console_language() noexcept {
    const auto index = select_title_language(console_language_code(), supported_languages.load(std::memory_order_relaxed));
    if (!index) return std::nullopt;
    return nacp_language_codes[*index];
}

std::string host_system_language(std::string_view locale) {
    // Subtags of a BCP 47 identifier ("zh-Hant-TW", "en_GB", "es-419"): the
    // language, an optional four-letter script and an optional region.
    std::vector<std::string> parts;
    std::string part;
    for (const unsigned char character : locale) {
        if (character == '-' || character == '_') {
            if (!part.empty()) parts.push_back(part);
            part.clear();
        } else if (std::isalnum(character)) {
            part += static_cast<char>(std::tolower(character));
        } else {
            return {};
        }
    }
    if (!part.empty()) parts.push_back(part);
    if (parts.empty()) return {};
    const std::string& language = parts[0];
    std::string script, region;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (parts[i].size() == 4 && script.empty() && region.empty()) script = parts[i];
        else if ((parts[i].size() == 2 || parts[i].size() == 3) && region.empty()) region = parts[i];
    }
    if (language == "zh") {
        // An explicit script decides; otherwise the region's script.
        if (script == "hant") return "zh-Hant";
        if (script == "hans") return "zh-Hans";
        return is_one_of(region, {"tw", "hk", "mo"}) ? "zh-Hant" : "zh-Hans";
    }
    if (language == "en")
        return is_one_of(region, {"gb", "ie", "au", "nz", "za", "in", "sg", "hk", "mt", "gi"}) ? "en-GB" : "en-US";
    if (language == "fr") return region == "ca" ? "fr-CA" : "fr";
    if (language == "es") return region.empty() || region == "es" ? "es" : "es-419";
    if (language == "pt") return region == "br" ? "pt-BR" : "pt";
    if (is_one_of(language, {"ja", "ko", "de", "it", "nl", "ru"})) return language;
    return {};
}

std::optional<std::string> title_language(std::string_view system_language, std::uint32_t supported_language_flag) {
    if (system_language.size() > 8) return std::nullopt;
    const auto index = select_title_language(code(system_language), supported_language_flag);
    if (!index) return std::nullopt;
    return code_text(nacp_language_codes[*index]);
}

std::optional<std::string> automatic_language(std::span<const std::string_view> preferred,
                                              std::uint32_t supported_language_flag) {
    for (const auto locale : preferred) {
        const auto system = host_system_language(locale);
        if (system.empty()) continue;
        if (auto chosen = title_language(system, supported_language_flag)) return chosen;
    }
    return title_language("en-US", supported_language_flag);
}

std::uint32_t market_region(std::string_view language) noexcept {
    if (language == "ja") return 0;
    if (is_one_of(language, {"en-GB", "fr", "de", "it", "es", "nl", "pt", "ru"})) return 2;
    if (is_one_of(language, {"zh-Hans", "zh-Hant", "ko"})) return 4;
    return 1;  // en-US, fr-CA, es-419, pt-BR and anything else: the Americas
}

SettingsSession::SettingsSession(std::uint64_t process_id) : process_id_(process_id) {}

horizon::IpcReply SettingsSession::request(const horizon::IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    if (sender_pid != process_id_) {
        diagnostic_ = "Settings session belongs to a different process";
        return {};
    }
    return method(message, nullptr);
}

horizon::IpcReply SettingsSession::request_with_buffers(const horizon::IpcMessage& message,
                                                        std::uint64_t sender_pid,
                                                        const horizon::GuestBuffers& buffers) {
    const std::lock_guard lock(mutex_);
    if (sender_pid != process_id_) {
        diagnostic_ = "Settings session belongs to a different process";
        return {};
    }
    return method(message, buffers.empty() ? nullptr : &buffers);
}

horizon::IpcReply SettingsSession::method(const horizon::IpcMessage& message, const horizon::GuestBuffers* buffers) {
    diagnostic_ = "Settings method or packet is unsupported";
    const auto type = read(message, 0) & 0xffffu;
    if (type == 5 || type == 7) {
        if (read(message, 16) != 0x49434653 || read(message, 24) != 3) return {};
        // QueryPointerBufferSize: this endpoint has no pointer buffer.
        auto reply = reply_with(1, false);
        write(reply.bytes, 32, 0);
        diagnostic_ = nullptr;
        return reply;
    }
    if (type != 4 && type != 6) return {};
    const auto cmif = locate_cmif(message);
    if (!cmif || read(message, static_cast<unsigned>(*cmif)) != 0x49434653) return {};
    const auto command = read(message, static_cast<unsigned>(*cmif) + 8);
    switch (command) {
    case 0: {  // GetLanguageCode
        auto reply = reply_with(2);
        const std::uint64_t language = console_language_code();
        write(reply.bytes, 32, static_cast<std::uint32_t>(language));
        write(reply.bytes, 36, static_cast<std::uint32_t>(language >> 32));
        diagnostic_ = nullptr;
        return reply;
    }
    case 3:    // GetAvailableLanguageCodeCount
    case 4:    // GetRegionCode
    case 6: {  // GetAvailableLanguageCodeCount2
        auto reply = reply_with(1);
        write(reply.bytes, 32, command == 4 ? console_region_code()
                               : static_cast<std::uint32_t>(command == 3 ? original_language_count
                                                                       : available_languages.size()));
        diagnostic_ = nullptr;
        return reply;
    }
    case 1:    // GetAvailableLanguageCodes
    case 5: {  // GetAvailableLanguageCodes2
        if (!buffers || buffers->size() != 1 || !(*buffers)[0].output) {
            diagnostic_ = "Settings language list needs one output buffer";
            return {};
        }
        const std::size_t available = command == 1 ? original_language_count : available_languages.size();
        const std::size_t count = std::min<std::size_t>(available, (*buffers)[0].size / 8);
        for (std::size_t i = 0; i != count; ++i) {
            std::array<std::byte, 8> bytes;
            std::memcpy(bytes.data(), &available_languages[i], 8);
            if (!buffers->write_at(0, i * 8, bytes)) {
                diagnostic_ = "Settings language list could not be written";
                return {};
            }
        }
        auto reply = reply_with(1);
        write(reply.bytes, 32, static_cast<std::uint32_t>(count));
        diagnostic_ = nullptr;
        return reply;
    }
    default:
        return {};
    }
}

const char* SettingsSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::set
