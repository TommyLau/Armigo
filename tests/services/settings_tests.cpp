// ============================================================================
// Armigo Tests - Horizon Service Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/settings.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
using namespace armigo::horizon;
using armigo::services::set::SettingsSession;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #c); ++failures; } } while (false)

void put(IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
std::uint32_t word(const IpcMessage& bytes, unsigned offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t{std::to_integer<unsigned char>(bytes[offset + i])} << (8 * i);
    return value;
}
// A non-domain method without descriptors: CMIF at byte 16.
IpcMessage plain(std::uint32_t command) {
    IpcMessage bytes{};
    bytes.fill(std::byte{0xa5});
    put(bytes, 0, 6);
    put(bytes, 4, 8);
    put(bytes, 16, 0x49434653);
    put(bytes, 20, 1);
    put(bytes, 24, command);
    put(bytes, 28, 0);
    return bytes;
}
// A non-domain method with one receive buffer: the descriptor at byte 8, CMIF at 32.
IpcMessage with_receive_buffer(std::uint32_t command) {
    IpcMessage bytes{};
    bytes.fill(std::byte{0xa5});
    put(bytes, 0, 0x01000006);
    put(bytes, 4, 8);
    put(bytes, 32, 0x49434653);
    put(bytes, 36, 1);
    put(bytes, 40, command);
    put(bytes, 44, 0);
    return bytes;
}

struct Memory {
    std::array<std::byte, 0x200> bytes{};
    static bool read(void* context, std::uint64_t address, std::span<std::byte> value) noexcept {
        auto& self = *static_cast<Memory*>(context);
        if (address + value.size() > self.bytes.size()) return false;
        std::memcpy(value.data(), self.bytes.data() + address, value.size());
        return true;
    }
    static bool write(void* context, std::uint64_t address, std::span<const std::byte> value) noexcept {
        auto& self = *static_cast<Memory*>(context);
        if (address + value.size() > self.bytes.size()) return false;
        std::memcpy(self.bytes.data() + address, value.data(), value.size());
        return true;
    }
    std::string code(std::size_t index) const {
        std::string text(reinterpret_cast<const char*>(bytes.data() + index * 8), 8);
        return text.substr(0, text.find('\0'));
    }
};

void test_settings() {
    SettingsSession session(37);
    // QueryPointerBufferSize.
    IpcMessage control{};
    put(control, 0, 5);
    put(control, 4, 8);
    put(control, 16, 0x49434653);
    put(control, 24, 3);
    auto reply = session.request(control, 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 44 && word(reply.bytes, 32) == 0);

    reply = session.request(plain(0), 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 48);
    CHECK(word(reply.bytes, 16) == 0x4f434653 && word(reply.bytes, 24) == 0);
    CHECK(word(reply.bytes, 32) == 0x552d6e65 && word(reply.bytes, 36) == 0x53);  // en-US
    reply = session.request(plain(4), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 32) == 1);
    reply = session.request(plain(3), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 32) == 15);
    reply = session.request(plain(6), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 32) == 18);

    // Language lists fill the output buffer up to its capacity.
    for (const auto [command, capacity, expected] : {std::array<unsigned, 3>{5, 0x200, 18},
                                                     std::array<unsigned, 3>{1, 0x200, 15},
                                                     std::array<unsigned, 3>{5, 20, 2}}) {
        Memory memory;
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0, capacity, false, true});
        reply = session.request_with_buffers(with_receive_buffer(command), 37, buffers);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 32) == expected);
        CHECK(memory.code(0) == "ja" && memory.code(1) == "en-US");
        if (expected == 18) CHECK(memory.code(16) == "zh-Hant" && memory.code(17) == "pt-BR");
        CHECK(memory.code(expected) == "");
    }

    {
        using armigo::services::set::host_system_language;
        using armigo::services::set::title_language;
        using armigo::services::set::automatic_language;
        using armigo::services::set::market_region;
        // Host locales: an explicit Chinese script decides, then the region.
        CHECK(host_system_language("zh-Hans-CN") == "zh-Hans" && host_system_language("zh-Hans-TW") == "zh-Hans");
        CHECK(host_system_language("zh-Hant-HK") == "zh-Hant" && host_system_language("zh-TW") == "zh-Hant" &&
              host_system_language("zh_HK") == "zh-Hant" && host_system_language("zh-MO") == "zh-Hant");
        CHECK(host_system_language("zh-CN") == "zh-Hans" && host_system_language("zh-SG") == "zh-Hans" &&
              host_system_language("zh") == "zh-Hans");
        CHECK(host_system_language("en-US") == "en-US" && host_system_language("en") == "en-US" &&
              host_system_language("en-GB") == "en-GB" && host_system_language("en-AU") == "en-GB");
        CHECK(host_system_language("pt-BR") == "pt-BR" && host_system_language("pt-PT") == "pt");
        CHECK(host_system_language("es-419") == "es-419" && host_system_language("es-MX") == "es-419" &&
              host_system_language("es-ES") == "es" && host_system_language("es") == "es");
        CHECK(host_system_language("fr-CA") == "fr-CA" && host_system_language("fr-FR") == "fr");
        CHECK(host_system_language("ja-JP") == "ja" && host_system_language("ko-KR") == "ko" &&
              host_system_language("de-DE") == "de" && host_system_language("ru") == "ru");
        CHECK(host_system_language("th-TH").empty() && host_system_language("").empty());
        // The firmware's rows: German prefers British English, Simplified
        // Chinese prefers Traditional Chinese then American English, and a
        // zero mask allows every language.
        CHECK(title_language("de", 0b11) == "en-GB");
        CHECK(title_language("zh-Hans", (1u << 13) | 1u) == "zh-Hant");
        CHECK(title_language("zh-Hans", 1u | (1u << 2)) == "en-US");
        CHECK(title_language("ja", 0) == "ja");
        CHECK(!title_language("zh-CN", 0));
        CHECK(!title_language("en-US", 1u << 16));
        // Auto: the first locale with a system language; none, American English's order.
        const std::string_view preferred[] = {"th-TH", "zh-Hant-TW", "en-US"};
        CHECK(automatic_language(preferred, 0xffff) == "zh-Hant");
        const std::string_view unmapped[] = {"th-TH"};
        CHECK(automatic_language(unmapped, 1u << 2) == "ja");
        // Markets, never China.
        CHECK(market_region("ja") == 0);
        CHECK(market_region("en-US") == 1 && market_region("fr-CA") == 1 && market_region("es-419") == 1 &&
              market_region("pt-BR") == 1);
        CHECK(market_region("en-GB") == 2 && market_region("fr") == 2 && market_region("de") == 2 &&
              market_region("it") == 2 && market_region("es") == 2 && market_region("nl") == 2 &&
              market_region("pt") == 2 && market_region("ru") == 2);
        CHECK(market_region("zh-Hans") == 4 && market_region("zh-Hant") == 4 && market_region("ko") == 4);
    }

    CHECK(armigo::services::set::configure_console_region(0));
    CHECK(armigo::services::set::configure_console_language("ja"));
    reply = session.request(plain(0), 37);
    CHECK(word(reply.bytes, 32) == 0x616a && word(reply.bytes, 36) == 0);
    reply = session.request(plain(4), 37);
    CHECK(word(reply.bytes, 32) == 0);
    CHECK(!armigo::services::set::configure_console_language("en-XX"));
    reply = session.request(plain(0), 37);
    CHECK(word(reply.bytes, 32) == 0x616a);
    CHECK(!armigo::services::set::configure_console_region(6));
    CHECK(armigo::services::set::configure_console_region(4));
    CHECK(armigo::services::set::configure_console_language("zh-Hans"));
    CHECK(armigo::services::set::console_region_code() == 4);
    CHECK(armigo::services::set::configure_console_region(2));
    CHECK(armigo::services::set::configure_console_language("zh-Hant"));
    CHECK(armigo::services::set::console_region_code() == 2);
    reply = session.request(plain(4), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 32) == 2);

    const auto language_code = [](std::string_view text) {
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < text.size(); ++index) {
            value |= std::uint64_t(static_cast<unsigned char>(text[index])) << (8 * index);
        }
        return value;
    };
    const std::string_view nacp_codes[] = {"en-US", "en-GB", "ja", "fr", "de", "es-419", "es", "it",
        "nl", "fr-CA", "pt", "ru", "ko", "zh-Hant", "zh-Hans", "pt-BR"};
    constexpr std::uint8_t priority[16][16] = {
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
    for (std::size_t selected = 0; selected < 16; ++selected) {
        CHECK(armigo::services::set::configure_console_language(nacp_codes[selected]));
        armigo::services::set::configure_supported_languages(0);
        CHECK(armigo::services::set::desired_console_language() == language_code(nacp_codes[selected]));
        for (const std::uint8_t candidate : priority[selected]) {
            armigo::services::set::configure_supported_languages(std::uint32_t{1} << candidate);
            CHECK(armigo::services::set::desired_console_language() == language_code(nacp_codes[candidate]));
        }
    }
    CHECK(armigo::services::set::configure_console_language("zh-CN"));
    armigo::services::set::configure_supported_languages(0);
    CHECK(!armigo::services::set::desired_console_language());
    CHECK(armigo::services::set::configure_console_language("en-US"));
    armigo::services::set::configure_supported_languages(0x30000);
    CHECK(!armigo::services::set::desired_console_language());
    armigo::services::set::configure_supported_languages(0xffff);
    CHECK(armigo::services::set::configure_console_language("en-US"));

    // Without its buffer, from another process, or for another method: unsupported.
    CHECK(session.request(with_receive_buffer(5), 37).disposition == IpcDisposition::unsupported);
    CHECK(session.request(plain(0), 38).disposition == IpcDisposition::unsupported);
    CHECK(session.request(plain(99), 37).disposition == IpcDisposition::unsupported);
    CHECK(session.diagnostic() != nullptr);
}
} // namespace

int main() {
    test_settings();
    if (failures) return 1;
    std::puts("settings checks passed");
    return 0;
}
