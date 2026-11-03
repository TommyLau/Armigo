// ============================================================================
// Armigo Tests - Horizon Service Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/shared_font.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
using namespace armigo::horizon;
using namespace armigo::services::pl;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #c); ++failures; } } while (false)

constexpr std::array<std::string_view, shared_font_count> directories{
    "0100000000000811", "0100000000000814", "0100000000000814",
    "0100000000000813", "0100000000000812", "0100000000000810"};
constexpr std::array<std::string_view, shared_font_count> names{
    "nintendo_udsg-r_std_003.bfttf", "nintendo_udsg-r_org_zh-cn_003.bfttf",
    "nintendo_udsg-r_ext_zh-cn_003.bfttf", "nintendo_udjxh-db_zh-tw_003.bfttf",
    "nintendo_udsg-r_ko_003.bfttf", "nintendo_ext_003.bfttf"};

void put_u32(std::vector<std::byte>& bytes, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) bytes.push_back(static_cast<std::byte>(value >> (8 * index)));
}

std::uint32_t word(const std::byte* bytes) {
    return std::to_integer<std::uint32_t>(bytes[0]) |
        (std::to_integer<std::uint32_t>(bytes[1]) << 8) |
        (std::to_integer<std::uint32_t>(bytes[2]) << 16) |
        (std::to_integer<std::uint32_t>(bytes[3]) << 24);
}

void put(IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) bytes[offset + index] = static_cast<std::byte>(value >> (8 * index));
}

IpcMessage plain(std::uint32_t command, std::uint32_t input = 0) {
    IpcMessage bytes{};
    put(bytes, 0, 6);
    put(bytes, 4, 10);
    put(bytes, 16, 0x49434653);
    put(bytes, 20, 1);
    put(bytes, 24, command);
    put(bytes, 28, 0);
    put(bytes, 32, input);
    return bytes;
}

void write_font(const std::filesystem::path& path, std::uint32_t seed, std::size_t words) {
    std::vector<std::byte> bytes;
    put_u32(bytes, 0x18029a7f ^ 0x06186249);
    put_u32(bytes, 0);
    for (std::size_t index = 0; index < words; ++index) put_u32(bytes, (seed + index) ^ 0x06186249);
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

struct Memory {
    std::array<std::byte, 24> types{};
    std::array<std::byte, 24> offsets{};
    std::array<std::byte, 24> sizes{};
    static bool read(void*, std::uint64_t, std::span<std::byte>) noexcept { return false; }
    static bool write(void* context, std::uint64_t address, std::span<const std::byte> input) noexcept {
        auto& self = *static_cast<Memory*>(context);
        if (address + input.size() > 72) return false;
        if (address < 24) std::memcpy(self.types.data() + address, input.data(), input.size());
        else if (address < 48) std::memcpy(self.offsets.data() + address - 24, input.data(), input.size());
        else std::memcpy(self.sizes.data() + address - 48, input.data(), input.size());
        return true;
    }
};

void test_shared_fonts() {
    const auto directory = std::filesystem::temp_directory_path() / "armigo_shared_fonts";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    for (std::size_t index = 0; index < names.size(); ++index) {
        std::filesystem::create_directories(directory / directories[index]);
        write_font(directory / directories[index] / names[index], 0x1000 + index, index + 1);
    }
    auto loaded = load_shared_fonts(directory);
    CHECK(std::holds_alternative<SharedFonts>(loaded));
    if (const auto* fonts = std::get_if<SharedFonts>(&loaded)) {
        CHECK(fonts->memory && fonts->memory->bytes().size() == shared_font_memory_size);
        std::size_t next = 0;
        for (std::size_t index = 0; index < names.size(); ++index) {
            CHECK(fonts->fonts[index].offset == next + 8);
            CHECK(fonts->fonts[index].size == (index + 1) * 4);
            CHECK(word(fonts->memory->bytes().data() + next) == 0x18029a7f);
            // The size word: the font's size encrypted with the key, a little-endian word.
            CHECK(word(fonts->memory->bytes().data() + next + 4) ==
                  (static_cast<std::uint32_t>(fonts->fonts[index].size) ^ 0x49621806u));
            CHECK(word(fonts->memory->bytes().data() + fonts->fonts[index].offset) == 0x1000 + index);
            next += 8 + fonts->fonts[index].size;
        }
        SharedFontSession session(91, *fonts);
        auto reply = session.request(plain(0, 1), 91);
        CHECK(reply.disposition == IpcDisposition::handled && reply.size == 40);
        reply = session.request(plain(1, 1), 91);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes.data() + 32) == 1);
        reply = session.request(plain(2, 1), 91);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes.data() + 32) == 8);
        reply = session.request(plain(3, 1), 91);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes.data() + 32) == 20);
        reply = session.request(plain(4), 91);
        CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1 &&
              reply.transferred_as_copy && reply.transferred[0] == fonts->memory);
        CHECK(session.request(plain(2, 6), 91).disposition == IpcDisposition::unsupported);
        CHECK(session.request(plain(0), 92).disposition == IpcDisposition::unsupported);

        for (const auto command : {5u, 6u}) {
            Memory memory;
            GuestBuffers buffers(Memory::read, Memory::write, &memory);
            buffers.add({0, 24, false, true});
            buffers.add({24, 24, false, true});
            buffers.add({48, 24, false, true});
            reply = session.request_with_buffers(plain(command), 91, buffers);
            CHECK(reply.disposition == IpcDisposition::handled && reply.bytes[32] == std::byte{1} &&
                  word(reply.bytes.data() + 36) == shared_font_count);
            for (std::size_t index = 0; index < shared_font_count; ++index) {
                CHECK(word(memory.types.data() + index * 4) == index);
                CHECK(word(memory.offsets.data() + index * 4) == fonts->fonts[index].offset);
                CHECK(word(memory.sizes.data() + index * 4) == fonts->fonts[index].size);
            }
        }
        Memory partial;
        GuestBuffers buffers(Memory::read, Memory::write, &partial);
        buffers.add({0, 24, false, true});
        buffers.add({24, 4, false, true});
        buffers.add({48, 24, false, true});
        reply = session.request_with_buffers(plain(5), 91, buffers);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes.data() + 36) == 1);
        CHECK(session.request(plain(5), 91).disposition == IpcDisposition::unsupported);
    }
    std::ofstream(directory / directories[0] / names[0], std::ios::binary | std::ios::trunc).write("bad", 3);
    CHECK(std::holds_alternative<std::string>(load_shared_fonts(directory)));
    std::filesystem::remove_all(directory);
}
} // namespace

int main() {
    test_shared_fonts();
    if (failures) return 1;
    std::puts("shared-font checks passed");
    return 0;
}
