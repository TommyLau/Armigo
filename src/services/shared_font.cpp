// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/shared_font.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace armigo::services::pl {
namespace {
using namespace horizon;

constexpr std::uint32_t font_key = 0x06186249;
constexpr std::uint32_t bfttf_magic = 0x18029a7f;
constexpr std::uint32_t size_key = 0x49621806;
constexpr std::array<std::string_view, shared_font_count> directory_names{
    "0100000000000811", "0100000000000814", "0100000000000814",
    "0100000000000813", "0100000000000812", "0100000000000810"};
constexpr std::array<std::string_view, shared_font_count> file_names{
    "nintendo_udsg-r_std_003.bfttf",
    "nintendo_udsg-r_org_zh-cn_003.bfttf",
    "nintendo_udsg-r_ext_zh-cn_003.bfttf",
    "nintendo_udjxh-db_zh-tw_003.bfttf",
    "nintendo_udsg-r_ko_003.bfttf",
    "nintendo_ext_003.bfttf",
};

std::uint32_t read_u32(const std::byte* bytes) {
    return std::to_integer<std::uint32_t>(bytes[0]) |
        (std::to_integer<std::uint32_t>(bytes[1]) << 8) |
        (std::to_integer<std::uint32_t>(bytes[2]) << 16) |
        (std::to_integer<std::uint32_t>(bytes[3]) << 24);
}

void write_u32(std::byte* bytes, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        bytes[index] = static_cast<std::byte>(value >> (8 * index));
    }
}

std::uint32_t read(const IpcMessage& bytes, unsigned offset) {
    return read_u32(bytes.data() + offset);
}

void write(IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    write_u32(bytes.data() + offset, value);
}

std::variant<std::vector<std::byte>, std::string> decrypt_font(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return "Cannot read shared font " + path.filename().string() + ".";
    const auto end = input.tellg();
    if (end < 8 || (end & 3) != 0 || end > 0x1000000) {
        return "Shared font " + path.filename().string() + " has an invalid BFTTF size.";
    }
    std::vector<std::byte> encoded(static_cast<std::size_t>(end));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char*>(encoded.data()), end)) {
        return "Cannot read shared font " + path.filename().string() + ".";
    }
    if ((read_u32(encoded.data()) ^ font_key) != bfttf_magic) {
        return "Shared font " + path.filename().string() + " has an invalid BFTTF header.";
    }
    std::vector<std::byte> decoded(encoded.size() - 8);
    for (std::size_t offset = 8; offset < encoded.size(); offset += 4) {
        write_u32(decoded.data() + offset - 8, read_u32(encoded.data() + offset) ^ font_key);
    }
    return decoded;
}

IpcReply reply_with(unsigned output_words) {
    IpcReply reply{IpcDisposition::handled};
    reply.size = 8 + 4 * (8 + output_words);
    write(reply.bytes, 4, 8 + output_words);
    write(reply.bytes, 16, 0x4f434653);
    return reply;
}
} // namespace

std::variant<SharedFonts, std::string> load_shared_fonts(
    const std::filesystem::path& directory) {
    SharedFonts result;
    std::vector<std::byte> payload(shared_font_memory_size);
    std::size_t next = 0;
    for (std::size_t index = 0; index < file_names.size(); ++index) {
        auto decoded = decrypt_font(directory / directory_names[index] / file_names[index]);
        if (const auto* error = std::get_if<std::string>(&decoded)) return *error;
        const auto& bytes = std::get<std::vector<std::byte>>(decoded);
        if (next > shared_font_memory_size - 8 ||
            bytes.size() > shared_font_memory_size - next - 8) {
            return "The shared fonts exceed the firmware shared-memory size.";
        }
        write_u32(payload.data() + next, bfttf_magic);
        // The header the platform service leaves before each font: the
        // decrypted magic, then the font's size encrypted with the key.
        write_u32(payload.data() + next + 4, static_cast<std::uint32_t>(bytes.size()) ^ size_key);
        std::copy(bytes.begin(), bytes.end(), payload.begin() + static_cast<std::ptrdiff_t>(next + 8));
        result.fonts[index] = {static_cast<std::uint32_t>(next + 8),
                               static_cast<std::uint32_t>(bytes.size())};
        next += 8 + bytes.size();
    }
    result.memory = std::make_shared<ReadOnlySharedMemory>(payload);
    return result;
}

SharedFontSession::SharedFontSession(std::uint64_t process_id, SharedFonts fonts)
    : process_id_(process_id), fonts_(std::move(fonts)) {
    if (!fonts_.memory || fonts_.memory->bytes().size() != shared_font_memory_size) {
        throw std::invalid_argument("Shared-font service requires its firmware shared-memory payload");
    }
}

IpcReply SharedFontSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return request_with_buffers(message, sender_pid, {});
}

IpcReply SharedFontSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                                 const GuestBuffers& buffers) {
    const std::lock_guard lock(mutex_);
    if (sender_pid != process_id_) {
        diagnostic_ = "Shared-font session belongs to a different process";
        return {};
    }
    return method(message, buffers.empty() ? nullptr : &buffers);
}

IpcReply SharedFontSession::method(const IpcMessage& message, const GuestBuffers* buffers) {
    diagnostic_ = "Shared-font method or packet is unsupported";
    const auto type = read(message, 0) & 0xffffu;
    if (type == 5 || type == 7) {
        if (read(message, 16) != 0x49434653 || read(message, 24) != 3) return {};
        auto reply = reply_with(1);
        write(reply.bytes, 32, 0);
        diagnostic_ = nullptr;
        return reply;
    }
    if (type != 4 && type != 6) return {};
    const auto cmif = locate_cmif(message);
    if (!cmif || read(message, static_cast<unsigned>(*cmif)) != 0x49434653) return {};
    const auto command = read(message, static_cast<unsigned>(*cmif) + 8);
    const auto input = read(message, static_cast<unsigned>(*cmif) + 16);
    if (command <= 3 && input >= shared_font_count) return {};
    if (command == 0) {
        auto reply = reply_with(0);
        diagnostic_ = nullptr;
        return reply;
    }
    if (command == 1) {
        auto reply = reply_with(1);
        write(reply.bytes, 32, 1);
        diagnostic_ = nullptr;
        return reply;
    }
    if (command == 2 || command == 3) {
        auto reply = reply_with(1);
        write(reply.bytes, 32, command == 2 ? fonts_.fonts[input].size : fonts_.fonts[input].offset);
        diagnostic_ = nullptr;
        return reply;
    }
    if (command == 4) {
        auto reply = reply_with(0);
        reply.transferred[0] = fonts_.memory;
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
        diagnostic_ = nullptr;
        return reply;
    }
    if (command == 5 || command == 6) {
        if (!buffers || buffers->size() != 3) return {};
        std::uint32_t loaded = 0;
        for (std::size_t index = 0; index < shared_font_count; ++index) {
            const std::size_t offset = index * 4;
            bool complete = true;
            for (std::size_t buffer = 0; buffer < 3; ++buffer) {
                if (!(*buffers)[buffer].output || offset + 4 > (*buffers)[buffer].size) complete = false;
            }
            if (!complete) break;
            std::array<std::byte, 4> value;
            write_u32(value.data(), static_cast<std::uint32_t>(index));
            if (!buffers->write_at(0, offset, value)) return {};
            write_u32(value.data(), fonts_.fonts[index].offset);
            if (!buffers->write_at(1, offset, value)) return {};
            write_u32(value.data(), fonts_.fonts[index].size);
            if (!buffers->write_at(2, offset, value)) return {};
            ++loaded;
        }
        auto reply = reply_with(2);
        reply.bytes[32] = std::byte{1};
        write(reply.bytes, 36, loaded);
        diagnostic_ = nullptr;
        return reply;
    }
    return {};
}

const char* SharedFontSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

} // namespace armigo::services::pl
