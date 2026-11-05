// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic CPK, CRILAYLA, GSC0 and HHNEEDLE fixtures. Every byte is constructed
// here; nothing is taken from a game.
#include "synthetic.h"
#include "test_support.h"

#include <armigo/shader_input/gfd_cache.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
namespace si = armigo::shader_input;
using armigo::shader_input::test::build_code;
using armigo::shader_input::test::build_control;
using armigo::shader_input::test::control_for;
using armigo::shader_input::test::CodeSpec;
using armigo::shader_input::test::must_parse;
using armigo::shader_input::test::span_of;

void put_be32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value >> 24));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8));
    bytes.push_back(static_cast<std::uint8_t>(value));
}

void put_le32_at(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        bytes[offset + index] = static_cast<std::byte>(value >> (8u * index));
    }
}

std::vector<std::byte> as_bytes(const std::vector<std::uint8_t>& bytes) {
    std::vector<std::byte> out(bytes.size());
    if (!bytes.empty()) std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

void scrub(std::vector<std::uint8_t>& bytes, std::size_t keep) {
    const std::uint8_t magic[] = {0x34, 0x12, 0x76, 0x98};
    const char needle[] = "HHNEEDLE";
    for (std::size_t index = 0; index + 4 <= bytes.size(); ++index) {
        if (index != keep && std::memcmp(bytes.data() + index, magic, 4) == 0) bytes[index] = 0;
    }
    for (std::size_t index = 0; index + 8 <= bytes.size(); ++index) {
        if (std::memcmp(bytes.data() + index, needle, 8) == 0) bytes[index] = 0;
    }
}

struct StageBytes {
    std::vector<std::uint8_t> control;
    std::vector<std::uint8_t> code;
};

StageBytes make_stage(std::uint32_t stage, const char* file, int line) {
    CodeSpec spec;
    spec.stage = stage;
    StageBytes out;
    out.code = build_code(spec);
    out.control = build_control(control_for(spec));
    scrub(out.control, 0);
    scrub(out.code, static_cast<std::size_t>(-1));
    const auto parsed = must_parse({span_of(out.control), span_of(out.code), {}}, file, line);
    if (parsed.control_size != out.control.size() || parsed.code_size != out.code.size()) {
        armigo::shader_input::test::record_failure(file, line, "synthetic stage lengths changed");
    }
    return out;
}

std::vector<std::uint8_t> descriptor_for(const StageBytes& stage) {
    const auto control_at = static_cast<std::uint32_t>(0x100);
    const auto code_at = control_at + static_cast<std::uint32_t>(stage.control.size());
    std::vector<std::uint8_t> descriptor(code_at + stage.code.size(), 0);
    auto put = [&](std::uint32_t offset, std::uint32_t value) {
        for (unsigned index = 0; index < 4; ++index) {
            descriptor[offset + index] = static_cast<std::uint8_t>(value >> (8u * index));
        }
    };
    put(0x30, control_at);
    put(0x34, code_at);
    put(0x38, static_cast<std::uint32_t>(stage.code.size()));
    put(0x3c, static_cast<std::uint32_t>(stage.control.size()));
    std::memcpy(descriptor.data() + control_at, stage.control.data(), stage.control.size());
    std::memcpy(descriptor.data() + code_at, stage.code.data(), stage.code.size());
    return descriptor;
}

std::vector<std::uint8_t> needle_package(const std::vector<std::uint8_t>& descriptor, int mode) {
    const auto descriptor_size = static_cast<std::uint32_t>(descriptor.size());
    const std::uint32_t extra = mode == 2 ? 8u : 0u;
    const std::uint32_t total = 44u + extra + descriptor_size;
    std::vector<std::uint8_t> out;
    auto text = [&](std::string_view value) {
        out.insert(out.end(), value.begin(), value.end());
    };
    text("HHNEEDLE");
    put_be32(out, total - 8);
    text(std::string_view("n\0\0\0", 4));
    text(mode == 2 ? "HNSHV002" : "HNSHV001");
    put_be32(out, total - 24);
    text(std::string_view("n\0\0\0", 4));
    put_be32(out, 0);
    if (mode == 2) {
        put_be32(out, 0);
        put_be32(out, 1);
    }
    put_be32(out, 8u + descriptor_size);
    put_be32(out, descriptor_size);
    out.insert(out.end(), descriptor.begin(), descriptor.end());
    return out;
}

std::vector<std::uint8_t> gsc_file(std::uint16_t type, const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint8_t> out(0x20, 0);
    out[0] = 'G';
    out[1] = 'F';
    out[2] = 'S';
    out[3] = '0';
    auto put_at = [&](std::size_t offset, std::uint32_t value) {
        out[offset] = static_cast<std::uint8_t>(value >> 24);
        out[offset + 1] = static_cast<std::uint8_t>(value >> 16);
        out[offset + 2] = static_cast<std::uint8_t>(value >> 8);
        out[offset + 3] = static_cast<std::uint8_t>(value);
    };
    put_at(0x10, 0x47534330u);
    put_at(0x14, 0x10000000u);
    put_at(0x18, 8u);
    out.insert(out.end(), 4, 0);
    out.push_back(static_cast<std::uint8_t>(type >> 8));
    out.push_back(static_cast<std::uint8_t>(type));
    put_be32(out, static_cast<std::uint32_t>(payload.size()));
    out.push_back(0);
    out.push_back(0);
    out.insert(out.end(), 24, 0);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

struct TempFile {
    std::string path;
    explicit TempFile(const std::vector<std::uint8_t>& bytes) {
        const auto pattern = (std::filesystem::temp_directory_path() / "armigo-gfd-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        const int fd = mkstemp(name.data());
        if (fd < 0) throw std::runtime_error("Cannot create a synthetic archive");
        if (!bytes.empty() && write(fd, bytes.data(), bytes.size()) != static_cast<ssize_t>(bytes.size())) {
            close(fd);
            unlink(name.data());
            throw std::runtime_error("Cannot write a synthetic archive");
        }
        close(fd);
        path.assign(name.data());
    }
    ~TempFile() {
        if (!path.empty()) unlink(path.c_str());
    }
};

std::vector<std::uint8_t> decode_hex(std::string_view text) {
    std::vector<std::uint8_t> out;
    out.reserve(text.size() / 2);
    auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        return -1;
    };
    for (std::size_t index = 0; index + 1 < text.size(); index += 2) {
        const int high = nibble(text[index]);
        const int low = nibble(text[index + 1]);
        if (high < 0 || low < 0) throw std::runtime_error("synthetic hex is malformed");
        out.push_back(static_cast<std::uint8_t>((high << 4) | low));
    }
    return out;
}

// One TOC row, constant header columns (row length 0), payload "hello-gsc".
const char kCpkHex[] =
    "43504b20ffffffff7600000000000000405554460000006e00000048000000480000006e00000000"
    "000400000000000136000000000000000000000100360000000a000000000000008b360000001200"
    "0000000000019b340000002000000001546f634f666673657400546f6353697a6500436f6e74656e"
    "744f66667365740046696c6573000000000000000000000000000000000000000000000000000000"
    "00000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "00000000000000000000000000000000000000000000000000000000000000000000000000000000"
    "00000000000000000000000000000000544f43200000000000000000000000004055544600000083"
    "0000003100000045000000830000000000050014000000015a000000005a00000008540000001154"
    "0000001c540000002500000031000000360000009b00000009000000094469724e616d650046696c"
    "654e616d650046696c654f66667365740046696c6553697a65004578747261637453697a65004241"
    "5345004f4e452e4753430068656c6c6f2d677363";

const si::GfdCacheError* as_error(const auto& result) {
    return std::get_if<si::GfdCacheError>(&result);
}

}  // namespace

ARMIGO_SI_TEST(gfd, crilayla_round_trip) {
    std::vector<std::byte> stored(0x10 + 6 + 0x100, std::byte{0x5a});
    std::memcpy(stored.data(), "CRILAYLA", 8);
    put_le32_at(stored, 8, 6);
    put_le32_at(stored, 12, 6);
    const std::uint8_t compressed[] = {0x00, 0x00, 0x50, 0xd5, 0x2e, 0x66};
    std::memcpy(stored.data() + 0x10, compressed, sizeof compressed);
    auto decoded = si::decompress_crilayla(stored);
    ARMIGO_SI_CHECK(as_error(decoded) == nullptr);
    if (const auto* bytes = std::get_if<std::vector<std::byte>>(&decoded)) {
        ARMIGO_SI_CHECK_EQ(bytes->size(), std::size_t{0x106});
        ARMIGO_SI_CHECK((*bytes)[0] == std::byte{0x5a} && (*bytes)[0xff] == std::byte{0x5a});
        const std::uint8_t payload[] = {0xaa, 0xbb, 0xcc, 0xaa, 0xbb, 0xcc};
        ARMIGO_SI_CHECK(std::memcmp(bytes->data() + 0x100, payload, sizeof payload) == 0);
    }
    si::CpkMember member;
    member.name = "ONE.GSC";
    member.offset = 0;
    member.stored = stored.size();
    member.extracted = 0x106;
    TempFile file(std::vector<std::uint8_t>(reinterpret_cast<const std::uint8_t*>(stored.data()),
                                            reinterpret_cast<const std::uint8_t*>(stored.data()) + stored.size()));
    auto read = si::read_cpk_member(file.path, member);
    ARMIGO_SI_CHECK(as_error(read) == nullptr);
    if (const auto* bytes = std::get_if<std::vector<std::byte>>(&read)) {
        ARMIGO_SI_CHECK(*bytes == std::get<std::vector<std::byte>>(decoded));
    }
    member.extracted = 1;
    ARMIGO_SI_CHECK(as_error(si::read_cpk_member(file.path, member)) != nullptr);
    stored[8] = std::byte{0xff};
    stored[9] = std::byte{0xff};
    stored[10] = std::byte{0xff};
    stored[11] = std::byte{0x7f};
    ARMIGO_SI_CHECK(as_error(si::decompress_crilayla(stored)) != nullptr);
    ARMIGO_SI_CHECK(as_error(si::decompress_crilayla({})) != nullptr);
}

ARMIGO_SI_TEST(gfd, cpk_listing_and_masked_header) {
    const auto plain = decode_hex(kCpkHex);
    ARMIGO_SI_CHECK_EQ(plain.size(), std::size_t{420});
    TempFile file(plain);
    auto listing = si::list_cpk(file.path);
    ARMIGO_SI_CHECK(as_error(listing) == nullptr);
    if (const auto* found = std::get_if<si::CpkListing>(&listing)) {
        ARMIGO_SI_CHECK_EQ(found->members.size(), std::size_t{1});
        if (!found->members.empty()) {
            const auto& member = found->members.front();
            ARMIGO_SI_CHECK(member.name == "BASE/ONE.GSC");
            ARMIGO_SI_CHECK_EQ(member.stored, std::uint64_t{9});
            ARMIGO_SI_CHECK_EQ(member.extracted, std::uint64_t{9});
            auto payload = si::read_cpk_member(file.path, member);
            ARMIGO_SI_CHECK(as_error(payload) == nullptr);
            if (const auto* bytes = std::get_if<std::vector<std::byte>>(&payload)) {
                ARMIGO_SI_CHECK(std::memcmp(bytes->data(), "hello-gsc", 9) == 0);
            }
        }
    }
    auto masked = plain;
    std::uint32_t mask = 0x655f;
    for (std::size_t index = 0x10; index < 0x10 + 0x76; ++index) {
        masked[index] = static_cast<std::uint8_t>(masked[index] ^ (mask & 0xff));
        mask = (mask * 0x4115u) & 0xffffu;
    }
    ARMIGO_SI_CHECK(std::memcmp(masked.data() + 0x10, "@UTF", 4) != 0);
    TempFile masked_file(masked);
    auto masked_listing = si::list_cpk(masked_file.path);
    ARMIGO_SI_CHECK(as_error(masked_listing) == nullptr);
    if (const auto* found = std::get_if<si::CpkListing>(&masked_listing)) {
        ARMIGO_SI_CHECK(found->members.size() == 1 && found->members.front().name == "BASE/ONE.GSC");
    }
    TempFile empty({});
    ARMIGO_SI_CHECK(as_error(si::list_cpk(empty.path)) != nullptr);
}

namespace {

void store_be16(std::vector<std::uint8_t>& bytes, std::size_t at, std::uint16_t value) {
    bytes[at] = static_cast<std::uint8_t>(value >> 8);
    bytes[at + 1] = static_cast<std::uint8_t>(value);
}

void store_be32(std::vector<std::uint8_t>& bytes, std::size_t at, std::uint32_t value) {
    bytes[at] = static_cast<std::uint8_t>(value >> 24);
    bytes[at + 1] = static_cast<std::uint8_t>(value >> 16);
    bytes[at + 2] = static_cast<std::uint8_t>(value >> 8);
    bytes[at + 3] = static_cast<std::uint8_t>(value);
}

void store_be64(std::vector<std::uint8_t>& bytes, std::size_t at, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) {
        bytes[at + index] = static_cast<std::uint8_t>(value >> (8u * (7u - index)));
    }
}

std::vector<std::uint8_t> wrap_utf(std::string_view magic, const std::vector<std::uint8_t>& table) {
    std::vector<std::uint8_t> packet(0x18 + table.size(), 0);
    std::memcpy(packet.data(), magic.data(), magic.size());
    const auto packet_size = static_cast<std::uint32_t>(packet.size() - 0x10);
    for (unsigned index = 0; index < 4; ++index) {
        packet[8 + index] = static_cast<std::uint8_t>(packet_size >> (8u * index));
    }
    packet[0x10] = '@';
    packet[0x11] = 'U';
    packet[0x12] = 'T';
    packet[0x13] = 'F';
    store_be32(packet, 0x14, static_cast<std::uint32_t>(table.size()));
    std::memcpy(packet.data() + 0x18, table.data(), table.size());
    return packet;
}

// 1024 zero-storage columns and a row length of 0. The row count is the
// number of value rows the table asks the reader to build.
std::vector<std::uint8_t> wide_utf_packet(std::string_view magic, std::uint32_t rows) {
    constexpr std::uint16_t kColumns = 1024;
    const std::size_t rows_off = 24 + static_cast<std::size_t>(kColumns) * 5;
    std::vector<std::uint8_t> table(rows_off + 1, 0);
    store_be16(table, 2, static_cast<std::uint16_t>(rows_off));
    store_be32(table, 4, static_cast<std::uint32_t>(rows_off));
    store_be32(table, 8, static_cast<std::uint32_t>(table.size()));
    store_be16(table, 16, kColumns);
    store_be16(table, 18, 0);
    store_be32(table, 20, rows);
    for (std::size_t index = 0; index < kColumns; ++index) {
        table[24 + index * 5] = 0x10;
    }
    return wrap_utf(magic, table);
}

// One constant row naming the three offsets a CPK header must carry.
std::vector<std::uint8_t> offset_header(std::uint64_t toc_offset, std::uint64_t toc_size,
                                       std::uint64_t content_offset) {
    const char* names[] = {"TocOffset", "TocSize", "ContentOffset"};
    const std::uint64_t values[] = {toc_offset, toc_size, content_offset};
    std::string strings;
    std::uint32_t name_at[3] = {};
    for (int index = 0; index < 3; ++index) {
        name_at[index] = static_cast<std::uint32_t>(strings.size());
        strings.append(names[index]);
        strings.push_back('\0');
    }
    constexpr std::size_t kColumnBytes = 13;
    const std::size_t rows_off = 24 + 3 * kColumnBytes;
    std::vector<std::uint8_t> table(rows_off + strings.size(), 0);
    store_be16(table, 2, static_cast<std::uint16_t>(rows_off));
    store_be32(table, 4, static_cast<std::uint32_t>(rows_off));
    store_be32(table, 8, static_cast<std::uint32_t>(table.size()));
    store_be16(table, 16, 3);
    store_be16(table, 18, 0);
    store_be32(table, 20, 1);
    for (int index = 0; index < 3; ++index) {
        const std::size_t at = 24 + static_cast<std::size_t>(index) * kColumnBytes;
        table[at] = 0x36;
        store_be32(table, at + 1, name_at[index]);
        store_be64(table, at + 5, values[index]);
    }
    std::memcpy(table.data() + rows_off, strings.data(), strings.size());
    return wrap_utf("CPK ", table);
}

}  // namespace

ARMIGO_SI_TEST(gfd, utf_allocation_budget) {
    const auto header = wide_utf_packet("CPK ", 1'000'000);
    ARMIGO_SI_CHECK(header.size() < 8 * 1024);
    TempFile header_file(header);
    const auto header_result = si::list_cpk(header_file.path);
    const auto* header_error = as_error(header_result);
    ARMIGO_SI_CHECK(header_error != nullptr &&
                    header_error->message.find("does not have one row") != std::string::npos);

    // Past the one-row header, the same column width still has to fit the
    // cell budget before any row storage is built.
    const auto toc = wide_utf_packet("TOC ", 2000);
    constexpr std::uint64_t kTocOffset = 0x200;
    const auto head = offset_header(kTocOffset, toc.size() - 0x10, kTocOffset);
    ARMIGO_SI_CHECK(head.size() <= static_cast<std::size_t>(kTocOffset));
    std::vector<std::uint8_t> archive(static_cast<std::size_t>(kTocOffset) + toc.size(), 0);
    std::memcpy(archive.data(), head.data(), head.size());
    std::memcpy(archive.data() + static_cast<std::size_t>(kTocOffset), toc.data(), toc.size());
    TempFile toc_file(archive);
    const auto toc_result = si::list_cpk(toc_file.path);
    const auto* toc_error = as_error(toc_result);
    ARMIGO_SI_CHECK(toc_error != nullptr &&
                    toc_error->message.find("allocation exceeds the resource limit") !=
                        std::string::npos);
}

ARMIGO_SI_TEST(gfd, gsc_package_round_trip) {
    const auto vertex = make_stage(0, __FILE__, __LINE__);
    const auto package = needle_package(descriptor_for(vertex), 0);
    auto parsed = si::parse_gsc(as_bytes(gsc_file(0, package)), "cache.gsc");
    ARMIGO_SI_CHECK(as_error(parsed) == nullptr);
    if (const auto* stages = std::get_if<std::vector<si::OwnedShaderStage>>(&parsed)) {
        ARMIGO_SI_CHECK_EQ(stages->size(), std::size_t{1});
        if (!stages->empty()) {
            ARMIGO_SI_CHECK((*stages)[0].stage == si::Stage::Vertex);
            ARMIGO_SI_CHECK((*stages)[0].control.size() == vertex.control.size());
            ARMIGO_SI_CHECK(std::memcmp((*stages)[0].control.data(), vertex.control.data(), vertex.control.size()) == 0);
            ARMIGO_SI_CHECK(std::memcmp((*stages)[0].code.data(), vertex.code.data(), vertex.code.size()) == 0);
        }
    }
    const auto second = needle_package(descriptor_for(vertex), 2);
    auto version_two = si::parse_gsc(as_bytes(gsc_file(0, second)), "cache.gsc");
    ARMIGO_SI_CHECK(as_error(version_two) == nullptr);
    if (const auto* stages = std::get_if<std::vector<si::OwnedShaderStage>>(&version_two)) {
        ARMIGO_SI_CHECK_EQ(stages->size(), std::size_t{1});
    }
    const auto fragment = make_stage(1, __FILE__, __LINE__);
    auto mismatch = si::parse_gsc(as_bytes(gsc_file(0, needle_package(descriptor_for(fragment), 0))), "cache.gsc");
    const auto* mismatch_error = as_error(mismatch);
    ARMIGO_SI_CHECK(mismatch_error != nullptr &&
                    mismatch_error->message.find("does not match") != std::string::npos);
    auto file = gsc_file(0, package);
    file.pop_back();
    ARMIGO_SI_CHECK(as_error(si::parse_gsc(as_bytes(file), "cache.gsc")) != nullptr);
    auto skipped = gsc_file(2, {});
    auto skipped_parsed = si::parse_gsc(as_bytes(skipped), "cache.gsc");
    ARMIGO_SI_CHECK(as_error(skipped_parsed) == nullptr);
    if (const auto* stages = std::get_if<std::vector<si::OwnedShaderStage>>(&skipped_parsed)) {
        ARMIGO_SI_CHECK(stages->empty());
    }
    auto broken = as_bytes(gsc_file(0, package));
    broken[0x14] = std::byte{0};
    ARMIGO_SI_CHECK(as_error(si::parse_gsc(broken, "cache.gsc")) != nullptr);
}

ARMIGO_SI_TEST(gfd, needle_scan_ignores_only_failures) {
    const auto vertex = make_stage(0, __FILE__, __LINE__);
    const auto package = as_bytes(needle_package(descriptor_for(vertex), 0));
    const std::uint64_t ignored_success[] = {0};
    auto kept = si::scan_hhneedle(package, "main", ignored_success);
    ARMIGO_SI_CHECK(kept.rejected.empty());
    ARMIGO_SI_CHECK(kept.uncovered_controls.empty());
    ARMIGO_SI_CHECK_EQ(kept.stages.size(), std::size_t{1});

    std::vector<std::byte> image(96, std::byte{0});
    auto plant = [&](std::size_t offset) {
        std::memcpy(image.data() + offset, "HHNEEDLE", 8);
        image[offset + 11] = std::byte{1};
    };
    plant(0);
    plant(40);
    image[80] = std::byte{0x34};
    image[81] = std::byte{0x12};
    image[82] = std::byte{0x76};
    image[83] = std::byte{0x98};
    auto reported = si::scan_hhneedle(image, "main", {});
    ARMIGO_SI_CHECK_EQ(reported.rejected.size(), std::size_t{2});
    ARMIGO_SI_CHECK_EQ(reported.uncovered_controls.size(), std::size_t{1});
    ARMIGO_SI_CHECK(!reported.uncovered_controls.empty() && reported.uncovered_controls.front() == 80);
    const std::uint64_t ignored[] = {0};
    auto filtered = si::scan_hhneedle(image, "main", ignored);
    ARMIGO_SI_CHECK_EQ(filtered.rejected.size(), std::size_t{1});
    ARMIGO_SI_CHECK(!filtered.rejected.empty() && filtered.rejected.front().find("@40") != std::string::npos);
    ARMIGO_SI_CHECK_EQ(filtered.uncovered_controls.size(), std::size_t{1});
}

ARMIGO_SI_TEST(gfd, missing_shader_caches) {
    const std::vector<std::string> present{"BASE/INIT.GSC", "BASE/ALT.GSC"};
    const std::vector<std::string> required{"BASE/INIT.GSC", "BASE/ONE.GSC"};
    const std::vector<std::pair<std::string, std::vector<std::string>>> fallbacks{
        {"BASE/ONE.GSC", {"BASE/ALT.GSC", "BASE/OTHER.GSC"}}};
    ARMIGO_SI_CHECK(si::missing_shader_caches(present, required, fallbacks).empty());
    ARMIGO_SI_CHECK(si::missing_shader_caches({"BASE/INIT.GSC"}, required, fallbacks) ==
                    std::vector<std::string>{"BASE/ONE.GSC"});
    ARMIGO_SI_CHECK(si::missing_shader_caches(present, {}, {}).empty());
}
