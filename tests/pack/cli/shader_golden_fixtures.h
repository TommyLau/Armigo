// ============================================================================
// Armigo Tests - Shader Reader Characterization Fixtures
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic shader containers, a CPK cache, an NSO image and a .nup archive.
// Every byte is built here. RomFS members of the archive are an arbitrary
// list of path and payload pairs.
#pragma once

#include "sha256.h"
#include "synthetic.h"

#include <armigo/loader/nso_module.h>
#include <armigo/shader_input/glslc_output.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace armigo::pack::shader_golden {

namespace si = armigo::shader_input;
namespace gf = armigo::shader_input::glslc_format;
using armigo::shader_input::test::build_code;
using armigo::shader_input::test::build_control;
using armigo::shader_input::test::CodeSpec;
using armigo::shader_input::test::control_for;

inline void put_le(std::vector<std::uint8_t>& bytes, std::size_t at, std::uint64_t value, unsigned width = 4) {
    for (unsigned index = 0; index < width; ++index) {
        bytes[at + index] = static_cast<std::uint8_t>(value >> (8u * index));
    }
}

inline void put_be(std::vector<std::uint8_t>& bytes, std::size_t at, std::uint64_t value, unsigned width = 4) {
    for (unsigned index = 0; index < width; ++index) {
        bytes[at + index] = static_cast<std::uint8_t>(value >> (8u * (width - 1u - index)));
    }
}

inline void put_le(std::vector<std::byte>& bytes, std::size_t at, std::uint64_t value, unsigned width = 4) {
    for (unsigned index = 0; index < width; ++index) {
        bytes[at + index] = static_cast<std::byte>(static_cast<unsigned>(value >> (8u * index)) & 0xffu);
    }
}

inline std::vector<std::byte> as_bytes(const std::vector<std::uint8_t>& bytes) {
    std::vector<std::byte> out(bytes.size());
    if (!bytes.empty()) std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

inline unsigned slot_of(si::Stage stage) {
    switch (stage) {
        case si::Stage::Vertex: return 0;
        case si::Stage::TessellationControl: return 1;
        case si::Stage::TessellationEvaluation: return 2;
        case si::Stage::Geometry: return 3;
        case si::Stage::Fragment: return 4;
        case si::Stage::Compute: return 5;
    }
    throw std::runtime_error("Synthetic stage has no container slot");
}

// One variation. Each requested stage owns a descriptor, a control and a code
// region. The slot matches the stage the container reader expects.
inline std::vector<std::uint8_t> bnsh_with(std::initializer_list<si::Stage> stages) {
    struct Piece {
        unsigned slot = 0;
        std::vector<std::uint8_t> control;
        std::vector<std::uint8_t> code;
    };
    std::vector<Piece> pieces;
    for (const si::Stage stage : stages) {
        CodeSpec spec;
        spec.stage = static_cast<std::uint32_t>(stage);
        pieces.push_back(Piece{slot_of(stage), build_control(control_for(spec)), build_code(spec)});
    }
    constexpr std::uint32_t kTable = 0x200;
    constexpr std::uint32_t kProgram = 0x300;
    constexpr std::uint32_t kDescriptors = 0x400;
    std::uint32_t payload = 0x1000;
    std::uint32_t end = 0x1000;
    struct Placed {
        std::uint32_t descriptor = 0;
        std::uint32_t control = 0;
        std::uint32_t code = 0;
    };
    std::vector<Placed> placed;
    placed.reserve(pieces.size());
    for (const Piece& piece : pieces) {
        Placed at;
        at.descriptor = kDescriptors + static_cast<std::uint32_t>(placed.size()) * 0x20u;
        at.control = payload;
        at.code = payload + 0x800u;
        payload += 0x1000u;
        end = std::max(end, at.code + static_cast<std::uint32_t>(piece.code.size()));
        placed.push_back(at);
    }
    std::vector<std::uint8_t> bytes(end, 0);
    std::memcpy(bytes.data(), "BNSH\0\0\0\0", 8);
    std::memcpy(bytes.data() + 0x60, "grsc", 4);
    put_le(bytes, 0x7c, 1);
    put_le(bytes, 0x80, kTable, 8);
    put_le(bytes, kTable + 0x10, kProgram, 8);
    for (std::size_t index = 0; index < pieces.size(); ++index) {
        const Piece& piece = pieces[index];
        const Placed& at = placed[index];
        put_le(bytes, kProgram + 8u + piece.slot * 8u, at.descriptor, 8);
        put_le(bytes, at.descriptor + 8, at.control, 8);
        put_le(bytes, at.descriptor + 16, at.code, 8);
        put_le(bytes, at.descriptor + 24, static_cast<std::uint32_t>(piece.code.size()));
        put_le(bytes, at.descriptor + 28, static_cast<std::uint32_t>(piece.control.size()));
        std::memcpy(bytes.data() + at.control, piece.control.data(), piece.control.size());
        std::memcpy(bytes.data() + at.code, piece.code.data(), piece.code.size());
    }
    return bytes;
}

// Signature and resource block are intact. The variation table lies outside
// the buffer, which the container reader reports as a range error.
inline std::vector<std::uint8_t> bnsh_structural() {
    std::vector<std::uint8_t> bytes(0x100, 0);
    std::memcpy(bytes.data(), "BNSH\0\0\0\0", 8);
    std::memcpy(bytes.data() + 0x60, "grsc", 4);
    put_le(bytes, 0x7c, 1);
    put_le(bytes, 0x80, 0x10000u, 8);
    return bytes;
}

inline void place(std::vector<std::uint8_t>& bytes, std::size_t at, const std::vector<std::uint8_t>& payload) {
    std::memcpy(bytes.data() + at, payload.data(), payload.size());
}

inline void shader_at(std::vector<std::uint8_t>& bytes, std::uint32_t control_at, std::uint32_t code_at,
                      si::Stage stage) {
    CodeSpec spec;
    spec.stage = static_cast<std::uint32_t>(stage);
    place(bytes, control_at, build_control(control_for(spec)));
    place(bytes, code_at, build_code(spec));
}

inline void bahs_entry(std::vector<std::uint8_t>& bytes, std::uint32_t at, std::uint32_t control,
                       std::uint32_t code_offset) {
    put_le(bytes, at, code_offset);
    put_le(bytes, at + 4, 0);
    put_le(bytes, at + 8, si::format::control_min_size);
    put_le(bytes, at + 12, control - at);
}

// Vertex and fragment stages. The third entry repeats the vertex pair so the
// reader can drop it as a duplicate candidate.
inline std::vector<std::uint8_t> bahs_container() {
    std::vector<std::uint8_t> bytes(0x2600, 0);
    std::memcpy(bytes.data(), "BAHS", 4);
    put_le(bytes, 0x18, 0x400);
    put_le(bytes, 0x1c, 0x2200);
    shader_at(bytes, 0x200, 0x2200, si::Stage::Vertex);
    shader_at(bytes, 0xb00, 0x2300, si::Stage::Fragment);
    bahs_entry(bytes, 0x40, 0x200, 0);
    bahs_entry(bytes, 0x1c00, 0xb00, 0x100);
    bahs_entry(bytes, 0x1c10, 0x200, 0);
    return bytes;
}

// One named member. The name offset in the node is a count of 4-byte units.
inline std::vector<std::uint8_t> sarc_member(const std::vector<std::uint8_t>& payload, std::string_view name) {
    std::string stored(name);
    stored.push_back('\0');
    while (stored.size() % 4u != 0) stored.push_back('\0');
    constexpr std::size_t kNode = 0x20;
    constexpr std::size_t kNames = 0x38;
    const std::size_t data = kNames + stored.size();
    std::vector<std::uint8_t> bytes(data + payload.size(), 0);
    std::memcpy(bytes.data(), "SARC", 4);
    put_le(bytes, 4, 0x14, 2);
    put_le(bytes, 6, 0xfeff, 2);
    put_le(bytes, 8, static_cast<std::uint32_t>(bytes.size()));
    put_le(bytes, 0x0c, static_cast<std::uint32_t>(data));
    std::memcpy(bytes.data() + 0x14, "SFAT", 4);
    put_le(bytes, 0x18, 12, 2);
    put_le(bytes, 0x1a, 1, 2);
    put_le(bytes, kNode + 4, 0x01000000u);
    put_le(bytes, kNode + 8, 0);
    put_le(bytes, kNode + 12, static_cast<std::uint32_t>(payload.size()));
    std::memcpy(bytes.data() + 0x30, "SFNT", 4);
    put_le(bytes, 0x34, 8, 2);
    std::memcpy(bytes.data() + kNames, stored.data(), stored.size());
    if (!payload.empty()) std::memcpy(bytes.data() + data, payload.data(), payload.size());
    return bytes;
}

// Stored Yaz0: every body bit is a literal. Groups of eight use a flag of
// 0xff; a short tail sets only the high bits it consumes.
inline std::vector<std::uint8_t> yaz0_literals(const std::vector<std::uint8_t>& raw) {
    std::vector<std::uint8_t> bytes(16, 0);
    std::memcpy(bytes.data(), "Yaz0", 4);
    put_be(bytes, 4, static_cast<std::uint32_t>(raw.size()));
    for (std::size_t cursor = 0; cursor < raw.size();) {
        const std::size_t count = std::min<std::size_t>(8, raw.size() - cursor);
        std::uint8_t flags = 0;
        for (std::size_t bit = 0; bit < count; ++bit) flags = static_cast<std::uint8_t>(flags | (1u << (7u - bit)));
        bytes.push_back(flags);
        bytes.insert(bytes.end(), raw.begin() + static_cast<std::ptrdiff_t>(cursor),
                     raw.begin() + static_cast<std::ptrdiff_t>(cursor + count));
        cursor += count;
    }
    return bytes;
}

inline std::vector<std::uint8_t> vertex_fragment_bnsh() {
    return bnsh_with({si::Stage::Vertex, si::Stage::Fragment});
}

inline std::vector<std::uint8_t> geometry_bnsh() { return bnsh_with({si::Stage::Geometry}); }

struct UtfColumn {
    std::string name;
    bool text = false;
    std::uint64_t number = 0;
    std::string value;
};

inline std::vector<std::uint8_t> utf_packet(std::string_view magic, const std::vector<UtfColumn>& columns) {
    std::string pool;
    std::vector<std::uint32_t> name_at;
    std::vector<std::uint32_t> value_at;
    name_at.reserve(columns.size());
    for (const UtfColumn& column : columns) {
        name_at.push_back(static_cast<std::uint32_t>(pool.size()));
        pool.append(column.name);
        pool.push_back('\0');
    }
    value_at.reserve(columns.size());
    for (const UtfColumn& column : columns) {
        value_at.push_back(static_cast<std::uint32_t>(pool.size()));
        if (column.text) {
            pool.append(column.value);
            pool.push_back('\0');
        }
    }
    std::size_t cursor = 24;
    for (const UtfColumn& column : columns) cursor += 5u + (column.text ? 4u : 8u);
    const auto rows_off = static_cast<std::uint32_t>(cursor);
    std::vector<std::uint8_t> table(rows_off + pool.size(), 0);
    put_be(table, 2, rows_off, 2);
    put_be(table, 4, rows_off);
    put_be(table, 8, static_cast<std::uint32_t>(table.size()));
    put_be(table, 16, static_cast<std::uint32_t>(columns.size()), 2);
    put_be(table, 18, 0, 2);
    put_be(table, 20, 1);
    std::size_t at = 24;
    for (std::size_t index = 0; index < columns.size(); ++index) {
        table[at] = columns[index].text ? 0x3a : 0x36;
        put_be(table, at + 1, name_at[index]);
        at += 5;
        if (columns[index].text) {
            put_be(table, at, value_at[index]);
            at += 4;
        } else {
            put_be(table, at, columns[index].number, 8);
            at += 8;
        }
    }
    std::memcpy(table.data() + rows_off, pool.data(), pool.size());
    std::vector<std::uint8_t> packet(0x18 + table.size(), 0);
    std::memcpy(packet.data(), magic.data(), magic.size());
    put_le(packet, 8, static_cast<std::uint32_t>(packet.size() - 0x10));
    packet[0x10] = '@';
    packet[0x11] = 'U';
    packet[0x12] = 'T';
    packet[0x13] = 'F';
    put_be(packet, 0x14, static_cast<std::uint32_t>(table.size()));
    std::memcpy(packet.data() + 0x18, table.data(), table.size());
    return packet;
}

inline std::size_t align16(std::size_t value) { return (value + 15u) & ~std::size_t{15}; }

// One stored member, BASE/ONE.GSC, at a content offset past the header.
inline std::vector<std::uint8_t> cpk_with_member(const std::vector<std::uint8_t>& payload) {
    const std::vector<std::uint8_t> toc = utf_packet("TOC ", {
        {"DirName", true, 0, "BASE"},
        {"FileName", true, 0, "ONE.GSC"},
        {"FileOffset", false, 0, {}},
        {"FileSize", false, payload.size(), {}},
        {"ExtractSize", false, payload.size(), {}},
    });
    const std::vector<std::uint8_t> probe = utf_packet("CPK ", {
        {"TocOffset", false, 0, {}},
        {"TocSize", false, 0, {}},
        {"ContentOffset", false, 0, {}},
    });
    const std::size_t content = align16(probe.size());
    const std::size_t toc_at = align16(content + payload.size());
    const std::vector<std::uint8_t> header = utf_packet("CPK ", {
        {"TocOffset", false, toc_at, {}},
        {"TocSize", false, toc.size() - 0x10, {}},
        {"ContentOffset", false, content, {}},
    });
    if (header.size() > content) throw std::runtime_error("Synthetic CPK header does not fit before its content");
    std::vector<std::uint8_t> archive(toc_at + toc.size(), 0);
    std::memcpy(archive.data(), header.data(), header.size());
    if (!payload.empty()) std::memcpy(archive.data() + content, payload.data(), payload.size());
    std::memcpy(archive.data() + toc_at, toc.data(), toc.size());
    return archive;
}

inline std::vector<std::uint8_t> stage_descriptor(si::Stage stage) {
    CodeSpec spec;
    spec.stage = static_cast<std::uint32_t>(stage);
    const std::vector<std::uint8_t> code = build_code(spec);
    const std::vector<std::uint8_t> control = build_control(control_for(spec));
    constexpr std::uint32_t kControl = 0x100;
    const auto code_at = kControl + static_cast<std::uint32_t>(control.size());
    std::vector<std::uint8_t> descriptor(code_at + code.size(), 0);
    put_le(descriptor, 0x30, kControl);
    put_le(descriptor, 0x34, code_at);
    put_le(descriptor, 0x38, static_cast<std::uint32_t>(code.size()));
    put_le(descriptor, 0x3c, static_cast<std::uint32_t>(control.size()));
    std::memcpy(descriptor.data() + kControl, control.data(), control.size());
    std::memcpy(descriptor.data() + code_at, code.data(), code.size());
    return descriptor;
}

inline std::vector<std::uint8_t> needle_package(const std::vector<std::uint8_t>& descriptor) {
    const auto descriptor_size = static_cast<std::uint32_t>(descriptor.size());
    const std::uint32_t total = 44u + descriptor_size;
    std::vector<std::uint8_t> out;
    auto text = [&](std::string_view value) { out.insert(out.end(), value.begin(), value.end()); };
    auto be32 = [&](std::uint32_t value) {
        out.push_back(static_cast<std::uint8_t>(value >> 24));
        out.push_back(static_cast<std::uint8_t>(value >> 16));
        out.push_back(static_cast<std::uint8_t>(value >> 8));
        out.push_back(static_cast<std::uint8_t>(value));
    };
    text("HHNEEDLE");
    be32(total - 8);
    text(std::string_view("n\0\0\0", 4));
    text("HNSHV001");
    be32(total - 24);
    text(std::string_view("n\0\0\0", 4));
    be32(0);
    be32(8u + descriptor_size);
    be32(descriptor_size);
    out.insert(out.end(), descriptor.begin(), descriptor.end());
    return out;
}

// A GFS0 cache with one vertex program record.
inline std::vector<std::uint8_t> gsc_vertex() {
    const std::vector<std::uint8_t> payload = needle_package(stage_descriptor(si::Stage::Vertex));
    std::vector<std::uint8_t> out(0x20, 0);
    out[0] = 'G';
    out[1] = 'F';
    out[2] = 'S';
    out[3] = '0';
    put_be(out, 0x10, 0x47534330u);
    put_be(out, 0x14, 0x10000000u);
    put_be(out, 0x18, 8u);
    out.insert(out.end(), 4, 0);
    out.push_back(0);
    out.push_back(0);
    out.push_back(static_cast<std::uint8_t>(payload.size() >> 24));
    out.push_back(static_cast<std::uint8_t>(payload.size() >> 16));
    out.push_back(static_cast<std::uint8_t>(payload.size() >> 8));
    out.push_back(static_cast<std::uint8_t>(payload.size()));
    out.push_back(0);
    out.push_back(0);
    out.insert(out.end(), 24, 0);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

inline std::vector<std::uint8_t> gsc_cpk() { return cpk_with_member(gsc_vertex()); }

inline void put_bytes(std::vector<std::byte>& bytes, std::size_t at, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        bytes[at + index] = static_cast<std::byte>(value >> (8u * index));
    }
}

// Uncompressed NSO. Text holds `text`. Rodata and data are empty and sit on
// the following guest pages. Hash flags stay clear.
inline std::vector<std::byte> nso_with_text(std::vector<std::byte> text) {
    if (text.empty()) text.assign(0x10, std::byte{0});
    const auto page = static_cast<std::uint32_t>(armigo::loader::kNsoGuestPageSize);
    const auto text_size = static_cast<std::uint32_t>(text.size());
    const std::uint32_t rodata_at = (text_size + page - 1u) & ~(page - 1u);
    const std::uint32_t data_at = rodata_at + page;
    std::vector<std::byte> file(0x100 + text.size(), std::byte{0});
    file[0] = std::byte{'N'};
    file[1] = std::byte{'S'};
    file[2] = std::byte{'O'};
    file[3] = std::byte{'0'};
    put_bytes(file, 0x10, 0x100);
    put_bytes(file, 0x14, 0);
    put_bytes(file, 0x18, text_size);
    put_bytes(file, 0x20, 0);
    put_bytes(file, 0x24, rodata_at);
    put_bytes(file, 0x30, 0);
    put_bytes(file, 0x34, data_at);
    put_bytes(file, 0x60, text_size);
    std::memcpy(file.data() + 0x100, text.data(), text.size());
    return file;
}

inline std::vector<std::byte> glslc_section(std::uint32_t stage) {
    CodeSpec spec;
    spec.stage = stage;
    const std::vector<std::uint8_t> control = build_control(control_for(spec));
    const std::vector<std::uint8_t> code = build_code(spec);
    const std::size_t sections = 1;
    std::size_t size = gf::first_section + gf::section_stride * sections;
    size += control.size() + code.size() + 16;
    std::vector<std::byte> out(size, std::byte{0});
    put_bytes(out, 0, gf::magic);
    put_bytes(out, gf::size_field, static_cast<std::uint32_t>(size));
    put_bytes(out, gf::section_count_field, 1);
    const std::size_t data = gf::first_section + gf::section_stride;
    const auto length = static_cast<std::uint32_t>(control.size() + code.size() + 16);
    put_bytes(out, gf::first_section + gf::section_size_field, length);
    put_bytes(out, gf::first_section + gf::section_data_field, static_cast<std::uint32_t>(data));
    put_bytes(out, gf::first_section + gf::section_type_field, gf::gpu_code_section);
    put_bytes(out, gf::first_section + gf::control_offset_field, 16);
    put_bytes(out, gf::first_section + gf::code_offset_field, static_cast<std::uint32_t>(16 + control.size()));
    put_bytes(out, gf::first_section + gf::code_size_field, static_cast<std::uint32_t>(code.size()));
    put_bytes(out, gf::first_section + gf::control_size_field, static_cast<std::uint32_t>(control.size()));
    std::memcpy(out.data() + data + 16, control.data(), control.size());
    std::memcpy(out.data() + data + 16 + control.size(), code.data(), code.size());
    return out;
}

inline std::vector<std::byte> nso_with_glslc_vertex() { return nso_with_text(glslc_section(0)); }

inline std::vector<std::byte> nso_without_shaders() { return nso_with_text({}); }

inline void word(std::vector<std::byte>& bytes, std::size_t at, std::uint64_t value, unsigned width = 4) {
    for (unsigned index = 0; index < width; ++index) {
        bytes[at + index] = std::byte(static_cast<unsigned>(value >> (8u * index)) & 0xffu);
    }
}

inline void text_at(std::vector<std::byte>& bytes, std::size_t at, const char* literal, std::size_t count) {
    std::memcpy(bytes.data() + at, literal, count);
}

// The container layout follows the native reader fixture. The program end and
// constant offset are set so the vertex slot has an empty constant block on
// a 16-byte instruction boundary. The fragment slot shares that control and
// therefore disagrees with the vertex selector.
inline void native_control(std::vector<std::byte>& bytes, std::size_t at, bool filled) {
    word(bytes, at, 0x98761234u);
    word(bytes, at + 4, 0xc9);
    word(bytes, at + 8, 20);
    word(bytes, at + 0x30, filled ? 0xe0u : 0u);
    word(bytes, at + 0x38, 0x90);
    word(bytes, at + 0x40, 0x90);
    word(bytes, at + 0x44, 0x100);
    word(bytes, at + 0xd4, 0xe0);
}

inline std::vector<std::byte> native_bnsh() {
    std::vector<std::byte> bytes(0x600, std::byte{0});
    text_at(bytes, 0, "BNSH\0\0\0\0", 8);
    word(bytes, 0x1c, bytes.size());
    text_at(bytes, 0x60, "grsc", 4);
    word(bytes, 0x7c, 3);
    word(bytes, 0x80, 0x90, 8);
    word(bytes, 0xa0, 0x180, 8);
    word(bytes, 0xe0, 0x180, 8);
    word(bytes, 0x188, 0x200, 8);
    word(bytes, 0x1a8, 0x200, 8);
    word(bytes, 0x208, 0x280, 8);
    word(bytes, 0x210, 0x400, 8);
    word(bytes, 0x218, 0x100);
    word(bytes, 0x21c, 0xe0);
    native_control(bytes, 0x280, true);
    return bytes;
}

inline std::vector<std::byte> native_bahs() {
    std::vector<std::byte> bytes(0x500, std::byte{0});
    text_at(bytes, 0, "BAHS", 4);
    word(bytes, 4, 11);
    word(bytes, 8, bytes.size());
    word(bytes, 0x18, 0x100);
    word(bytes, 0x1c, 0x400);
    native_control(bytes, 0x100, false);
    for (const std::uint32_t at : {0x40u, 0x300u, 0x3f0u}) {
        word(bytes, at + 8, 0xe0);
        word(bytes, at + 12, static_cast<std::uint32_t>(0x100u - at));
    }
    return bytes;
}

struct RomMember {
    std::string path;
    std::vector<std::byte> bytes;
};

struct DirectoryNode {
    std::string name;
    std::map<std::string, std::size_t> directories;
    std::map<std::string, std::size_t> files;
};

inline std::array<std::byte, 32> digest_bytes(const std::vector<std::byte>& bytes) {
    const auto digest = armigo::loader::detail::sha256(bytes);
    if (!digest) throw std::runtime_error("Synthetic archive hash failed");
    return *digest;
}

// One base package. `members` are RomFS paths such as "Shader/a.bin". An empty
// ExeFS directory is always present because selection requires one.
inline std::vector<std::byte> nup_archive(const std::vector<RomMember>& members) {
    std::vector<DirectoryNode> directories(3);
    directories[1].name = "exefs";
    directories[2].name = "romfs";
    directories[0].directories.emplace("exefs", 1);
    directories[0].directories.emplace("romfs", 2);
    std::vector<RomMember> files;
    auto ensure_directory = [&](std::size_t parent, const std::string& name) {
        const auto found = directories[parent].directories.find(name);
        if (found != directories[parent].directories.end()) return found->second;
        const auto index = directories.size();
        directories[parent].directories.emplace(name, index);
        directories.push_back(DirectoryNode{name, {}, {}});
        return index;
    };
    for (const RomMember& member : members) {
        if (member.path.empty() || member.bytes.empty()) {
            throw std::runtime_error("Synthetic RomFS member needs a path and a payload");
        }
        std::size_t directory = 2;
        std::size_t start = 0;
        for (;;) {
            const auto slash = member.path.find('/', start);
            if (slash == std::string::npos) {
                const std::string name = member.path.substr(start);
                if (name.empty() || directories[directory].files.contains(name)) {
                    throw std::runtime_error("Synthetic RomFS path is not a unique file");
                }
                directories[directory].files.emplace(name, files.size());
                files.push_back(RomMember{name, member.bytes});
                break;
            }
            const std::string part = member.path.substr(start, slash - start);
            if (part.empty()) throw std::runtime_error("Synthetic RomFS path has an empty component");
            directory = ensure_directory(directory, part);
            start = slash + 1;
        }
    }

    std::string pool;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> directory_names(directories.size());
    for (std::size_t index = 0; index < directories.size(); ++index) {
        if (directories[index].name.empty()) continue;
        directory_names[index] = {static_cast<std::uint32_t>(pool.size()),
                                  static_cast<std::uint32_t>(directories[index].name.size())};
        pool += directories[index].name;
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> file_names(files.size());
    for (std::size_t index = 0; index < files.size(); ++index) {
        file_names[index] = {static_cast<std::uint32_t>(pool.size()),
                             static_cast<std::uint32_t>(files[index].path.size())};
        pool += files[index].path;
    }

    std::vector<std::vector<std::uint32_t>> children(directories.size());
    std::size_t child_count = 0;
    for (std::size_t index = 0; index < directories.size(); ++index) {
        for (const auto& [name, child] : directories[index].directories) {
            static_cast<void>(name);
            children[index].push_back(static_cast<std::uint32_t>(child));
        }
        for (const auto& [name, child] : directories[index].files) {
            static_cast<void>(name);
            children[index].push_back(static_cast<std::uint32_t>(directories.size() + child));
        }
        child_count += children[index].size();
    }
    const std::size_t records_end = 0x70 + directories.size() * 0x18 + files.size() * 0x50;
    const std::size_t pool_at = records_end + child_count * 4;
    const std::size_t toc_size = pool_at + pool.size();
    const std::size_t blob = align16(0x80 + toc_size);

    std::vector<std::uint64_t> file_offsets(files.size());
    std::vector<std::array<std::byte, 32>> file_hashes(files.size());
    std::size_t cursor = blob;
    for (std::size_t index = 0; index < files.size(); ++index) {
        file_offsets[index] = cursor;
        file_hashes[index] = digest_bytes(files[index].bytes);
        cursor = align16(cursor + files[index].bytes.size());
    }

    std::vector<std::byte> archive(cursor, std::byte{0});
    put_le(archive, 0, 0x0150554eu);
    put_le(archive, 4, 0x80, 2);
    put_le(archive, 6, 1, 2);
    put_le(archive, 8, 0x80, 8);
    put_le(archive, 16, toc_size, 8);
    put_le(archive, 0x38, blob, 8);
    put_le(archive, 0x40, archive.size(), 8);
    put_le(archive, 0x48, 1);

    std::vector<std::byte> toc(toc_size, std::byte{0});
    put_le(toc, 0, 0x01434f54u);
    put_le(toc, 4, 1);
    put_le(toc, 8, directories.size(), 8);
    put_le(toc, 16, files.size(), 8);
    put_le(toc, 24, pool_at, 8);
    put_le(toc, 32, pool.size(), 8);
    put_le(toc, 0x30, 0x0100000000010000ull, 8);
    put_le(toc, 0x38, 0);
    put_le(toc, 0x3c, 1);
    put_le(toc, 0x40, 2);
    put_le(toc, 0x44, 0x10000u);
    std::size_t child_cursor = 0;
    for (std::size_t index = 0; index < directories.size(); ++index) {
        const std::size_t at = 0x70 + index * 0x18;
        put_le(toc, at, directory_names[index].first);
        put_le(toc, at + 4, directory_names[index].second);
        put_le(toc, at + 8, static_cast<std::uint32_t>(child_cursor));
        put_le(toc, at + 12, static_cast<std::uint32_t>(children[index].size()));
        for (const std::uint32_t child : children[index]) {
            put_le(toc, records_end + child_cursor * 4, child);
            ++child_cursor;
        }
    }
    for (std::size_t index = 0; index < files.size(); ++index) {
        const std::size_t at = 0x70 + directories.size() * 0x18 + index * 0x50;
        put_le(toc, at, file_names[index].first);
        put_le(toc, at + 4, file_names[index].second);
        put_le(toc, at + 8, files[index].bytes.size(), 8);
        put_le(toc, at + 16, file_offsets[index], 8);
        put_le(toc, at + 24, files[index].bytes.size(), 8);
        std::memcpy(toc.data() + at + 32, file_hashes[index].data(), file_hashes[index].size());
    }
    if (!pool.empty()) std::memcpy(toc.data() + pool_at, pool.data(), pool.size());
    const auto toc_hash = digest_bytes(toc);
    std::memcpy(archive.data() + 0x18, toc_hash.data(), toc_hash.size());
    std::memcpy(archive.data() + 0x80, toc.data(), toc.size());
    for (std::size_t index = 0; index < files.size(); ++index) {
        std::memcpy(archive.data() + static_cast<std::size_t>(file_offsets[index]), files[index].bytes.data(),
                    files[index].bytes.size());
    }
    return archive;
}

// Shader/a.bin is one native BNSH. Shader/b.bin is that BNSH followed by a BAHS.
inline std::vector<std::byte> native_shader_nup() {
    const std::vector<std::byte> bnsh = native_bnsh();
    std::vector<std::byte> both = bnsh;
    const std::vector<std::byte> bahs = native_bahs();
    both.insert(both.end(), bahs.begin(), bahs.end());
    return nup_archive({RomMember{"Shader/a.bin", bnsh}, RomMember{"Shader/b.bin", std::move(both)}});
}

}  // namespace armigo::pack::shader_golden
