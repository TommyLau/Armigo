// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "pso_manifest.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace armigo::platform::metal::nvn_driver {
namespace {

std::size_t field_count(char kind) {
    switch (kind) {
        case 'S': case 'C': return 1;
        case 'R': case 'O': return 3;
        case 'K': return 7;
        default: return 0;
    }
}

} // namespace

std::string PsoRow::text() const {
    std::string line(1, kind);
    for (const auto& field : fields) {
        line += '\t';
        line += field;
    }
    return line;
}

std::optional<PsoRow> parse_pso_row(std::string_view line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.remove_suffix(1);
    if (line.size() < 3 || line[0] == '#' || line[1] != '\t') return std::nullopt;
    PsoRow row;
    row.kind = line[0];
    const auto expected = field_count(row.kind);
    if (!expected) return std::nullopt;
    line.remove_prefix(2);
    for (;;) {
        const auto tab = line.find('\t');
        row.fields.emplace_back(line.substr(0, tab));
        if (row.fields.back().empty()) return std::nullopt;
        if (tab == std::string_view::npos) break;
        line.remove_prefix(tab + 1);
    }
    if (row.fields.size() != expected) return std::nullopt;
    return row;
}

std::vector<PsoRow> parse_pso_manifest(std::string_view text) {
    std::vector<PsoRow> rows;
    std::unordered_set<std::string> seen;
    while (!text.empty()) {
        const auto end = text.find('\n');
        if (auto row = parse_pso_row(text.substr(0, end)); row && seen.insert(row->text()).second)
            rows.push_back(std::move(*row));
        if (end == std::string_view::npos) break;
        text.remove_prefix(end + 1);
    }
    return rows;
}

std::vector<PsoRow> merge_pso_manifests(const std::vector<PsoRow>& seed, const std::vector<PsoRow>& user) {
    std::vector<PsoRow> rows = seed;
    std::unordered_set<std::string> seen;
    for (const auto& row : rows) seen.insert(row.text());
    for (const auto& row : user)
        if (seen.insert(row.text()).second) rows.push_back(row);
    return rows;
}

std::string pso_hex_encode(const void* bytes, std::size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(size * 2);
    for (std::size_t i = 0; i != size; ++i) {
        const auto byte = static_cast<const unsigned char*>(bytes)[i];
        text += digits[byte >> 4];
        text += digits[byte & 15];
    }
    return text;
}

bool pso_hex_decode(std::string_view text, void* bytes, std::size_t size) {
    if (text.size() != size * 2) return false;
    const auto nibble = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
    };
    for (std::size_t i = 0; i != size; ++i) {
        const int high = nibble(text[2 * i]), low = nibble(text[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        static_cast<unsigned char*>(bytes)[i] = static_cast<unsigned char>(high << 4 | low);
    }
    return true;
}

PsoRecorder::PsoRecorder(const std::string& record_path, const std::string& user_path,
                         const std::vector<PsoRow>& known) {
    open(record_, record_path);
    open(user_, user_path);
    if (user_.file)
        for (const auto& row : known) user_.seen.insert(row.text());
}

PsoRecorder::~PsoRecorder() {
    if (record_.file) std::fclose(record_.file);
    if (user_.file) std::fclose(user_.file);
}

void PsoRecorder::open(Sink& sink, const std::string& path) {
    if (path.empty()) return;
    std::error_code error;
    const std::filesystem::path file(path);
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path(), error);
    {
        std::ifstream in(file, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        for (const auto& row : parse_pso_manifest(text)) sink.seen.insert(row.text());
    }
    // A file cut off mid-line would swallow the next row appended to it.
    bool needs_newline = false;
    if (std::FILE* tail = std::fopen(path.c_str(), "rb")) {
        if (std::fseek(tail, -1, SEEK_END) == 0) needs_newline = std::fgetc(tail) != '\n';
        std::fclose(tail);
    }
    sink.file = std::fopen(path.c_str(), "a");
    if (!sink.file) armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: cannot append to the pipeline manifest %s\n", path.c_str());
    else if (needs_newline) std::fputc('\n', sink.file);
}

void PsoRecorder::append(Sink& sink, const std::string& line) {
    if (!sink.file || !sink.seen.insert(line).second) return;
    std::fputs(line.c_str(), sink.file);
    std::fputc('\n', sink.file);
    const auto now = std::chrono::steady_clock::now();
    if (now - sink.flushed >= std::chrono::seconds(1)) {
        std::fflush(sink.file);
        sink.flushed = now;
    }
}

void PsoRecorder::note(const PsoRow& row) {
    const auto line = row.text();
    const std::lock_guard lock(mutex_);
    append(record_, line);
    append(user_, line);
}

} // namespace armigo::platform::metal::nvn_driver
