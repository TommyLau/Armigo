// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pso_manifest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

namespace {

using namespace armigo::platform::metal::nvn_driver;

int failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)

PsoRow row(char kind, std::vector<std::string> fields) { return {kind, std::move(fields)}; }

std::string read_all(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void parsing() {
    // Order is the first-use order; a repeat is dropped at its later place;
    // comments, blanks and malformed rows are skipped.
    const std::string text =
        "# header\n"
        "S\t1.aa.0\n"
        "R\t0.bb.2\t-\tv1:00ff\n"
        "\n"
        "C\t5.cc.1x1x1s0\n"
        "S\t1.aa.0\n"
        "X\tunknown kind\n"
        "S\ttoo\tmany\n"
        "R\tonly\ttwo\n"
        "S\t\n"
        "R\t0.bb.2\t1.dd.0\tv1:0100\r\n"
        "K\tclear_vertex\tclear_color_frag\t70\t15\t0\t0\t4\n"
        "K\tclear_vertex\tclear_color_frag\t70\n"
        "O\t0.bb.2\t1.ee.0\tv1:0100\n"
        "O\tonly\ttwo\n";
    const auto rows = parse_pso_manifest(text);
    CHECK(rows.size() == 6);
    CHECK(rows[0] == row('S', {"1.aa.0"}));
    CHECK(rows[1] == row('R', {"0.bb.2", "-", "v1:00ff"}));
    CHECK(rows[2] == row('C', {"5.cc.1x1x1s0"}));
    CHECK(rows[3] == row('R', {"0.bb.2", "1.dd.0", "v1:0100"}));
    CHECK(rows[4] == row('K', {"clear_vertex", "clear_color_frag", "70", "15", "0", "0", "4"}));
    CHECK(rows[5] == row('O', {"0.bb.2", "1.ee.0", "v1:0100"}));
    CHECK(rows[1].text() == "R\t0.bb.2\t-\tv1:00ff");
    // Text round trip.
    for (const auto& value : rows) CHECK(parse_pso_row(value.text()) == value);
}

void merging() {
    const auto seed = parse_pso_manifest("S\ta\nS\tb\nC\tc\n");
    const auto user = parse_pso_manifest("S\tz\nS\tb\nC\tc\nS\ty\n");
    const auto merged = merge_pso_manifests(seed, user);
    // Seed order first, then the user's rows the seed lacks, in their order.
    CHECK(merged.size() == 5);
    CHECK(merged[0].fields[0] == "a" && merged[1].fields[0] == "b" && merged[2].fields[0] == "c");
    CHECK(merged[3].fields[0] == "z" && merged[4].fields[0] == "y");
    CHECK(merge_pso_manifests({}, user).size() == 4);
}

void hex() {
    const unsigned char bytes[] = {0x00, 0x7f, 0xa5, 0xff};
    const auto text = pso_hex_encode(bytes, sizeof bytes);
    CHECK(text == "007fa5ff");
    unsigned char back[4]{};
    CHECK(pso_hex_decode(text, back, sizeof back) && back[2] == 0xa5 && back[3] == 0xff);
    CHECK(!pso_hex_decode("007fa5f", back, sizeof back));
    CHECK(!pso_hex_decode("007fa5FF", back, sizeof back));
    CHECK(!pso_hex_decode("007fa5fg", back, sizeof back));
}

void recording() {
    const auto dir = std::filesystem::temp_directory_path() / ("pso_manifest_tests." + std::to_string(getpid()));
    std::filesystem::remove_all(dir);
    const auto record = (dir / "nested" / "record.tsv").string();
    const auto user = (dir / "user.tsv").string();
    // The user file already holds one row from an earlier run.
    std::filesystem::create_directories(dir);
    { std::ofstream(user) << "S\told\n"; }
    const auto known = parse_pso_manifest("S\tseed\nC\tseedc\n");
    {
        PsoRecorder recorder(record, user, known);
        CHECK(recorder.enabled());
        recorder.note(row('S', {"seed"}));       // in the record file, not the user file
        recorder.note(row('S', {"new1"}));
        recorder.note(row('R', {"v", "-", "v1:00"}));
        recorder.note(row('S', {"new1"}));       // repeat
        recorder.note(row('S', {"old"}));        // already in the user file
        recorder.note(row('C', {"seedc"}));
        recorder.note(row('S', {"new2"}));
    }
    CHECK(read_all(record) == "S\tseed\nS\tnew1\nR\tv\t-\tv1:00\nS\told\nC\tseedc\nS\tnew2\n");
    CHECK(read_all(user) == "S\told\nS\tnew1\nR\tv\t-\tv1:00\nS\tnew2\n");
    {
        // A later session appends only what the files lack, in first-use order.
        PsoRecorder recorder(record, user, known);
        recorder.note(row('S', {"new2"}));
        recorder.note(row('S', {"new3"}));
    }
    CHECK(read_all(record) == "S\tseed\nS\tnew1\nR\tv\t-\tv1:00\nS\told\nC\tseedc\nS\tnew2\nS\tnew3\n");
    CHECK(read_all(user) == "S\told\nS\tnew1\nR\tv\t-\tv1:00\nS\tnew2\nS\tnew3\n");
    {
        // A file cut off mid-row gets a line break before the next row.
        { std::ofstream(user, std::ios::binary | std::ios::trunc) << "S\told\nS\tcut"; }
        PsoRecorder recorder(record, user, known);
        recorder.note(row('S', {"after"}));
    }
    CHECK(read_all(user) == "S\told\nS\tcut\nS\tafter\n");
    // No paths: nothing is written and nothing breaks.
    PsoRecorder idle("", "", known);
    CHECK(!idle.enabled());
    idle.note(row('S', {"x"}));
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    parsing();
    merging();
    hex();
    recording();
    std::printf("pso manifest failures: %d\n", failures);
    return failures ? 1 : 0;
}
