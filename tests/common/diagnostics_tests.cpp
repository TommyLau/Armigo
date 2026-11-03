// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* text) {
    if (condition) return;
    std::cerr << "FAIL: " << text << '\n';
    ++failures;
}

struct Capture {
    std::mutex mutex;
    std::vector<std::pair<armigo::diag::Level, std::string>> lines;
};

void sink(armigo::diag::Level level, std::string_view line, void* context) {
    auto* capture = static_cast<Capture*>(context);
    const std::lock_guard lock(capture->mutex);
    capture->lines.emplace_back(level, std::string(line));
}

void discard_sink(armigo::diag::Level, std::string_view, void*) {}

void clear_environment() {
    unsetenv("ARMIGO_LOG");
    unsetenv("ARMIGO_FRAME_STATS");
}

bool contains(const Capture& capture, std::string_view text) {
    for (const auto& [level, line] : capture.lines) {
        (void)level;
        if (line.find(text) != std::string::npos) return true;
    }
    return false;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), {});
}

void test_parse() {
    using armigo::diag::Level;
    check(armigo::diag::parse_level("error") == Level::error, "parse error");
    check(armigo::diag::parse_level(" Warning ") == Level::warning, "parse warning");
    check(armigo::diag::parse_level("INFO") == Level::info, "parse info");
    check(armigo::diag::parse_level("trace") == Level::trace, "parse trace");
    check(!armigo::diag::parse_level("verbose"), "reject unknown level");
    check(!armigo::diag::parse_level(""), "reject empty level");
    check(armigo::diag::parse_frame_stats("1") == true, "stats 1");
    check(armigo::diag::parse_frame_stats("true") == true, "stats true");
    check(armigo::diag::parse_frame_stats("ON") == true, "stats on");
    check(armigo::diag::parse_frame_stats("yes") == true, "stats yes");
    check(armigo::diag::parse_frame_stats("0") == false, "stats 0");
    check(armigo::diag::parse_frame_stats("false") == false, "stats false");
    check(armigo::diag::parse_frame_stats("off") == false, "stats off");
    check(armigo::diag::parse_frame_stats("no") == false, "stats no");
    check(!armigo::diag::parse_frame_stats("maybe"), "reject unknown stats");
}

void test_filtering() {
    clear_environment();
    armigo::diag::testing_reset();
    Capture capture;
    armigo::diag::testing_set_sink(sink, &capture);
    armigo::diag::testing_set_level(armigo::diag::Level::warning);
    armigo::diag::testing_set_frame_stats(false);
    armigo::diag::error("error-line %d\n", 1);
    armigo::diag::warning("warning-line %d\n", 2);
    armigo::diag::info("info-line %d\n", 3);
    armigo::diag::trace("trace-line %d\n", 4);
    armigo::diag::frame("frame-line %d\n", 5);
    check(contains(capture, "error-line 1"), "error passes a warning minimum");
    check(contains(capture, "warning-line 2"), "warning passes a warning minimum");
    check(!contains(capture, "info-line"), "info is filtered");
    check(!contains(capture, "trace-line"), "trace is filtered");
    check(!contains(capture, "frame-line"), "frame stats are filtered when off");
    check(!armigo::diag::enabled(armigo::diag::Level::info), "enabled(info) is false");
    check(armigo::diag::enabled(armigo::diag::Level::error), "enabled(error) is true");

    capture.lines.clear();
    armigo::diag::testing_set_level(armigo::diag::Level::trace);
    armigo::diag::testing_set_frame_stats(true);
    armigo::diag::info("info-on\n");
    armigo::diag::trace("trace-on\n");
    armigo::diag::frame("frame-on\n");
    check(contains(capture, "info-on"), "info passes trace");
    check(contains(capture, "trace-on"), "trace passes trace");
    check(contains(capture, "frame-on"), "frame stats emit when on");
    check(armigo::diag::frame_stats_enabled(), "frame stats flag is on");
}

void test_fragments() {
    clear_environment();
    armigo::diag::testing_reset();
    Capture capture;
    armigo::diag::testing_set_sink(sink, &capture);
    armigo::diag::testing_set_level(armigo::diag::Level::info);
    armigo::diag::info("hello ");
    armigo::diag::info("world");
    armigo::diag::newline();
    check(capture.lines.size() == 1, "fragments join into one line");
    check(!capture.lines.empty() && capture.lines[0].second == "hello world\n", "joined text");

    capture.lines.clear();
    armigo::diag::testing_set_level(armigo::diag::Level::error);
    armigo::diag::info("hidden ");
    armigo::diag::newline();
    check(capture.lines.empty(), "a filtered fragment does not finish as a blank line");
}

void test_environment() {
    clear_environment();
    armigo::diag::testing_reset();
    check(armigo::diag::minimum_level() == armigo::diag::Level::info, "development level");
    check(armigo::diag::frame_stats_enabled(), "development frame stats");

    setenv("ARMIGO_LOG", "error", 1);
    setenv("ARMIGO_FRAME_STATS", "0", 1);
    armigo::diag::use_development_defaults();
    check(armigo::diag::minimum_level() == armigo::diag::Level::error, "env level overrides development");
    check(!armigo::diag::frame_stats_enabled(), "env stats override development");

    clear_environment();
    armigo::diag::use_player_defaults();
    check(armigo::diag::minimum_level() == armigo::diag::Level::warning, "player level");
    check(!armigo::diag::frame_stats_enabled(), "player frame stats off");

    setenv("ARMIGO_LOG", "trace", 1);
    setenv("ARMIGO_FRAME_STATS", "1", 1);
    armigo::diag::use_player_defaults();
    check(armigo::diag::minimum_level() == armigo::diag::Level::trace, "env overrides the player level");
    check(armigo::diag::frame_stats_enabled(), "env overrides the player frame stats");

    setenv("ARMIGO_LOG", "nope", 1);
    unsetenv("ARMIGO_FRAME_STATS");
    armigo::diag::use_player_defaults();
    check(armigo::diag::minimum_level() == armigo::diag::Level::warning, "invalid level keeps the player default");
    check(!armigo::diag::frame_stats_enabled(), "unset stats keep the player default");
    clear_environment();
}

void test_last_run_log() {
    clear_environment();
    armigo::diag::testing_reset();
    const auto directory = std::filesystem::temp_directory_path() / "armigo-diagnostics-tests";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto path = directory / "last-run.log";
    check(armigo::diag::open_last_run_log(path.c_str()), "open last-run log");
    armigo::diag::testing_set_sink(discard_sink, nullptr);
    armigo::diag::testing_set_level(armigo::diag::Level::trace);
    armigo::diag::error("kept-error\n");
    armigo::diag::warning("kept-warning\n");
    armigo::diag::info("skipped-info\n");
    armigo::diag::trace("skipped-trace\n");
    armigo::diag::frame("skipped-frame\n");
    armigo::diag::opt_in("skipped-opt\n");
    armigo::diag::testing_close_log();
    const auto first = read_file(path);
    check(first.find("kept-error") != std::string::npos, "file keeps errors");
    check(first.find("kept-warning") != std::string::npos, "file keeps warnings");
    check(first.find("skipped-info") == std::string::npos, "file skips info");
    check(first.find("skipped-trace") == std::string::npos, "file skips trace");
    check(first.find("skipped-frame") == std::string::npos, "file skips frame stats");
    check(first.find("skipped-opt") == std::string::npos, "file skips opt-in traces");

    check(armigo::diag::open_last_run_log(path.c_str()), "reopen truncates");
    armigo::diag::warning("after-truncate\n");
    armigo::diag::testing_close_log();
    const auto truncated = read_file(path);
    check(truncated.find("kept-error") == std::string::npos, "reopen drops the previous launch");
    check(truncated.find("after-truncate") != std::string::npos, "reopen keeps the new line");

    check(armigo::diag::open_last_run_log(path.c_str()), "open for the cap");
    constexpr int line_bytes = 1024;
    std::string line(static_cast<std::size_t>(line_bytes - 1), 'a');
    line.push_back('\n');
    // 1 MiB holds 1024 of these lines. Write past the cap, then a unique tail.
    for (int i = 0; i < 1100; ++i) armigo::diag::warning("%s", line.c_str());
    armigo::diag::warning("TAIL-MARKER-SHOULD-NOT-FIT\n");
    const auto written = armigo::diag::testing_last_run_bytes();
    armigo::diag::testing_close_log();
    const auto capped = read_file(path);
    check(capped.size() <= 1024ull * 1024ull, "last-run log stays within 1 MB");
    check(written <= 1024ull * 1024ull, "recorded byte count stays within 1 MB");
    check(capped.find("stopped at 1 MB") != std::string::npos, "cap notice is present");
    check(capped.find("TAIL-MARKER-SHOULD-NOT-FIT") == std::string::npos, "lines past the cap are omitted");
    std::filesystem::remove_all(directory);
}

void test_threads() {
    clear_environment();
    armigo::diag::testing_reset();
    Capture capture;
    armigo::diag::testing_set_sink(sink, &capture);
    armigo::diag::testing_set_level(armigo::diag::Level::error);
    constexpr int threads = 4;
    constexpr int lines = 40;
    std::vector<std::thread> workers;
    for (int thread = 0; thread < threads; ++thread) {
        workers.emplace_back([thread] {
            for (int line = 0; line < lines; ++line)
                armigo::diag::error("thread %d line %d\n", thread, line);
        });
    }
    for (auto& worker : workers) worker.join();
    check(static_cast<int>(capture.lines.size()) == threads * lines, "every threaded line arrived");
    for (const auto& [level, text] : capture.lines) {
        (void)level;
        check(text.find('\n') == text.size() - 1, "a threaded write is one whole line");
        check(text.rfind('\n') == text.find('\n'), "a threaded write has one newline");
    }
}

} // namespace

int main() {
    test_parse();
    test_filtering();
    test_fragments();
    test_environment();
    test_last_run_log();
    test_threads();
    if (failures) {
        std::cerr << failures << " diagnostics check(s) failed\n";
        return 1;
    }
    return 0;
}
