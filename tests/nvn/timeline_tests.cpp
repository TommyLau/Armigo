// ============================================================================
// Armigo NVN - Frame Timeline Trace
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/nvn/timeline.h"
#include "timeline_ring.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <spawn.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;
using namespace armigo::nvn;
using namespace armigo::nvn::detail;
namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #value); ++failures; } } while (false)

void bounded_capture() {
    TimelineRing ring(4);
    for (unsigned i = 0; i < 4; ++i) CHECK(ring.push({10 + i, i, 99, 7}));
    for (unsigned i = 0; i < 3; ++i) CHECK(!ring.push({0, 0, 0, 0}));
    CHECK(ring.dropped() == 3);
    TimelineRecord record{};
    for (unsigned i = 0; i < 4; ++i) {
        CHECK(ring.pop(record));
        CHECK(record.time == 10 + i && record.key == i && record.value == 99 && record.event == 7);
    }
    CHECK(!ring.pop(record));
    CHECK(ring.push({20, 4, 99, 7}));
    ring.close();
    CHECK(!ring.push({30, 5, 99, 7}));
    unsigned emitted = 0;
    CHECK(ring.finish([&](const auto& item) { ++emitted; CHECK(item.key == 4); }) == 0);
    CHECK(emitted == 1 && ring.dropped() == 3);
}

void concurrent_order_and_wrap() {
    TimelineRing ring(64);
    constexpr unsigned threads = 6, count = 4000;
    std::atomic<unsigned> done{0};
    std::vector<std::thread> producers;
    for (unsigned t = 0; t < threads; ++t) producers.emplace_back([&, t] {
        for (unsigned i = 0; i < count; ++i)
            while (!ring.push({i, t, i, t})) std::this_thread::yield();
        done.fetch_add(1, std::memory_order_release);
    });
    std::array<unsigned, threads> next{};
    unsigned read = 0;
    while (read != threads * count) {
        TimelineRecord item{};
        if (!ring.pop(item)) { std::this_thread::yield(); continue; }
        CHECK(item.key < threads);
        if (item.key < threads) CHECK(item.value == next[item.key]++);
        CHECK(item.time == item.value && item.event == item.key);
        ++read;
    }
    for (auto& thread : producers) thread.join();
    CHECK(done.load(std::memory_order_acquire) == threads);
    ring.close();
    CHECK(ring.finish([](const auto&) {}) == 0);
    for (auto value : next) CHECK(value == count);
}

void close_with_producers() {
    TimelineRing ring(64);
    std::atomic<unsigned> ready{0};
    std::atomic<bool> start{false};
    std::atomic<unsigned> accepted{0};
    std::vector<std::thread> producers;
    for (unsigned t = 0; t < 4; ++t) producers.emplace_back([&, t] {
        ++ready;
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        for (unsigned i = 0; i < 10000; ++i)
            if (ring.push({i, t, i, t})) ++accepted;
    });
    while (ready != 4) std::this_thread::yield();
    start.store(true, std::memory_order_release);
    TimelineRecord record{};
    unsigned emitted = 0;
    while (emitted < 32) if (ring.pop(record)) ++emitted;
    ring.close();
    for (auto& thread : producers) thread.join();
    CHECK(ring.finish([&](const auto& item) { CHECK(item.key < 4 && item.value < 10000); ++emitted; }) == 0);
    CHECK(emitted == accepted && !ring.push({0, 0, 0, 0}));
}

void stalled_reservation_at_shutdown() {
    TimelineRing ring(4);
    std::atomic<bool> reserved{false}, release{false};
    std::thread first([&] {
        ring.push({11, 1, 111, 1}, [&] {
            reserved.store(true, std::memory_order_release);
            while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
        });
    });
    while (!reserved.load(std::memory_order_acquire)) std::this_thread::yield();
    CHECK(ring.push({22, 2, 222, 2}));
    TimelineRecord record{};
    CHECK(!ring.pop(record));
    ring.close();
    unsigned emitted = 0;
    CHECK(ring.finish([&](const auto& item) {
        ++emitted;
        CHECK(item.time == 22 && item.key == 2 && item.value == 222 && item.event == 2);
    }) == 1);
    CHECK(emitted == 1);
    release.store(true, std::memory_order_release);
    first.join();
    CHECK(ring.finish([&](const auto&) { ++emitted; }) == 0 && emitted == 1);
}

void shutdown_before_loss_publication() {
    TimelineAdmission admission;
    TimelineRing ring(2);
    CHECK(ring.push({1, 1, 1, 1}) && ring.push({2, 2, 2, 2}));
    std::atomic<bool> entered{false}, release{false};
    std::thread producer([&] {
        if (!admission.begin()) return;
        entered.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
        // Queue or dictionary failure can be observed after the footer. The
        // captured admission count must already make that trace incomplete.
        ring.push({3, 3, 3, 3});
        admission.end();
    });
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    const auto active = admission.close();
    CHECK(active == 1 && !admission.begin());
    ring.close();
    CHECK(ring.finish([](const auto&) {}) == 0 && ring.dropped() == 0);
    release.store(true, std::memory_order_release);
    producer.join();
    CHECK(active == 1 && admission.close() == 0);
}

void event_name_lifetime() {
    TimelineSymbols symbols;
    char transient[] = "wait_begin:present";
    const auto first = symbols.intern(transient);
    transient[5] = 'X';
    CHECK(std::string(symbols.name(first)) == "wait_begin:present");
    CHECK(first == symbols.intern("wait_begin:present"));
    CHECK(first == symbols.intern("wait_begin:", "present"));
    CHECK(std::string(symbols.name(symbols.intern(nullptr))) == "?");
    std::vector<std::thread> workers;
    std::atomic<unsigned> incorrect{0};
    for (unsigned i = 0; i < 8; ++i) workers.emplace_back([&] {
        for (unsigned j = 0; j < 1000; ++j) {
            const auto id = symbols.intern("concurrent_event");
            if (id == TimelineSymbols::invalid || std::string(symbols.name(id)) != "concurrent_event") ++incorrect;
        }
    });
    for (auto& thread : workers) thread.join();
    CHECK(incorrect == 0);
    CHECK(symbols.intern(std::string(64, 'x').c_str()) == TimelineSymbols::invalid);
    CHECK(symbols.dropped() == 1);
    for (unsigned i = 0; i < 1024; ++i) symbols.intern(("unique_" + std::to_string(i)).c_str());
    CHECK(symbols.dropped() > 1);
}

void emit_child() {
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < 4; ++t) workers.emplace_back([t] {
        for (unsigned i = 0; i < 1000; ++i) {
            const std::string name = t == 0 ? "pres_wait_slot" : t == 1 ? "present_call" : "worker_" + std::to_string(t);
            if (t == 3) timeline_wait_event((i & 1) == 0, "worker", t, i);
            else timeline_event_at(100 + i, name.c_str(), t, i);
        }
    });
    for (auto& thread : workers) thread.join();
}

void check_output(const char* executable, const std::filesystem::path& directory,
                  const char* filter, const char* capacity) {
    const auto file = directory / (std::string("trace-") + filter + "-" + capacity + ".tsv");
    std::vector<std::string> environment;
    for (char** item = environ; *item; ++item)
        if (std::string_view(*item).find("ARMIGO_NVN_TIMELINE") != 0) environment.emplace_back(*item);
    environment.push_back("ARMIGO_NVN_TIMELINE=" + file.string());
    environment.push_back(std::string("ARMIGO_NVN_TIMELINE_PRESENT_ONLY=") + filter);
    environment.push_back(std::string("ARMIGO_NVN_TIMELINE_CAPACITY=") + capacity);
    std::vector<char*> env;
    for (auto& item : environment) env.push_back(item.data());
    env.push_back(nullptr);
    char mode[] = "--emit";
    char* args[] = {const_cast<char*>(executable), mode, nullptr};
    pid_t child = 0;
    const int spawned = posix_spawn(&child, executable, nullptr, nullptr, args, env.data());
    CHECK(spawned == 0);
    if (spawned) return;
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
    std::ifstream input(file);
    std::string line;
    CHECK(static_cast<bool>(std::getline(input, line)) && line == "0\ttimeline_start\t0\t0");
    std::array<int, 4> previous{-1, -1, -1, -1};
    unsigned records = 0;
    std::uint64_t overflow = 0;
    while (std::getline(input, line)) {
        CHECK(std::count(line.begin(), line.end(), '\t') == 3);
        std::istringstream row(line);
        std::uint64_t time = 0, key = 0, value = 0;
        std::string event;
        CHECK(static_cast<bool>(row >> time >> event >> key >> value));
        if (event == "timeline_overflow") { CHECK(time == 0 && value == 0); overflow += key; continue; }
        CHECK(key < 4 && value < 1000 && (key == 3 ? time > 0 : time == 100 + value));
        if (key < 4) { CHECK(static_cast<int>(value) > previous[key]); previous[key] = static_cast<int>(value); }
        CHECK(event == (key == 0 ? "pres_wait_slot" : key == 1 ? "present_call" : key == 3 ? ((value & 1) == 0 ? "wait_begin:worker" : "wait_end:worker") : "worker_" + std::to_string(key)));
        if (std::string_view(filter) == "1") CHECK(key < 2);
        ++records;
    }
    CHECK(records + overflow == (std::string_view(filter) == "1" ? 2000 : 4000));
    if (std::string_view(capacity) == "65536") CHECK(overflow == 0);
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--emit") { emit_child(); return 0; }
    bounded_capture();
    concurrent_order_and_wrap();
    close_with_producers();
    stalled_reservation_at_shutdown();
    shutdown_before_loss_publication();
    event_name_lifetime();
    auto path = (std::filesystem::temp_directory_path() / "armigo-timeline-XXXXXX").string();
    const char* created = mkdtemp(path.data());
    CHECK(created != nullptr);
    if (created) {
        check_output(argv[0], created, "0", "65536");
        check_output(argv[0], created, "1", "65536");
        check_output(argv[0], created, "0", "2");
        std::filesystem::remove_all(created);
    }
    return failures ? 1 : 0;
}
