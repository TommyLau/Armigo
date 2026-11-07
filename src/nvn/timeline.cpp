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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace armigo::nvn {
namespace {

constexpr std::size_t default_capacity = 1 << 20;

std::size_t buffer_capacity() {
    const char* value = std::getenv("ARMIGO_NVN_TIMELINE_CAPACITY");
    if (!value) return default_capacity;
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (!*value || *end || parsed < 2 || parsed > (1 << 24)) {
        std::fprintf(stderr, "nvn_timeline: invalid capacity; using %zu records\n", default_capacity);
        return default_capacity;
    }
    return static_cast<std::size_t>(parsed);
}

struct TraceFile {
    std::FILE* file;
    const bool presentation_only;
    detail::TimelineAdmission admission;
    detail::TimelineRing ring;
    std::uint64_t unquiesced = 0;
    detail::TimelineSymbols symbols;
    std::thread writer;

    TraceFile(std::FILE* output, bool filter, std::size_t capacity)
        : file(output), presentation_only(filter), ring(capacity) {
        // This happens before main, not on a guest/replay/encode thread.
        std::fprintf(file, "0\ttimeline_start\t0\t0\n");
        std::fflush(file);
        writer = std::thread([this] { write(); });
    }

    bool accepts(const char* event) const noexcept {
        return !presentation_only || (event &&
            (std::strncmp(event, "pres_", 5) == 0 || std::strncmp(event, "present_", 8) == 0));
    }

    void append(std::uint64_t time, const char* event, std::uint64_t key, std::uint64_t value,
                const char* suffix = nullptr) noexcept {
        if (!admission.begin()) return;
        struct Leave {
            detail::TimelineAdmission& admission;
            ~Leave() { admission.end(); }
        } leave{admission};
        const auto id = symbols.intern(event, suffix);
        if (id != detail::TimelineSymbols::invalid) ring.push({time, key, value, id});
    }

    void emit(const detail::TimelineRecord& record) {
        std::fprintf(file, "%llu\t%s\t%llu\t%llu\n",
            static_cast<unsigned long long>(record.time), symbols.name(record.event),
            static_cast<unsigned long long>(record.key), static_cast<unsigned long long>(record.value));
    }

    void write() {
#if defined(__APPLE__)
        pthread_setname_np("nvn-timeline");
        pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
#endif
        while (!ring.closed()) {
            detail::TimelineRecord record;
            unsigned count = 0;
            while (count < 16384 && ring.pop(record)) { emit(record); ++count; }
            if (!count) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        const auto incomplete = ring.finish([this](const auto& record) { emit(record); });
        const auto dropped = ring.dropped(), names = symbols.dropped();
        if (dropped) std::fprintf(file, "0\ttimeline_overflow\t%llu\t0\n", static_cast<unsigned long long>(dropped));
        if (incomplete) std::fprintf(file, "0\ttimeline_incomplete\t%llu\t0\n", static_cast<unsigned long long>(incomplete));
        if (names) std::fprintf(file, "0\ttimeline_symbols_dropped\t%llu\t0\n", static_cast<unsigned long long>(names));
        if (unquiesced) std::fprintf(file, "0\ttimeline_unquiesced\t%llu\t0\n", static_cast<unsigned long long>(unquiesced));
        if (dropped || incomplete || names || unquiesced)
            std::fprintf(stderr, "nvn_timeline: dropped %llu full-ring, %llu unpublished and %llu event-name records; %llu attempts active at shutdown\n",
                static_cast<unsigned long long>(dropped), static_cast<unsigned long long>(incomplete),
                static_cast<unsigned long long>(names), static_cast<unsigned long long>(unquiesced));
        const bool failed = std::ferror(file);
        if (std::fclose(file) || failed)
            std::fprintf(stderr, "nvn_timeline: output could not be written completely\n");
    }

    void finish() {
        unquiesced = admission.close();
        ring.close();
        if (writer.joinable()) writer.join();
    }
};

TraceFile* initialize();
// Eager initialization keeps allocation, file opening and writer startup out
// of event callers. Retain the recorder through process exit so late runtime
// teardown callbacks cannot touch freed ring or symbol storage.
TraceFile* const trace = initialize();
void finish_trace() { if (trace) trace->finish(); }

TraceFile* initialize() {
    const char* path = std::getenv("ARMIGO_NVN_TIMELINE");
    if (!path || !*path) return nullptr;
    auto* file = std::fopen(path, "w");
    if (!file) {
        std::fprintf(stderr, "nvn_timeline: output file could not be opened\n");
        return nullptr;
    }
    const char* filter = std::getenv("ARMIGO_NVN_TIMELINE_PRESENT_ONLY");
    try {
        auto* result = new TraceFile(file, filter && std::strcmp(filter, "1") == 0, buffer_capacity());
        if (std::atexit(finish_trace) != 0) {
            result->finish();
            std::fprintf(stderr, "nvn_timeline: exit drain registration failed\n");
            return nullptr;
        }
        return result;
    } catch (...) {
        std::fclose(file);
        std::fprintf(stderr, "nvn_timeline: recorder initialization failed\n");
        return nullptr;
    }
}

} // namespace

bool timeline_enabled() noexcept { return trace != nullptr; }

std::uint64_t timeline_now_ns() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void timeline_event(const char* event, std::uint64_t key, std::uint64_t value) noexcept {
    if (trace && trace->accepts(event)) trace->append(timeline_now_ns(), event, key, value);
}

void timeline_wait_event(bool begin, const char* tag, std::uint64_t key, std::uint64_t value) noexcept {
    const char* prefix = begin ? "wait_begin:" : "wait_end:";
    if (trace && trace->accepts(prefix)) trace->append(timeline_now_ns(), prefix, key, value, tag ? tag : "wait");
}

void timeline_event_at(std::uint64_t time, const char* event, std::uint64_t key, std::uint64_t value) noexcept {
    if (trace && trace->accepts(event)) trace->append(time, event, key, value);
}

} // namespace armigo::nvn
