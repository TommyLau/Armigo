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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace armigo::nvn {
namespace {

struct TraceFile {
    std::FILE* file = nullptr;
    std::mutex mutex;

    TraceFile() {
        const char* path = std::getenv("ARMIGO_NVN_TIMELINE");
        if (!path || !*path) return;
        file = std::fopen(path, "w");
        if (!file) return;
        // The first record names the clock so a reader can align the trace with
        // the harness's own log lines.
        std::fprintf(file, "0\ttimeline_start\t0\t0\n");
        std::fflush(file);
    }

    ~TraceFile() {
        if (file) std::fclose(file);
    }
};

TraceFile& trace() {
    static TraceFile* file = new TraceFile();
    return *file;
}

} // namespace

bool timeline_enabled() noexcept {
    return trace().file != nullptr;
}

std::uint64_t timeline_now_ns() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void timeline_event(const char* event, std::uint64_t key, std::uint64_t value) noexcept {
    timeline_event_at(timeline_now_ns(), event, key, value);
}

void timeline_event_at(std::uint64_t time_ns, const char* event, std::uint64_t key,
                       std::uint64_t value) noexcept {
    auto& file = trace();
    if (!file.file) return;
    std::lock_guard lock(file.mutex);
    std::fprintf(file.file, "%llu\t%s\t%llu\t%llu\n",
                 static_cast<unsigned long long>(time_ns), event ? event : "?",
                 static_cast<unsigned long long>(key),
                 static_cast<unsigned long long>(value));
}

} // namespace armigo::nvn
