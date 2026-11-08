// ============================================================================
// Armigo Diagnostics - Host Diagnostics
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

namespace armigo::diag {
namespace {

enum class Channel { level, frame, opt_in };

struct Pending {
    std::string text;
    Level level = Level::info;
    Channel channel = Channel::level;
};

// exit() destroys the calling thread's thread_locals before it runs atexit.
// The native shader package emits its hit line from atexit on the main thread,
// so that write must not touch the thread_local line buffer after its
// destructor has freed it. Worker threads do not retire the main buffer.
std::atomic<bool> main_pending_retired{false};
std::atomic<std::uint64_t> exit_pending_appends{0};
const std::thread::id process_main_thread = std::this_thread::get_id();

Pending& exit_pending() noexcept {
    static Pending* pending = new Pending;
    return *pending;
}

struct PendingSlot {
    Pending* pending = new Pending;
    bool main = false;

    PendingSlot() : main(std::this_thread::get_id() == process_main_thread) {}

    ~PendingSlot() {
        delete pending;
        pending = nullptr;
        if (main) main_pending_retired.store(true, std::memory_order_release);
    }

    PendingSlot(const PendingSlot&) = delete;
    PendingSlot& operator=(const PendingSlot&) = delete;
};

Pending& pending_state() noexcept {
    if (main_pending_retired.load(std::memory_order_acquire) &&
        std::this_thread::get_id() == process_main_thread) {
        auto& pending = exit_pending();
        exit_pending_appends.fetch_add(1, std::memory_order_relaxed);
        return pending;
    }
    thread_local PendingSlot slot;
    return *slot.pending;
}

struct FileState {
    std::mutex mutex;
    std::FILE* file = nullptr;
    std::string path;
    std::uint64_t bytes = 0;
    bool capped = false;
    TestingSink sink = nullptr;
    void* sink_context = nullptr;
};

// Leaked for the process lifetime. Shutdown still logs after other static
// objects are destroyed, and locking a destroyed mutex aborts the process.
FileState& files() noexcept {
    static FileState* state = new FileState;
    return *state;
}

std::atomic<int> env_level_set{0};
std::atomic<int> env_stats_set{0};

constexpr std::uint64_t last_run_cap = 1024ull * 1024ull;
constexpr std::string_view cap_notice = "Armigo diagnostics: last-run.log stopped at 1 MB.\n";

std::string_view trim(std::string_view text) noexcept {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
    return text;
}

bool same_text(std::string_view text, std::string_view expected) noexcept {
    if (text.size() != expected.size()) return false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto left = static_cast<unsigned char>(text[i]);
        const auto right = static_cast<unsigned char>(expected[i]);
        if (std::tolower(left) != std::tolower(right)) return false;
    }
    return true;
}

void write_stderr(std::string_view line) noexcept {
    if (line.empty()) return;
    std::FILE* stream = stderr;
    flockfile(stream);
    std::fwrite(line.data(), 1, line.size(), stream);
    std::fflush(stream);
    funlockfile(stream);
}

void write_file(FileState& state, std::string_view line) noexcept {
    if (!state.file || state.capped || line.empty()) return;
    // Leave room for the cap notice so a full log still explains why it stopped.
    const auto content_cap = last_run_cap - cap_notice.size();
    if (state.bytes >= content_cap || state.bytes + line.size() > content_cap) {
        state.capped = true;
        if (state.bytes + cap_notice.size() <= last_run_cap) {
            std::fwrite(cap_notice.data(), 1, cap_notice.size(), state.file);
            state.bytes += cap_notice.size();
        }
        std::fflush(state.file);
        return;
    }
    std::fwrite(line.data(), 1, line.size(), state.file);
    state.bytes += line.size();
    std::fflush(state.file);
}

void emit_line(Level level, Channel channel, std::string_view line) noexcept {
    auto& state = files();
    std::lock_guard lock(state.mutex);
    if (state.sink) state.sink(level, line, state.sink_context);
    else write_stderr(line);
    if (channel == Channel::level && (level == Level::error || level == Level::warning))
        write_file(state, line);
}

void flush_complete(Pending& pending) noexcept {
    for (;;) {
        const auto end = pending.text.find('\n');
        if (end == std::string::npos) return;
        const std::string line = pending.text.substr(0, end + 1);
        pending.text.erase(0, end + 1);
        emit_line(pending.level, pending.channel, line);
    }
}

void append_text(Channel channel, Level level, std::string_view text) {
    if (text.empty()) return;
    auto& pending = pending_state();
    if (!pending.text.empty() && (pending.channel != channel || pending.level != level)) {
        pending.text.push_back('\n');
        flush_complete(pending);
    }
    pending.channel = channel;
    pending.level = level;
    pending.text.append(text);
    flush_complete(pending);
}

void format_into(Channel channel, Level level, const char* format, va_list args) {
    char stack[2048];
    va_list copy;
    va_copy(copy, args);
    const int needed = std::vsnprintf(stack, sizeof stack, format, copy);
    va_end(copy);
    if (needed < 0) return;
    if (static_cast<std::size_t>(needed) < sizeof stack) {
        append_text(channel, level, std::string_view(stack, static_cast<std::size_t>(needed)));
        return;
    }
    std::string heap(static_cast<std::size_t>(needed) + 1, '\0');
    std::vsnprintf(heap.data(), heap.size(), format, args);
    append_text(channel, level, std::string_view(heap.data(), static_cast<std::size_t>(needed)));
}

void refresh_environment() noexcept {
    if (const char* value = std::getenv("ARMIGO_LOG")) {
        if (const auto parsed = parse_level(value)) {
            detail::minimum_level.store(static_cast<int>(*parsed), std::memory_order_relaxed);
            env_level_set.store(1, std::memory_order_relaxed);
        } else {
            env_level_set.store(0, std::memory_order_relaxed);
        }
    } else {
        env_level_set.store(0, std::memory_order_relaxed);
    }
    if (const char* value = std::getenv("ARMIGO_FRAME_STATS")) {
        if (const auto parsed = parse_frame_stats(value)) {
            detail::frame_stats.store(*parsed ? 1 : 0, std::memory_order_relaxed);
            env_stats_set.store(1, std::memory_order_relaxed);
        } else {
            env_stats_set.store(0, std::memory_order_relaxed);
        }
    } else {
        env_stats_set.store(0, std::memory_order_relaxed);
    }
}

struct Startup {
    Startup() noexcept { use_development_defaults(); }
};

Startup startup;

} // namespace

std::optional<Level> parse_level(std::string_view text) noexcept {
    text = trim(text);
    if (same_text(text, "error")) return Level::error;
    if (same_text(text, "warning")) return Level::warning;
    if (same_text(text, "info")) return Level::info;
    if (same_text(text, "trace")) return Level::trace;
    return std::nullopt;
}

std::optional<bool> parse_frame_stats(std::string_view text) noexcept {
    text = trim(text);
    if (same_text(text, "1") || same_text(text, "true") || same_text(text, "on") || same_text(text, "yes"))
        return true;
    if (same_text(text, "0") || same_text(text, "false") || same_text(text, "off") || same_text(text, "no"))
        return false;
    return std::nullopt;
}

void use_development_defaults() noexcept {
    refresh_environment();
    if (env_level_set.load(std::memory_order_relaxed) == 0)
        detail::minimum_level.store(static_cast<int>(Level::info), std::memory_order_relaxed);
    if (env_stats_set.load(std::memory_order_relaxed) == 0)
        detail::frame_stats.store(1, std::memory_order_relaxed);
}

void use_player_defaults() noexcept {
    refresh_environment();
    if (env_level_set.load(std::memory_order_relaxed) == 0)
        detail::minimum_level.store(static_cast<int>(Level::warning), std::memory_order_relaxed);
    if (env_stats_set.load(std::memory_order_relaxed) == 0)
        detail::frame_stats.store(0, std::memory_order_relaxed);
}

bool open_last_run_log(const char* path) noexcept {
    if (!path || !*path) return false;
    try {
        auto& state = files();
        std::lock_guard lock(state.mutex);
        if (state.file && state.path == path) return true;
        if (state.file) {
            std::fclose(state.file);
            state.file = nullptr;
            state.path.clear();
        }
        const std::filesystem::path destination(path);
        if (destination.has_parent_path()) {
            std::error_code error;
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error) return false;
        }
        state.file = std::fopen(path, "wb");
        if (!state.file) return false;
        state.path = path;
        state.bytes = 0;
        state.capped = false;
        return true;
    } catch (...) {
        return false;
    }
}

void write(Level level, const char* format, ...) noexcept {
    if (!format || !enabled(level)) return;
    va_list args;
    va_start(args, format);
    try {
        format_into(Channel::level, level, format, args);
    } catch (...) {
    }
    va_end(args);
}

void error(const char* format, ...) noexcept {
    if (!format || !enabled(Level::error)) return;
    va_list args;
    va_start(args, format);
    try {
        format_into(Channel::level, Level::error, format, args);
    } catch (...) {
    }
    va_end(args);
}

void warning(const char* format, ...) noexcept {
    if (!format || !enabled(Level::warning)) return;
    va_list args;
    va_start(args, format);
    try {
        format_into(Channel::level, Level::warning, format, args);
    } catch (...) {
    }
    va_end(args);
}

void info(const char* format, ...) noexcept {
    if (!format || !enabled(Level::info)) return;
    va_list args;
    va_start(args, format);
    try {
        format_into(Channel::level, Level::info, format, args);
    } catch (...) {
    }
    va_end(args);
}

void trace(const char* format, ...) noexcept {
    if (!format || !enabled(Level::trace)) return;
    va_list args;
    va_start(args, format);
    try {
        format_into(Channel::level, Level::trace, format, args);
    } catch (...) {
    }
    va_end(args);
}

void frame(const char* format, ...) noexcept {
    if (!format || !frame_stats_enabled()) return;
    va_list args;
    va_start(args, format);
    try {
        format_into(Channel::frame, Level::trace, format, args);
    } catch (...) {
    }
    va_end(args);
}

void opt_in(const char* format, ...) noexcept {
    if (!format) return;
    va_list args;
    va_start(args, format);
    try {
        format_into(Channel::opt_in, Level::trace, format, args);
    } catch (...) {
    }
    va_end(args);
}

void newline() noexcept {
    try {
        auto& pending = pending_state();
        if (pending.text.empty()) return;
        pending.text.push_back('\n');
        flush_complete(pending);
    } catch (...) {
    }
}

void testing_reset() noexcept {
    {
        auto& state = files();
        std::lock_guard lock(state.mutex);
        if (state.file) {
            std::fclose(state.file);
            state.file = nullptr;
        }
        state.path.clear();
        state.bytes = 0;
        state.capped = false;
        state.sink = nullptr;
        state.sink_context = nullptr;
    }
    pending_state() = {};
    use_development_defaults();
}

void testing_set_level(Level level) noexcept {
    detail::minimum_level.store(static_cast<int>(level), std::memory_order_relaxed);
}

void testing_set_frame_stats(bool enabled) noexcept {
    detail::frame_stats.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

void testing_set_sink(TestingSink sink, void* context) noexcept {
    auto& state = files();
    std::lock_guard lock(state.mutex);
    state.sink = sink;
    state.sink_context = context;
}

void testing_close_log() noexcept {
    auto& state = files();
    std::lock_guard lock(state.mutex);
    if (state.file) {
        std::fclose(state.file);
        state.file = nullptr;
    }
    state.path.clear();
    state.bytes = 0;
    state.capped = false;
}

std::uint64_t testing_last_run_bytes() noexcept {
    auto& state = files();
    std::lock_guard lock(state.mutex);
    return state.bytes;
}

std::uint64_t testing_exit_pending_appends() noexcept {
    return exit_pending_appends.load(std::memory_order_relaxed);
}

} // namespace armigo::diag
