// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "pacing_schedule.h"
#include <armigo/runtime/presentation_clock.h>
#include <array>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>

namespace armigo::tools {

struct PacingWatchStep { std::uint64_t offset = 0; std::size_t width = 0; };
struct PacingWatch {
    std::uint64_t address = 0;
    std::size_t width = 0;
    std::string module;
    std::vector<PacingWatchStep> chain;
    std::optional<std::uint64_t> module_base;

    std::string column() const {
        std::string name = "watch_";
        if (!module.empty()) name += module + '_';
        name += std::to_string(address) + '_' + std::to_string(width);
        for (const auto step : chain)
            name += "_p" + std::to_string(step.offset) + '_' + std::to_string(step.width);
        return name;
    }
    void bind(std::span<const runtime::PresentationModule> modules) {
        module_base.reset();
        for (const auto& loaded : modules) {
            if (loaded.name != module) continue;
            if (module_base) { module_base.reset(); return; }
            module_base = loaded.base_address;
        }
    }
    std::optional<std::uint64_t> read(const runtime::PresentationMemoryReader& memory) const {
        if (!memory || (!module.empty() && !module_base)) return std::nullopt;
        const auto base = module.empty() ? 0 : *module_base;
        if (address > UINT64_MAX - base) return std::nullopt;
        auto at = base + address;
        auto bytes_to_read = width;
        std::uint64_t value = 0;
        for (std::size_t index = 0; ; ++index) {
            std::array<std::byte, 8> bytes{};
            if (at > UINT64_MAX - bytes_to_read ||
                !memory(at, std::span(bytes).first(bytes_to_read))) return std::nullopt;
            value = 0;
            for (std::size_t i = 0; i < bytes_to_read; ++i)
                value |= std::uint64_t(std::to_integer<unsigned>(bytes[i])) << (8 * i);
            if (index == chain.size()) return value;
            const auto step = chain[index];
            if (!value || step.offset > UINT64_MAX - value) return std::nullopt;
            at = value + step.offset;
            bytes_to_read = step.width;
        }
    }
};
inline std::variant<std::vector<PacingWatch>, std::string> parse_pacing_watches(std::string_view text) {
    std::vector<PacingWatch> watches;
    const auto number = [](std::string_view text, std::uint64_t& value) {
        int base = 10;
        if (text.starts_with("0x")) { text.remove_prefix(2); base = 16; }
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, base);
        return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
    };
    while (!text.empty()) {
        const auto comma = text.find(',');
        auto entry = text.substr(0, comma);
        PacingWatch watch;
        bool root = true;
        while (!entry.empty()) {
            const auto arrow = entry.find('>');
            auto field = entry.substr(0, arrow);
            const auto colon = field.find(':');
            if (colon == std::string_view::npos) return "Watch steps require ADDRESS:WIDTH or +OFFSET:WIDTH.";
            auto address = field.substr(0, colon);
            const auto width = field.substr(colon + 1);
            if (root) {
                const auto plus = address.find('+');
                if (plus != std::string_view::npos) {
                    const auto name = address.substr(0, plus);
                    if (name.empty() || name.size() > 256 ||
                        name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string_view::npos)
                        return "Watch module names contain letters, digits, underscores, dots or hyphens.";
                    watch.module = name;
                    address.remove_prefix(plus + 1);
                }
            } else {
                if (!address.starts_with('+')) return "Pointer-chain steps require +OFFSET:WIDTH.";
                address.remove_prefix(1);
            }
            std::uint64_t offset = 0, bytes = 0;
            if (!number(address, offset) || !number(width, bytes) ||
                (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) || offset > UINT64_MAX - bytes)
                return "Watch addresses must be valid and widths must be 1, 2, 4 or 8.";
            if (root) { watch.address = offset; watch.width = bytes; }
            else {
                if (watch.chain.size() >= 16) return "Watch chains permit at most 16 pointer dereferences.";
                watch.chain.push_back({offset, std::size_t(bytes)});
            }
            root = false;
            if (arrow == std::string_view::npos) break;
            entry.remove_prefix(arrow + 1);
            if (entry.empty()) return "Watch entries must not end with a chain separator.";
        }
        if (root || watches.size() >= 64) return "Watch lists permit 1 to 64 nonempty entries.";
        watches.push_back(std::move(watch));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
        if (text.empty()) return "Watch list must not end with a comma.";
    }
    return watches;
}

// A headless-only wrapper. Both VI and native/command-stream presentation use
// its immutable opportunity grid. It never scales the guest monotonic clock.
class PacingClock final : public runtime::PresentationClock {
public:
    PacingClock(std::shared_ptr<runtime::PresentationClock> base, std::optional<PacingSchedule> schedule,
                std::unique_ptr<std::ostream> log = {}, std::vector<PacingWatch> watches = {})
        : base_(std::move(base)), schedule_(std::move(schedule)), log_(std::move(log)),
          watches_(std::move(watches)), epoch_(base_->configuration().epoch) {
        if (log_ != nullptr) {
            *log_ << "frame\telapsed_ns\tpresent_ns\traw_period_ns\tcapped_period_ns"
                    "\trequested_interval\trequested_period_ns\trefresh_index\tmissed_opportunities"
                    "\tvi_ns\tvi_target_ns\tvi_index";
            for (const auto& watch : watches_) *log_ << '\t' << watch.column();
            *log_ << '\n';
            log_->flush();
            if (!*log_) remember(failure("Could not write pacing frame log header.", runtime::LaunchError::host_failure));
        }
    }
    std::shared_ptr<runtime::GuestClock> clock() const noexcept override { return base_->clock(); }
    bool observes_presents() const noexcept override { return log_ != nullptr; }
    void bind_observation_modules(std::span<const runtime::PresentationModule> modules) override {
        const std::lock_guard lock(observation_mutex_);
        for (auto& watch : watches_) if (!watch.module.empty()) watch.bind(modules);
    }
    bool imposed_pacing() const noexcept override { return schedule_.has_value(); }
    runtime::PresentationClockConfiguration configuration() const noexcept override {
        auto config = base_->configuration();
        if (schedule_) {
            const auto sample = clock()->sample();
            if (const auto* now = std::get_if<runtime::ClockSample>(&sample))
                config.refresh_period = std::chrono::nanoseconds{schedule_->period_at((now->instant - epoch_).count())};
        }
        return config;
    }
    bool set_refresh_period(std::chrono::nanoseconds period) noexcept override {
        return !schedule_ && base_->set_refresh_period(period);
    }
    runtime::LaunchResult<runtime::PresentationSchedule> schedule(
        std::optional<std::uint64_t> last, std::int32_t interval) const override {
        if (!schedule_) return base_->schedule(last, interval);
        if (interval < 0) return failure("Negative pacing interval.");
        const auto sampled = clock()->sample();
        if (const auto* error = std::get_if<runtime::LaunchFailure>(&sampled)) return *error;
        const auto now = std::get<runtime::ClockSample>(sampled).instant;
        const std::lock_guard lock(mutex_);
        const auto elapsed = (now - epoch_).count();
        if (!extend(elapsed, last.value_or(0))) return failure("Pacing grid exceeded its bounded horizon.");
        const auto after = std::upper_bound(points_.begin(), points_.end(), elapsed);
        const auto completed = after == points_.begin() ? 0 : std::uint64_t(after - points_.begin() - 1);
        if (interval == 0) return runtime::PresentationSchedule{now, completed, true};
        const auto previous = last.value_or(completed);
        if (previous > UINT64_MAX - std::uint64_t(interval)) return failure("Pacing index overflow.");
        const auto earliest = previous + std::uint64_t(interval);
        const auto lower = std::lower_bound(points_.begin(), points_.end(), elapsed);
        const auto index = std::max(earliest, std::uint64_t(lower - points_.begin()));
        if (!extend(elapsed, index)) return failure("Pacing grid exceeded its bounded horizon.");
        const auto offset = points_[index];
        std::int64_t absolute = 0;
        if (__builtin_add_overflow(epoch_.time_since_epoch().count(), offset, &absolute))
            return failure("Pacing deadline overflow.");
        return runtime::PresentationSchedule{runtime::ClockInstant{std::chrono::nanoseconds{absolute}}, index, false};
    }
    runtime::LaunchResult<runtime::PresentationSchedule> next_refresh() const override {
        return schedule_ ? schedule(std::nullopt, 1) : base_->next_refresh();
    }
    void observe_vsync(const runtime::PresentationSchedule& target) noexcept override {
        if (!log_) return;
        // No grid, observation or I/O lock is acquired here. This tiny lock
        // only protects the VI snapshot; readers release it before any copy
        // or logging. Sample after acquiring it and after Event::signal().
        const std::lock_guard lock(vi_mutex_);
        const auto sample = clock()->sample();
        if (const auto* now = std::get_if<runtime::ClockSample>(&sample)) {
            vi_ns_ = now->instant.time_since_epoch().count();
            vi_target_ns_ = target.deadline.time_since_epoch().count();
            vi_index_ = target.refresh_index;
        }
    }
    std::optional<runtime::LaunchFailure> observation_failure() const override {
        const std::lock_guard lock(failure_mutex_);
        return observation_failure_;
    }
    runtime::LaunchStatus observe_present(std::uint64_t frame, std::int32_t interval,
                                          const runtime::PresentationMemoryReader& memory) override {
        if (!log_) return std::monostate{};
        // Serialize sampling as well as append across display producers. Slow
        // copies/streams can delay only observers, never VI or grid planning.
        const std::lock_guard lock(observation_mutex_);
        if (auto error = observation_failure()) return *error;
        try {
            return observe_present_locked(frame, interval, memory);
        } catch (...) {
            return remember(failure("Pacing frame observation threw an exception.", runtime::LaunchError::host_failure));
        }
    }
private:
    runtime::LaunchStatus observe_present_locked(std::uint64_t frame, std::int32_t interval,
                                                 const runtime::PresentationMemoryReader& memory) {
        const auto sample = clock()->sample();
        if (const auto* error = std::get_if<runtime::LaunchFailure>(&sample)) return remember(*error);
        const auto now = std::get<runtime::ClockSample>(sample).instant;
        const auto index_result = schedule(std::nullopt, 0);
        if (const auto* error = std::get_if<runtime::LaunchFailure>(&index_result)) return remember(*error);
        const auto index = std::get<runtime::PresentationSchedule>(index_result).refresh_index;
        const auto period = configuration().refresh_period.count();
        const auto ns = now.time_since_epoch().count();
        std::int64_t vi_ns, vi_target_ns;
        std::uint64_t vi_index;
        {
            const std::lock_guard lock(vi_mutex_);
            vi_ns = vi_ns_;
            vi_target_ns = vi_target_ns_;
            vi_index = vi_index_;
        }
        const auto raw = last_present_ns_ ? ns - *last_present_ns_ : 0;
        const auto expected = std::uint64_t(std::max(1, interval));
        const auto missed = last_present_index_ && index > *last_present_index_ &&
            index - *last_present_index_ > expected ? index - *last_present_index_ - expected : 0;
        *log_ << frame << '\t' << (now - epoch_).count() << '\t' << ns << '\t' << raw
             << '\t' << std::min<std::int64_t>(raw, 50'000'000) << '\t' << interval
             << '\t' << period * interval << '\t' << index << '\t' << missed
             << '\t' << vi_ns << '\t' << vi_target_ns << '\t' << vi_index;
        // Each pointer step is an independent bounded checked copy. A missing
        // module, null pointer, overflow or unreadable step is observational NA.
        for (const auto& watch : watches_) {
            *log_ << '\t';
            if (const auto value = watch.read(memory)) *log_ << *value;
            else *log_ << "NA";
        }
        *log_ << '\n';
        log_->flush();
        if (!*log_) return remember(failure("Could not write pacing frame log.", runtime::LaunchError::host_failure));
        last_present_ns_ = ns;
        last_present_index_ = index;
        return std::monostate{};
    }
    runtime::LaunchFailure remember(runtime::LaunchFailure error) {
        const std::lock_guard lock(failure_mutex_);
        if (!observation_failure_) observation_failure_ = std::move(error);
        return *observation_failure_;
    }
    static runtime::LaunchFailure failure(std::string detail,
                                         runtime::LaunchError code = runtime::LaunchError::invalid_input) {
        runtime::LaunchFailure error;
        error.stage = runtime::LaunchStage::presentation;
        error.code = code;
        error.operation = "headless.pacing";
        error.detail = std::move(detail);
        return error;
    }
    bool extend(std::int64_t elapsed, std::uint64_t index) const {
        // At most 10 million opportunities and 24 hours, with structured
        // failure instead of an unbounded allocation or overflowing deadline.
        if (elapsed > 86'400'000'000'000LL || index >= 10'000'000) return false;
        while (points_.back() <= elapsed || points_.size() <= index) {
            if (points_.size() >= 10'000'000) return false;
            auto next = points_.back() + schedule_->period_at(points_.back());
            for (const auto hitch : schedule_->hitches) {
                if (next >= hitch.at_ns && points_.back() < hitch.at_ns + hitch.duration_ns)
                    next = std::max(next, hitch.at_ns + hitch.duration_ns);
            }
            points_.push_back(next);
        }
        return true;
    }
    std::shared_ptr<runtime::PresentationClock> base_;
    std::optional<PacingSchedule> schedule_;
    std::unique_ptr<std::ostream> log_;
    std::vector<PacingWatch> watches_;
    runtime::ClockInstant epoch_;
    mutable std::mutex mutex_;
    std::mutex observation_mutex_;
    std::mutex vi_mutex_;
    mutable std::mutex failure_mutex_;
    std::optional<runtime::LaunchFailure> observation_failure_;
    mutable std::vector<std::int64_t> points_{0};
    std::optional<std::int64_t> last_present_ns_;
    std::optional<std::uint64_t> last_present_index_;
    std::int64_t vi_ns_ = 0, vi_target_ns_ = 0;
    std::uint64_t vi_index_ = 0;
};

} // namespace armigo::tools
