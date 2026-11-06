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

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace armigo::tools {

// Segment boundaries and phase are elapsed monotonic time, never presented
// frame counts. Even frame alternation keeps its phase when rendering is late.
struct PacingSegment {
    enum class Kind { fixed, alternate, frame_alternate, ramp } kind = Kind::fixed;
    std::int64_t begin_ns = 0, duration_ns = 0, hold_ns = 0;
    double first_hz = 0, second_hz = 0;
};
struct PacingHitch { std::int64_t at_ns = 0, duration_ns = 0; };
struct PacingSchedule {
    std::vector<PacingSegment> segments;
    std::vector<PacingHitch> hitches;
    double ceiling_hz = 0;

    static std::int64_t period(double hz) { return std::llround(1e9 / hz); }
    std::int64_t period_at(std::int64_t elapsed) const {
        const auto found = std::upper_bound(segments.begin(), segments.end(), elapsed,
            [](auto time, const auto& segment) { return time < segment.begin_ns; });
        const auto& segment = found == segments.begin() ? segments.front() : *std::prev(found);
        const auto local = std::clamp(elapsed - segment.begin_ns, std::int64_t{0}, segment.duration_ns);
        double hz = segment.first_hz;
        if (elapsed >= segments.back().begin_ns + segments.back().duration_ns) {
            hz = segments.back().second_hz;
        } else if (segment.kind == PacingSegment::Kind::ramp) {
            hz += (segment.second_hz - hz) * double(local) / double(segment.duration_ns);
        } else if (segment.kind == PacingSegment::Kind::alternate) {
            if ((local / segment.hold_ns) % 2) hz = segment.second_hz;
        } else if (segment.kind == PacingSegment::Kind::frame_alternate) {
            const auto first = period(std::min(segment.first_hz, ceiling_hz));
            const auto second = period(std::min(segment.second_hz, ceiling_hz));
            if (local % (first + second) >= first) hz = segment.second_hz;
        }
        return period(std::min(hz, ceiling_hz));
    }
};

inline std::variant<PacingSchedule, std::string> parse_pacing_schedule(std::istream& input, double ceiling) {
    if (!std::isfinite(ceiling) || ceiling < 1 || ceiling > 1000)
        return "Pacing ceiling must be between 1 and 1000 Hz.";
    PacingSchedule out;
    out.ceiling_hz = ceiling;
    std::string line;
    std::int64_t end = 0;
    unsigned number = 0;
    constexpr double maximum_ms = 86'400'000;
    while (std::getline(input, line)) {
        ++number;
        if (const auto comment = line.find('#'); comment != std::string::npos) line.resize(comment);
        std::istringstream row(line);
        std::string kind, extra;
        if (!(row >> kind)) continue;
        double duration = 0, first = 0, second = 0, hold = 0;
        auto invalid = [&] { return "Invalid pacing schedule line " + std::to_string(number) + "."; };
        auto ms = [](double value) { return std::isfinite(value) && value > 0 && value <= maximum_ms; };
        auto hz = [](double value) { return std::isfinite(value) && value >= 1 && value <= 1000; };
        if (kind == "hitch") {
            double at = 0;
            if (!(row >> at >> duration) || !std::isfinite(at) || at < 0 || at > maximum_ms ||
                !ms(duration) || row >> extra) return invalid();
            const auto duration_ns = std::llround(duration * 1e6);
            if (duration_ns < 1) return invalid();
            out.hitches.push_back({std::llround(at * 1e6), duration_ns});
            continue;
        }
        PacingSegment segment;
        if (!(row >> duration >> first) || !ms(duration) || !hz(first)) return invalid();
        second = first;
        if (kind == "ramp" || kind == "alternate" || kind == "alternate-frame") {
            if (!(row >> second) || !hz(second)) return invalid();
            if (kind == "ramp") segment.kind = PacingSegment::Kind::ramp;
            if (kind == "alternate-frame") segment.kind = PacingSegment::Kind::frame_alternate;
            if (kind == "alternate") {
                segment.kind = PacingSegment::Kind::alternate;
                if (!(row >> hold) || !ms(hold)) return invalid();
            }
        } else if (kind != "fixed") return invalid();
        if (row >> extra) return invalid();
        segment.begin_ns = end;
        segment.duration_ns = std::llround(duration * 1e6);
        segment.hold_ns = std::llround(hold * 1e6);
        if (segment.duration_ns < 1 || (kind == "alternate" && segment.hold_ns < 1)) return invalid();
        segment.first_hz = first;
        segment.second_hz = second;
        end += segment.duration_ns;
        if (end > std::int64_t(maximum_ms * 1e6)) return invalid();
        out.segments.push_back(segment);
    }
    if (input.bad()) return "Could not read pacing schedule.";
    if (out.segments.empty()) return "Pacing schedule requires a rate segment.";
    std::sort(out.hitches.begin(), out.hitches.end(), [](auto a, auto b) { return a.at_ns < b.at_ns; });
    for (std::size_t i = 1; i < out.hitches.size(); ++i)
        if (out.hitches[i].at_ns < out.hitches[i-1].at_ns + out.hitches[i-1].duration_ns)
            return "Pacing hitches must not overlap.";
    return out;
}

} // namespace armigo::tools
