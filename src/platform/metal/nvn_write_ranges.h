// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace armigo::platform::metal::nvn_driver {

// Ordinary nonempty byte ranges have an exact overlap result. Writer selection
// and ordering still use their original checks. Empty ranges and overflow keep
// a conservative negative filter without changing legacy point-overlap rules.
class WriteRanges {
public:
    void clear() {
        ranges_.clear();
        uncertain_ = false;
        precise_ = true;
    }

    void add(std::uint64_t begin, std::uint64_t size) {
        if (uncertain_) return;
        if (!size) precise_ = false;
        size = std::max<std::uint64_t>(size, 1);
        if (size > std::numeric_limits<std::uint64_t>::max() - begin) {
            disable();
            return;
        }
        auto end = begin + size;
        if (ranges_.empty() || ranges_.back().end < begin) {
            ranges_.push_back({begin, end});
        } else {
            auto first = std::lower_bound(ranges_.begin(), ranges_.end(), begin,
                [](const Range& range, std::uint64_t value) { return range.end < value; });
            auto last = first;
            while (last != ranges_.end() && last->begin <= end) {
                begin = std::min(begin, last->begin);
                end = std::max(end, last->end);
                ++last;
            }
            if (first == last) {
                ranges_.insert(first, {begin, end});
            } else {
                *first = {begin, end};
                ranges_.erase(first + 1, last);
            }
        }
        // Losing precision only restores the original scan. Guest command
        // counts cannot make this optional index grow without a bound.
        if (ranges_.size() > max_ranges) disable();
    }

    [[nodiscard]] std::optional<bool> intersection(std::uint64_t begin, std::uint64_t size) const {
        if (uncertain_) return std::nullopt;
        if (ranges_.empty()) return false;
        const bool empty = !size;
        size = std::max<std::uint64_t>(size, 1);
        if (size > std::numeric_limits<std::uint64_t>::max() - begin) return std::nullopt;
        const auto end = begin + size;
        if (end <= ranges_.front().begin || ranges_.back().end <= begin) return false;
        const auto found = std::lower_bound(ranges_.begin(), ranges_.end(), begin,
            [](const Range& range, std::uint64_t value) { return range.end <= value; });
        const bool overlap = found != ranges_.end() && found->begin < end;
        if (overlap && (empty || !precise_)) return std::nullopt;
        return overlap;
    }

    [[nodiscard]] bool may_intersect(std::uint64_t begin, std::uint64_t size) const {
        return intersection(begin, size).value_or(true);
    }

private:
    struct Range { std::uint64_t begin, end; };
    static constexpr std::size_t max_ranges = 4096;
    std::vector<Range> ranges_;
    bool uncertain_ = false;
    bool precise_ = true;

    void disable() {
        ranges_.clear();
        uncertain_ = true;
    }
};

} // namespace armigo::platform::metal::nvn_driver
