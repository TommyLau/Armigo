// ============================================================================
// Armigo NVN - Conditional Report Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "conditional_reports.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace armigo::nvn {
namespace {
std::uint8_t byte_at(std::uint64_t value, unsigned index) {
    return static_cast<std::uint8_t>(value >> (8 * index));
}
void set_byte(std::uint64_t& value, unsigned index, std::uint8_t byte) {
    const auto shift = 8 * index;
    value = (value & ~(std::uint64_t{0xff} << shift)) | (std::uint64_t{byte} << shift);
}
}

ConditionalReports::Report* ConditionalReports::find(std::uint64_t address) {
    const auto found = reports_.find(address);
    return found == reports_.end() ? nullptr : found->second.get();
}

void ConditionalReports::page_add(std::uint64_t page) {
    const auto at = std::lower_bound(page_order_.begin(), page_order_.end(), page);
    if (at != page_order_.end() && *at == page) return;
    page_order_.insert(at, page);
}

ConditionalReports::Report* ConditionalReports::ensure(std::uint64_t address) {
    if (Report* existing = find(address)) return existing;
    auto created = std::make_unique<Report>();
    Report* report = created.get();
    report->address = address;
    report->readable = true;
    report->gpu_epoch = gpu_epoch_;
    const auto page = address >> 12;
    auto& head = pages_[page];
    report->page_next = head;
    if (!head) page_add(page);
    head = report;
    reports_.emplace(address, std::move(created));
    return report;
}

void ConditionalReports::query_operand(Report* report) {
    if (report->gpu_domain) return;
    report->gpu_domain = true;
    // An earlier shader store may have raced this address before a comparison
    // used it. Those bytes are not a counter.
    if (report->opaque_later) {
        report->known = 0;
        report->opaque_later = false;
    }
    if (report->opaque_unreliable) report->unreliable = true;
}

void ConditionalReports::touch(Report* report) {
    report->fence = 0;
    report->gpu_epoch = gpu_epoch_;
    if (!report->dirty) {
        report->dirty = true;
        report->dirty_next = dirty_;
        dirty_ = report;
    }
}

void ConditionalReports::note_report(std::uint64_t address, Kind kind) {
    Report* report = ensure(address);
    // Samples-passed becomes a memory read once its fence completes. Other
    // counter kinds have no implemented value and stay unknown. The bit is
    // sticky: a later report on the same address does not clear it.
    if (kind == Kind::other) report->conservative = true;
    // A comparison reads this record and the one 16 bytes away. Either side
    // can be the query, so the neighbour is tracked before a later buffer
    // writer. An entry that already exists keeps its own pending bytes.
    if (kind == Kind::samples_passed) {
        query_operand(report);
        if (address >= 16) query_operand(ensure(address - 16));
        query_operand(ensure(address + 16));
    }
    report->value = 0;
    report->known = kind == Kind::timestamp ? 255 : 0;
    report->readable = kind != Kind::other;
    report->reported = true;
    report->report_dirty = true;
    report->report_fence = 0;
    report->unreliable = false;
    report->opaque_unreliable = false;
    report->opaque_later = false;
    touch(report);
}

bool ConditionalReports::gpu_unknown(Report* report, std::uint64_t done) {
    if (!report->gpu_domain || report->gpu_epoch == gpu_epoch_) return false;
    // An unbounded shader store may race a counter that has already landed.
    // A fresh report or a full explicit write can name the counter again.
    // A neighbour that was never reported has no deferred CPU write, so it
    // becomes readable once that store's fence completes.
    if (report->reported || gpu_dirty_ || gpu_fence_ > done) return true;
    report->gpu_epoch = gpu_epoch_;
    return false;
}

bool ConditionalReports::read_u64(std::uint64_t address, std::uint64_t& value) const {
    if (!reader_) return false;
    return reader_(address, value);
}

bool ConditionalReports::read_byte(std::uint64_t address, std::uint8_t& byte) const {
    std::uint64_t word = 0;
    if (!read_u64(address, word)) return false;
    byte = static_cast<std::uint8_t>(word);
    return true;
}

bool ConditionalReports::value(std::uint64_t address, std::uint64_t& out) {
    Report* report = find(address);
    const auto done = completed_.load(std::memory_order_acquire);
    if (report) {
        if (report->conservative) return false;
        // A report of a failed submission never reached memory.
        if (report->reported && report->report_fence &&
            report->report_fence <= failed_.load(std::memory_order_acquire)) return false;
        if (gpu_unknown(report, done)) return false;
        if (report->dirty || report->fence > done) {
            out = report->value;
            return report->known == 255;
        }
        if (!report->readable || report->unreliable) return false;
    }
    if (!read_u64(address, out)) return false;
    return true;
}

ConditionalReports::Report* ConditionalReports::containing(std::uint64_t address) {
    // A counter need not start on the same page as the byte. The previous
    // seven bytes cover an unaligned counter that crosses a page.
    const auto first = (address > 7 ? address - 7 : 0) >> 12;
    const auto last = address >> 12;
    for (auto page = first; page <= last; ++page) {
        const auto found = pages_.find(page);
        if (found == pages_.end()) continue;
        for (Report* report = found->second; report; report = report->page_next)
            if (address >= report->address && address - report->address < 8) return report;
    }
    return nullptr;
}

std::uint8_t ConditionalReports::source(std::uint64_t address, unsigned count, std::uint8_t* out,
                                        bool& unreliable) {
    std::uint8_t known = 0;
    const auto done = completed_.load(std::memory_order_acquire);
    for (unsigned index = 0; index != count; ++index) {
        const auto at = address + index;
        Report* report = containing(at);
        if (report) {
            const auto offset = static_cast<unsigned>(at - report->address);
            if (report->unreliable || report->opaque_unreliable ||
                (report->reported && (report->report_dirty || report->report_fence > done)))
                unreliable = true;
            if (report->opaque_later) continue;
            // A completed timestamp is not a copy source after an unbounded
            // shader store, even when value() may still read it: the store
            // did not mark the address a query operand, so value() does not
            // treat the epoch change as unknown.
            if ((report->reported && report->gpu_epoch != gpu_epoch_) || gpu_unknown(report, done)) {
                unreliable = true;
                continue;
            }
            if (!report->dirty && report->fence <= done) {
                if (report->readable && !report->unreliable && !report->opaque_unreliable) {
                    std::uint8_t byte = 0;
                    if (read_byte(at, byte)) {
                        out[index] = byte;
                        known = static_cast<std::uint8_t>(known | (1u << index));
                    }
                }
            } else if ((report->known >> offset) & 1) {
                out[index] = byte_at(report->value, offset);
                known = static_cast<std::uint8_t>(known | (1u << index));
            }
            continue;
        }
        std::uint8_t byte = 0;
        if (read_byte(at, byte)) {
            out[index] = byte;
            known = static_cast<std::uint8_t>(known | (1u << index));
        }
    }
    return known;
}

void ConditionalReports::overwrite(Report* report, std::uint64_t destination, std::uint64_t size,
                                   const void* data, unsigned period, std::uint64_t source_address,
                                   std::uint64_t done, bool opaque) {
    const auto lo = std::max(destination, report->address);
    const auto hi = std::min(destination + size, report->address + 8);
    if (hi <= lo) return;
    if (opaque && !report->gpu_domain) {
        // Ordinary timestamp forwarding stays as it is. The store is remembered
        // in case a later comparison uses the address.
        if (report->reported) report->opaque_unreliable = report->opaque_later = true;
        return;
    }
    const bool report_pending = report->reported && (report->report_dirty || report->report_fence > done);
    if (report->opaque_later) {
        report->known = 0;
        report->opaque_later = false;
    }
    if (report->opaque_unreliable) report->unreliable = true;
    if ((report->reported && report->gpu_epoch != gpu_epoch_) || gpu_unknown(report, done)) {
        report->known = 0;
        if (report->reported) report->unreliable = true;
    }
    if (opaque) {
        if (report_pending) report->unreliable = true;
        report->known = 0;
        touch(report);
        return;
    }
    // The previous writer has landed. Bytes this write does not replace stay
    // at the current memory value, including a CPU store after that writer.
    if (!report->dirty && report->fence <= done) {
        if (report->readable && !report->unreliable && !gpu_unknown(report, done)) {
            std::uint64_t memory = 0;
            if (read_u64(report->address, memory)) {
                report->value = memory;
                report->known = 255;
            } else {
                report->known = 0;
            }
        } else {
            report->known = 0;
        }
    }
    bool unreliable = report_pending;
    const auto at = static_cast<unsigned>(lo - report->address);
    const auto bytes = static_cast<unsigned>(hi - lo);
    if (data) {
        const auto* bytes_in = static_cast<const std::uint8_t*>(data);
        for (unsigned index = 0; index != bytes; ++index) {
            const auto offset = lo + index - destination;
            set_byte(report->value, at + index, bytes_in[period ? offset % period : offset]);
            report->known = static_cast<std::uint8_t>(report->known | (1u << (at + index)));
        }
    } else {
        std::uint8_t copied[8]{};
        const auto copied_known = source(source_address + (lo - destination), bytes, copied, unreliable);
        const auto mask = static_cast<std::uint8_t>(((1u << bytes) - 1u) << at);
        for (unsigned index = 0; index != bytes; ++index)
            if ((copied_known >> index) & 1) set_byte(report->value, at + index, copied[index]);
        report->known = static_cast<std::uint8_t>((report->known & static_cast<std::uint8_t>(~mask)) |
                                                   (copied_known << at));
    }
    if (report->known == 255) report->readable = true;
    // A report still lands on the CPU when its fence completes. A later GPU
    // overwrite can therefore leave bytes that must not be taken from that
    // landing: the command-known value stays usable while the overwrite is
    // pending, and completed memory is not used afterwards. A full explicit
    // write that does not race the report names the counter again.
    if (unreliable) report->unreliable = true;
    else if (at == 0 && bytes == 8) {
        report->unreliable = false;
        report->opaque_unreliable = false;
        report->opaque_later = false;
    }
    touch(report);
}

void ConditionalReports::change(std::uint64_t destination, std::uint64_t size, const void* data,
                                unsigned period, std::uint64_t source_address, bool opaque) {
    if (reports_.empty() || !size) return;
    const auto done = completed_.load(std::memory_order_acquire);
    const auto last_byte = size - 1 > std::numeric_limits<std::uint64_t>::max() - destination
                               ? std::numeric_limits<std::uint64_t>::max()
                               : destination + size - 1;
    const auto first = (destination > 7 ? destination - 7 : 0) >> 12;
    const auto last = last_byte >> 12;
    const auto visit = [&](std::uint64_t page) {
        const auto found = pages_.find(page);
        if (found == pages_.end()) return;
        for (Report* report = found->second; report; report = report->page_next)
            overwrite(report, destination, size, data, period, source_address, done, opaque);
    };
    // Large storage bindings are common. Search the recorded pages and visit
    // only the intersection; walking every report made unrelated stores scan
    // the whole table.
    if (last - first >= 8) {
        for (auto it = std::lower_bound(page_order_.begin(), page_order_.end(), first);
             it != page_order_.end() && *it <= last; ++it)
            visit(*it);
    } else {
        for (auto page = first; page <= last; ++page) visit(page);
    }
}

void ConditionalReports::write(std::uint64_t destination, std::uint64_t size, const void* data,
                               unsigned period) {
    if (!data) return;
    change(destination, size, data, period, 0, false);
}

void ConditionalReports::copy(std::uint64_t source_address, std::uint64_t destination,
                              std::uint64_t size) {
    if (source_address != destination) change(destination, size, nullptr, 0, source_address, false);
}

void ConditionalReports::gpu_write(std::uint64_t destination, std::uint64_t size) {
    if (!size) {
        ++gpu_epoch_;
        gpu_dirty_ = true;
        return;
    }
    change(destination, size, nullptr, 0, 0, true);
}

void ConditionalReports::gpu_write_unbounded() { gpu_write(0, 0); }

void ConditionalReports::commit(std::uint64_t fence) {
    for (Report* report = dirty_; report;) {
        Report* next = report->dirty_next;
        report->fence = fence;
        report->dirty = false;
        report->dirty_next = nullptr;
        if (report->report_dirty) {
            report->report_fence = fence;
            report->report_dirty = false;
        }
        report = next;
    }
    dirty_ = nullptr;
    if (gpu_dirty_) {
        gpu_fence_ = fence;
        gpu_dirty_ = false;
    }
}

void ConditionalReports::set_completed(std::uint64_t fence) {
    auto seen = completed_.load(std::memory_order_relaxed);
    while (seen < fence && !completed_.compare_exchange_weak(
               seen, fence, std::memory_order_release, std::memory_order_relaxed)) {
    }
}

void ConditionalReports::set_failed(std::uint64_t fence) {
    auto seen = failed_.load(std::memory_order_relaxed);
    while (seen < fence && !failed_.compare_exchange_weak(
               seen, fence, std::memory_order_release, std::memory_order_relaxed)) {
    }
}

void ConditionalReports::forget() {
    dirty_ = nullptr;
    gpu_epoch_ = 0;
    gpu_fence_ = 0;
    gpu_dirty_ = false;
    pages_.clear();
    reports_.clear();
    page_order_.clear();
}

} // namespace armigo::nvn
