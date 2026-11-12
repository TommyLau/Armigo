// ============================================================================
// Armigo NVN - Conditional Report Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace armigo::nvn {

// Command-ordered knowledge of report counters, evaluated while a submission
// is encoded. A finished writer is read from guest memory. A writer that has
// not finished contributes the bytes the command stream already named, and
// only when every byte of the counter is known. Anything else is unknown, and
// a conditional on an unknown counter draws.
//
// Address lookup is constant time. The dirty list is the writers this prefix
// has not stamped yet. The page index keeps a large buffer update off every
// report that does not intersect it. Guest memory is read through `reader`:
// it stores the 8 bytes at `address` into `value` and returns false when those
// bytes are not available.
class ConditionalReports {
public:
    enum class Kind : std::uint8_t { timestamp, samples_passed, other };

    using Reader = std::function<bool(std::uint64_t address, std::uint64_t& value)>;

    void set_reader(Reader reader) { reader_ = std::move(reader); }

    // A counter report at `address`. A timestamp's counter is 0. A
    // samples-passed counter is unknown until the report's fence completes,
    // then it is read from memory; the address and the records 16 bytes on
    // either side become query operands. Any other counter kind stays unknown.
    void note_report(std::uint64_t address, Kind kind);

    // An explicit CPU-side write of `size` bytes at `destination`. `period` 0
    // reads `data` at the byte's offset; a positive period tiles `data`.
    void write(std::uint64_t destination, std::uint64_t size, const void* data, unsigned period);

    // A copy of `size` bytes. Bytes come from a report's command-known value
    // or, once that writer has finished, from memory.
    void copy(std::uint64_t source, std::uint64_t destination, std::uint64_t size);

    // A shader store of `size` bytes. The bytes are not known. A size of 0 is
    // not a store of nothing: the range is unknown (see gpu_write_unbounded).
    void gpu_write(std::uint64_t destination, std::uint64_t size);
    void gpu_write_unbounded();

    // Stamp this prefix's dirty writers with `fence`. Fence 0 means the bytes
    // are already in memory (a prefix the GPU has finished during the encode).
    void commit(std::uint64_t fence);

    // Advance the completion watermark. Writers stamped at or below it are
    // read from memory. The watermark only moves forward, and forget() leaves
    // it in place.
    void set_completed(std::uint64_t fence);

    // A submission whose GPU work failed: its report effects never landed.
    // Reported counters stamped at or below the highest failed fence stay
    // unknown, even once the watermark passes it. Atomic, like
    // set_completed: the completion thread may call it.
    void set_failed(std::uint64_t fence);

    // Known counter at `address`. Pending command-known bytes when all 8 are
    // known; a completed counter from memory; false when the value is unknown.
    bool value(std::uint64_t address, std::uint64_t& out);

    // Drop every report. Device teardown. The completion watermark stays.
    void forget();

private:
    struct Report {
        std::uint64_t address = 0;
        std::uint64_t fence = 0;
        std::uint64_t value = 0;
        std::uint64_t report_fence = 0;
        std::uint64_t gpu_epoch = 0;
        std::uint8_t known = 0;
        bool dirty = false;
        bool readable = false;
        bool reported = false;
        bool report_dirty = false;
        bool unreliable = false;
        bool gpu_domain = false;
        bool opaque_unreliable = false;
        bool opaque_later = false;
        bool conservative = false;
        Report* page_next = nullptr;
        Report* dirty_next = nullptr;
    };

    Report* ensure(std::uint64_t address);
    Report* find(std::uint64_t address);
    void page_add(std::uint64_t page);
    void query_operand(Report* report);
    void touch(Report* report);
    bool gpu_unknown(Report* report, std::uint64_t done);
    Report* containing(std::uint64_t address);
    std::uint8_t source(std::uint64_t address, unsigned count, std::uint8_t* out, bool& unreliable);
    void overwrite(Report* report, std::uint64_t destination, std::uint64_t size, const void* data,
                   unsigned period, std::uint64_t source_address, std::uint64_t done, bool opaque);
    void change(std::uint64_t destination, std::uint64_t size, const void* data, unsigned period,
                std::uint64_t source_address, bool opaque);
    bool read_u64(std::uint64_t address, std::uint64_t& value) const;
    bool read_byte(std::uint64_t address, std::uint8_t& byte) const;

    Reader reader_;
    std::unordered_map<std::uint64_t, std::unique_ptr<Report>> reports_;
    std::unordered_map<std::uint64_t, Report*> pages_;
    std::vector<std::uint64_t> page_order_;
    Report* dirty_ = nullptr;
    std::uint64_t gpu_epoch_ = 0;
    std::uint64_t gpu_fence_ = 0;
    bool gpu_dirty_ = false;
    // Published after a submission's report bytes are in memory and before a
    // signal can wake a CPU writer. The guest fence publication is later and
    // would hide that writer behind command bytes for a moment.
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<std::uint64_t> failed_{0};
};

} // namespace armigo::nvn
