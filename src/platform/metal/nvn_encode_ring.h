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

#include <dispatch/dispatch.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace armigo::platform::metal::nvn_driver {

// Single-producer / single-consumer ring of variable-size records. The
// producer appends records and publishes its write position; the consumer
// replays them in order. Positions are monotonic byte counts and the ring
// offset is the position modulo the capacity.
class EncodeRing {
public:
    static constexpr std::size_t capacity = std::size_t{16} << 20;
    // size: the whole record, a multiple of 8 bytes; argument: a small operand.
    struct Header {
        std::uint16_t op, size;
        std::uint32_t argument;
    };
    // A record that does not fit before the end of the ring is preceded by a
    // wrap record (or the bare end of the ring) that sends the consumer to
    // the start.
    static constexpr std::uint16_t wrap = 1;
    static constexpr std::size_t max_payload = 0xfff0;

    EncodeRing()
        : storage_(std::make_unique<std::byte[]>(capacity)),
          wake_(dispatch_semaphore_create(0)) {}

    // Producer: reserves a record and returns its payload. The record reaches
    // the consumer at the next publish().
    void* put(std::uint16_t op, std::uint32_t argument, std::size_t payload) {
        const std::size_t size = (sizeof(Header) + payload + 7) & ~std::size_t{7};
        auto write = local_write_;
        auto offset = write & (capacity - 1);
        if (offset + size > capacity) {
            room(write + (capacity - offset));
            if (capacity - offset >= sizeof(Header))
                *reinterpret_cast<Header*>(storage_.get() + offset) = {wrap, 0, 0};
            write += capacity - offset;
            offset = 0;
        }
        room(write + size);
        auto* header = reinterpret_cast<Header*>(storage_.get() + offset);
        *header = {op, static_cast<std::uint16_t>(size), argument};
        local_write_ = write + size;
        return header + 1;
    }

    void publish() {
        write_.store(local_write_, std::memory_order_seq_cst);
        if (sleeping_.load(std::memory_order_seq_cst)) wake();
    }

    // Producer: waits until the consumer has replayed every record put so far.
    void drain() {
        const auto target = local_write_;
        write_.store(target, std::memory_order_seq_cst);
        wake();
        for (int i = 0; read_.load(std::memory_order_acquire) < target; ++i) {
            if (i > 256) usleep(10);
            else sched_yield();
        }
    }

    // Times the producer found the ring full and waited for the consumer.
    // How often the producer found the ring full; readable from any thread.
    std::uint64_t full_waits() const { return full_waits_.load(std::memory_order_relaxed); }

    // Consumer: replays records in order until exec returns false. Between
    // records it spins for up to spin_ns while idle() is false (records come
    // in bursts and a wake-up costs the producer a system call); otherwise it
    // sleeps until the producer publishes. spin_ns is read at every gap, so
    // exec may change it between records; zero never spins.
    template <class Exec, class Idle>
    void consume(Exec&& exec, Idle&& idle, const std::uint64_t& spin_ns) {
        std::uint64_t read = 0;
        for (;;) {
            auto write = write_.load(std::memory_order_seq_cst);
            if (read == write) {
                for (const auto start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
                     !idle() && read == write &&
                     clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start < spin_ns;) {
                    for (int i = 0; i < 64 && read == write; ++i) {
#if defined(__aarch64__)
                        __asm__ volatile("yield");
#endif
                        write = write_.load(std::memory_order_seq_cst);
                    }
                }
                if (read == write) {
                    sleeping_.store(1, std::memory_order_seq_cst);
                    write = write_.load(std::memory_order_seq_cst);
                    if (read == write) dispatch_semaphore_wait(wake_, DISPATCH_TIME_FOREVER);
                    sleeping_.store(0, std::memory_order_seq_cst);
                    continue;
                }
            }
            for (unsigned count = 0; read < write;) {
                const auto offset = read & (capacity - 1);
                const auto* header = reinterpret_cast<const Header*>(storage_.get() + offset);
                if (capacity - offset < sizeof(Header) || header->op == wrap) {
                    read += capacity - offset;
                    continue;
                }
                const bool more = exec(*header, static_cast<const void*>(header + 1));
                read += header->size;
                if (!more) {
                    read_.store(read, std::memory_order_release);
                    return;
                }
                if (!(++count & 63)) read_.store(read, std::memory_order_release);
            }
            read_.store(read, std::memory_order_release);
        }
    }

private:
    void wake() {
        if (sleeping_.exchange(0, std::memory_order_seq_cst)) dispatch_semaphore_signal(wake_);
    }

    // Waits until the consumer has freed the ring up to end.
    void room(std::uint64_t end) {
        if (end - cached_read_ <= capacity) return;
        if (end - (cached_read_ = read_.load(std::memory_order_acquire)) <= capacity) return;
        full_waits_.fetch_add(1, std::memory_order_relaxed);
        write_.store(local_write_, std::memory_order_seq_cst);
        wake();
        for (int i = 0; end - (cached_read_ = read_.load(std::memory_order_acquire)) > capacity; ++i) {
            if (i > 64) usleep(20);
            else sched_yield();
        }
    }

    std::unique_ptr<std::byte[]> storage_;
    dispatch_semaphore_t wake_;
    // Producer-only: its write position and its last view of the read position.
    std::uint64_t local_write_ = 0, cached_read_ = 0;
    std::atomic<std::uint64_t> full_waits_{0};
    // Each shared position on its own line: the two threads write them from
    // different cores.
    alignas(128) std::atomic<std::uint64_t> write_{0};
    // Published every 64 records and when the consumer runs out of records.
    alignas(128) std::atomic<std::uint64_t> read_{0};
    alignas(128) std::atomic<int> sleeping_{0};
};

} // namespace armigo::platform::metal::nvn_driver
