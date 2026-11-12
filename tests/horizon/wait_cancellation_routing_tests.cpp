// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/startup_queries.h>
#include <armigo/horizon/event.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

namespace {
using namespace armigo::horizon;
using namespace std::chrono_literals;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "Check failed at %d: %s\n", __LINE__, #condition); std::abort(); } } while (false)
constexpr std::uint64_t address = 0x20000000;
struct Words {
    std::array<std::atomic<std::uint32_t>, 2> values{};
    bool mutex_readable = true;
    bool mutex_writable = true;
    static bool read(void* raw, std::uint64_t at, std::span<std::byte, 4> bytes) noexcept {
        if (at < address || (at & 3) || at - address >= 8) return false;
        if (at == address && !static_cast<Words*>(raw)->mutex_readable) return false;
        const auto value = static_cast<Words*>(raw)->values[(at - address) / 4].load();
        for (unsigned i = 0; i < 4; ++i) bytes[i] = std::byte(value >> (i * 8));
        return true;
    }
    static bool write(void* raw, std::uint64_t at, std::span<const std::byte, 4> bytes) noexcept {
        if (at < address || (at & 3) || at - address >= 8) return false;
        if (at == address && !static_cast<Words*>(raw)->mutex_writable) return false;
        std::uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i) value |= std::to_integer<std::uint32_t>(bytes[i]) << (i * 8);
        static_cast<Words*>(raw)->values[(at - address) / 4].store(value);
        return true;
    }
    static ProcessWideKey::ExchangeResult exchange(void* raw, std::uint64_t at,
                                                   std::uint32_t& expected,
                                                   std::uint32_t desired) noexcept {
        if (at < address || (at & 3) || at - address >= 8)
            return ProcessWideKey::ExchangeResult::invalid_address;
        return static_cast<Words*>(raw)->values[(at - address) / 4].compare_exchange_strong(expected, desired)
            ? ProcessWideKey::ExchangeResult::exchanged : ProcessWideKey::ExchangeResult::mismatch;
    }
};
std::shared_ptr<ProcessRecord> process() {
    ProcessConfiguration config;
    config.heap_address = 0x1000000000;
    config.alias_address = 0x2000000000;
    config.stack_address = 0x6000000000;
    config.allowed_core_mask = 1;
    const std::array blocks{MemoryBlock{{address, 4096}, MemoryType::normal, 0, 3}};
    auto result = ProcessRecord::create(config, {0, 4096, 0, 0, 0x100000, 4096}, blocks);
    CHECK(result.ok());
    return result.process;
}
template<class Predicate> void settle(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!predicate()) {
        CHECK(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    }
}
void routed_waits() {
    auto owner = process();
    HandleTable handles(4);
    auto thread = std::make_shared<ThreadRecord>(owner, 1, 32);
    const auto owner_handle = handles.insert(thread);
    CHECK(owner_handle.has_value());
    auto arbitration = std::make_shared<Arbitration>();
    auto keys = std::make_shared<ProcessWideKey>(arbitration);
    Words words;
    const GuestCopyOut backing{Words::write, Words::read, &words, Words::exchange};
    for (const auto svc : {0x34, 0x1a, 0x1c}) {
        std::stop_source cancellation;
        StartupQueries queries(owner, handles, thread, backing, arbitration, keys, cancellation.get_token());
        QueryRegisters registers{};
        if (svc == 0x34) {
            words.values[0] = 7;
            registers[0] = address;
            registers[1] = 2;
            registers[2] = 7;
            registers[3] = ~0ull;
        } else if (svc == 0x1a) {
            words.values[0] = *owner_handle | (1u << 30);
            registers[0] = *owner_handle;
            registers[1] = address;
            registers[2] = 8;
        } else {
            words.values[0] = 8;
            registers[0] = address;
            registers[1] = address + 4;
            registers[2] = 8;
            registers[3] = ~0ull;
        }
        const auto original = registers;
        auto waiter = std::async(std::launch::async, [&] { return queries.dispatch(svc, registers); });
        settle([&] {
            return svc == 0x34 ? arbitration->waiters(address) != 0 :
                   svc == 0x1a ? keys->mutex_waiters(address) != 0 : keys->waiters(address + 4) != 0;
        });
        cancellation.request_stop();
        CHECK(waiter.wait_for(5s) == std::future_status::ready);
        CHECK(waiter.get() == QueryDisposition::cancelled);
        CHECK(registers == original);
        CHECK(arbitration->waiters(address) == 0);
        CHECK(keys->mutex_waiters(address) == 0);
        CHECK(keys->waiters(address + 4) == 0);
        if (svc == 0x1c) CHECK(words.values[0].load() == 0);
        if (svc == 0x1a) CHECK(keys->unlock({
            [](void* raw, std::uint64_t at, std::uint32_t& value) noexcept {
                std::array<std::byte, 4> bytes{};
                if (!Words::read(raw, at, bytes)) return false;
                value = 0;
                for (unsigned i = 0; i < 4; ++i) value |= std::to_integer<std::uint32_t>(bytes[i]) << (i * 8);
                return true;
            },
            [](void* raw, std::uint64_t at, std::uint32_t value) noexcept {
                std::array<std::byte, 4> bytes{};
                for (unsigned i = 0; i < 4; ++i) bytes[i] = std::byte(value >> (i * 8));
                return Words::write(raw, at, bytes);
            }, &words, Words::exchange}, address));
    }
}


struct SynchronizationWords {
    static constexpr std::uint64_t base = 0x1200000040;
    std::array<Handle, 64> handles{};
    std::size_t readable = handles.size();
    unsigned reads = 0;
    unsigned byte_reads = 0;
    std::uint64_t byte_base = base + 1;
    std::uint64_t last_copy_address = 0;
    std::size_t last_copy_size = 0;
    static bool read_bytes(void* raw, std::uint64_t at, std::span<std::byte> bytes) noexcept {
        auto& words = *static_cast<SynchronizationWords*>(raw);
        ++words.byte_reads;
        words.last_copy_address = at;
        words.last_copy_size = bytes.size();
        const auto capacity = words.readable * sizeof(Handle);
        if (at < words.byte_base || at - words.byte_base > capacity ||
            bytes.size() > capacity - (at - words.byte_base)) return false;
        for (std::size_t index = 0; index != bytes.size(); ++index) {
            const auto offset = at - words.byte_base + index;
            bytes[index] = std::byte(words.handles[offset / 4] >> ((offset % 4) * 8));
        }
        return true;
    }
    static bool read(void* raw, std::uint64_t at, std::span<std::byte, 4> bytes) noexcept {
        auto& words = *static_cast<SynchronizationWords*>(raw);
        ++words.reads;
        if (at < base || (at - base) % 4 != 0 || (at - base) / 4 >= words.readable)
            return false;
        const auto value = words.handles[(at - base) / 4];
        for (unsigned index = 0; index != 4; ++index)
            bytes[index] = std::byte(value >> (index * 8));
        return true;
    }
};

void synchronization_validation_and_output_abi() {
    const auto owner = process();
    HandleTable handles(4);
    const auto thread = std::make_shared<ThreadRecord>(owner, 2, 32);
    const auto event = std::make_shared<Event>(true);
    const auto event_handle = handles.insert(event).value();
    const auto thread_handle = handles.insert(thread).value();
    const auto transfer = handles.insert(
        std::make_shared<TransferMemoryRecord>(AddressRange{address, 4096}, 0)).value();
    SynchronizationWords words;
    words.handles[0] = event_handle;
    const GuestCopyOut backing{nullptr, SynchronizationWords::read, &words};
    StartupQueries queries(owner, handles, thread, backing);
    const auto invoke = [&](std::uint16_t svc, QueryRegisters registers) {
        CHECK(queries.dispatch(svc, registers) == QueryDisposition::handled);
        for (std::size_t index = 2; index != registers.size(); ++index)
            CHECK(registers[index] == 0);
        return registers;
    };
    for (const auto count : {65ull, 0xffffffffull}) {
        const auto result = invoke(0x18, {0, ~0ull, count, ~0ull});
        CHECK(result[0] == 0xee01 && result[1] == ~0ull);
        CHECK(words.reads == 0);
    }
    for (const auto pointer : {ProcessRecord::query_end - 3, ProcessRecord::query_end,
                               ~std::uint64_t{0}}) {
        const auto result = invoke(0x18, {0, pointer, 1, ~0ull});
        CHECK(result[0] == 0xe601 && result[1] == pointer);
        CHECK(words.reads == 0);
    }
    words.handles[0] = 0;
    words.readable = 1;
    auto result = invoke(0x18, {0, SynchronizationWords::base, 2, ~0ull});
    CHECK(result[0] == 0xe601 && result[1] == SynchronizationWords::base);
    CHECK(words.reads == 2);
    for (const auto invalid : {Handle{0}, transfer, current_thread_handle}) {
        words.handles[0] = invalid;
        result = invoke(0x18, {0, SynchronizationWords::base, 1, ~0ull});
        CHECK(result[0] == 0xe401 && result[1] == SynchronizationWords::base);
    }
    words.handles[0] = event_handle;
    result = invoke(0x18, {0, SynchronizationWords::base, 0x100000001ull, ~0ull});
    CHECK(result[0] == 0 && result[1] == 0x1200000000);
    CHECK(event->reset());
    result = invoke(0x18, {0, SynchronizationWords::base, 1, 0});
    CHECK(result[0] == 0xea01 && result[1] == SynchronizationWords::base);
    result = invoke(0x18, {0, SynchronizationWords::base, 1, 1000000});
    CHECK(result[0] == 0xea01 && result[1] == 0x12ffffffff);
    for (const auto invalid : {Handle{0}, event_handle, current_process_handle}) {
        result = invoke(0x19, {invalid, ~0ull, ~0ull});
        CHECK(result[0] == 0xe401 && result[1] == 0);
    }
    for (const auto target : {thread_handle, current_thread_handle}) {
        result = invoke(0x19, {target, ~0ull, ~0ull});
        CHECK(result[0] == 0 && result[1] == 0);
        result = invoke(0x18, {0, SynchronizationWords::base, 1, 0});
        CHECK(result[0] == 0xea01 && result[1] == SynchronizationWords::base);
        result = invoke(0x18, {0, SynchronizationWords::base, 1, ~0ull});
        CHECK(result[0] == 0xec01 && result[1] == SynchronizationWords::base);
    }
    StartupQueries no_backing(owner, handles, thread);
    QueryRegisters empty{0, ~0ull, 0, 0};
    CHECK(no_backing.dispatch(0x18, empty) == QueryDisposition::handled);
    CHECK(empty[0] == 0xea01 && empty[1] == ~0ull);
    QueryRegisters unavailable{0, SynchronizationWords::base, 1, 0};
    const auto original = unavailable;
    CHECK(no_backing.dispatch(0x18, unavailable) == QueryDisposition::unsupported_operation);
    CHECK(unavailable == original);
    words.readable = 2;
    words.handles[1] = event_handle;
    // Several handles, none signaled, zero timeout: an immediate timeout that
    // keeps X1, as for one handle.
    QueryRegisters multiple{0, SynchronizationWords::base, 2, 0};
    CHECK(queries.dispatch(0x18, multiple) == QueryDisposition::handled);
    CHECK(multiple[0] == 0xea01 && multiple[1] == SynchronizationWords::base);

    // An aligned synchronization reader alone cannot implement an unaligned copy.
    for (const auto offset : {1u, 2u, 3u}) {
        QueryRegisters unaligned{0, SynchronizationWords::base + offset, 1, 0};
        const auto original_unaligned = unaligned;
        const auto previous_reads = words.reads;
        CHECK(queries.dispatch(0x18, unaligned) == QueryDisposition::unsupported_operation);
        CHECK(unaligned == original_unaligned && words.reads == previous_reads);
    }
    auto generic_backing = backing;
    generic_backing.read_bytes = SynchronizationWords::read_bytes;
    StartupQueries generic(owner, handles, thread, generic_backing);
    event->signal();
    words.readable = 1;
    for (const auto offset : {0u, 1u, 2u, 3u}) {
        words.byte_base = SynchronizationWords::base + offset;
        QueryRegisters unaligned{0, words.byte_base, 1, 0};
        const auto previous_word_reads = words.reads;
        CHECK(generic.dispatch(0x18, unaligned) == QueryDisposition::handled);
        CHECK(unaligned[0] == 0 && unaligned[1] == 0x1200000000);
        CHECK(words.last_copy_address == words.byte_base && words.last_copy_size == 4);
        CHECK(words.reads == previous_word_reads);
    }
    // The exact four-byte readable range rejects any rounded-down or oversized copy.
    CHECK(words.byte_reads == 4);
    const auto previous_byte_reads = words.byte_reads;
    QueryRegisters excessive{0, ~0ull, 65, 0};
    CHECK(generic.dispatch(0x18, excessive) == QueryDisposition::handled);
    CHECK(excessive[0] == 0xee01 && words.byte_reads == previous_byte_reads);
    QueryRegisters zero{0, ~0ull, 0, 0};
    CHECK(generic.dispatch(0x18, zero) == QueryDisposition::handled);
    CHECK(zero[0] == 0xea01 && words.byte_reads == previous_byte_reads);
    words.handles[0] = 0;
    QueryRegisters partial{0, words.byte_base, 2, 0};
    CHECK(generic.dispatch(0x18, partial) == QueryDisposition::handled);
    CHECK(partial[0] == 0xe601 && partial[1] == words.byte_base);
    CHECK(words.last_copy_address == words.byte_base && words.last_copy_size == 8);
    QueryRegisters invalid_handle{0, words.byte_base, 1, 0};
    CHECK(generic.dispatch(0x18, invalid_handle) == QueryDisposition::handled);
    CHECK(invalid_handle[0] == 0xe401 && invalid_handle[1] == words.byte_base);
    words.handles.fill(event_handle);
    words.readable = words.handles.size();
    // 64 handles of the signaled event: the first one completes the wait.
    QueryRegisters maximum{0, words.byte_base, 64, 0};
    CHECK(generic.dispatch(0x18, maximum) == QueryDisposition::handled);
    CHECK(maximum[0] == 0 && words.last_copy_size == 256);
}

void synchronization_blocked_cancellation_abi() {
    const auto owner = process();
    HandleTable handles(3);
    const auto caller = std::make_shared<ThreadRecord>(owner, 3, 32);
    const auto canceller = std::make_shared<ThreadRecord>(owner, 4, 32);
    const auto target_handle = handles.insert(caller).value();
    const auto event = std::make_shared<Event>();
    SynchronizationWords words;
    words.handles[0] = handles.insert(event).value();
    const GuestCopyOut backing{nullptr, SynchronizationWords::read, &words};
    std::stop_source host_stop;
    StartupQueries waiting(owner, handles, caller, backing, {}, {}, host_stop.get_token());
    StartupQueries cancelling(owner, handles, canceller);
    for (const auto count : {0ull, 1ull}) {
        QueryRegisters registers{0, SynchronizationWords::base, count, ~0ull};
        auto waiter = std::async(std::launch::async, [&] {
            return waiting.dispatch(0x18, registers);
        });
        settle([&] { return caller->synchronization_wait_active(); });
        QueryRegisters cancel{target_handle, ~0ull, ~0ull};
        CHECK(cancelling.dispatch(0x19, cancel) == QueryDisposition::handled);
        CHECK(cancel[0] == 0 && cancel[1] == 0);
        CHECK(waiter.wait_for(5s) == std::future_status::ready);
        CHECK(waiter.get() == QueryDisposition::handled);
        CHECK(registers[0] == 0xec01 && registers[1] == 0x12ffffffff);
        CHECK(!host_stop.stop_requested() && !event->is_signaled());
        for (std::size_t index = 2; index != registers.size(); ++index)
            CHECK(registers[index] == 0);
    }
    QueryRegisters registers{0, SynchronizationWords::base, 1, ~0ull};
    const auto original = registers;
    auto waiter = std::async(std::launch::async, [&] {
        return waiting.dispatch(0x18, registers);
    });
    settle([&] { return caller->synchronization_wait_active(); });
    CHECK(host_stop.request_stop());
    CHECK(waiter.wait_for(5s) == std::future_status::ready);
    CHECK(waiter.get() == QueryDisposition::cancelled && registers == original);
}

void mutex_and_condition_abi() {
    auto owner = process();
    HandleTable handles(4);
    auto thread = std::make_shared<ThreadRecord>(owner, 1, 32);
    const auto owner_handle = handles.insert(thread);
    const auto process_handle = handles.insert(owner);
    CHECK(owner_handle && process_handle);
    auto keys = std::make_shared<ProcessWideKey>();
    Words words;
    const GuestCopyOut backing{Words::write, Words::read, &words, Words::exchange};
    StartupQueries queries(owner, handles, thread, backing, {}, keys);
    const auto call = [&](std::uint8_t svc, QueryRegisters registers) {
        CHECK(queries.dispatch(svc, registers) == QueryDisposition::handled);
        for (std::size_t index = 1; index != registers.size(); ++index)
            CHECK(registers[index] == 0);
        return registers[0];
    };

    // The mutex range check precedes alignment and handle lookup for all three SVCs.
    // The kernel's word envelope ends at 0x40000000000, beyond the mapped 39-bit space.
    for (const auto svc : {0x1a, 0x1b, 0x1c}) {
        for (const auto bad : {0ull, 1ull, 0x1fffffull, 0x40000000000ull,
                               0x40000000001ull, ~0ull}) {
            QueryRegisters registers{};
            registers[svc == 0x1a ? 1 : 0] = bad;
            CHECK(call(svc, registers) == 0xd401);
        }
        for (const auto bad : {0x200001ull, address + 1, 0x7fffffffffull, 0x8000000001ull}) {
            QueryRegisters registers{};
            registers[svc == 0x1a ? 1 : 0] = bad;
            CHECK(call(svc, registers) == 0xcc01);
        }
        for (const auto unmapped : {0x200000ull, address + 8, 0x7ffffffffcull, 0x8000000000ull}) {
            QueryRegisters registers{};
            registers[svc == 0x1a ? 1 : 0] = unmapped;
            CHECK(call(svc, registers) == 0xd401);
        }
    }

    // A changed mutex word succeeds even with an invalid owner. Only a matching
    // owner word reaches typed handle validation; the waiter tag is not a handle check.
    for (const auto invalid_owner : {0u, 7u, *process_handle, 0xffff8000u}) {
        words.values[0] = 0;
        CHECK(call(0x1a, {invalid_owner, address, 0xffffffffu}) == 0);
        words.values[0] = invalid_owner | 0x40000000u;
        CHECK(call(0x1a, {invalid_owner, address, 0xffffffffu}) == 0xe401);
    }
    words.mutex_readable = false;
    CHECK(call(0x1a, {7, address, 8}) == 0xd401);
    words.mutex_readable = true;
    words.values[0] = *owner_handle | 0x40000000u;
    auto lock_waiter = std::async(std::launch::async, [&] {
        return call(0x1a, {*owner_handle, address, 0xffffffffu});
    });
    settle([&] { return keys->mutex_waiters(address) == 1; });
    CHECK(call(0x1b, {address}) == 0);
    CHECK(lock_waiter.wait_for(5s) == std::future_status::ready);
    CHECK(lock_waiter.get() == 0);
    CHECK(words.values[0] == 0xffffffffu);

    // An inaccessible condition word is still a valid queue key. Neither arming nor
    // clearing it fails the SVC, and the CV wait does not read the mutex before release.
    words.mutex_readable = false;
    for (const auto key : {0ull, 1ull, ~0ull}) {
        words.values[0] = 8;
        CHECK(call(0x1c, {address, key, 8, 0}) == 0xea01);
        CHECK(words.values[0] == 0);
        CHECK(call(0x1d, {key, 0}) == 0);
    }
    words.mutex_readable = true;
    words.mutex_writable = false;
    words.values[1] = 0;
    CHECK(call(0x1c, {address, address + 4, 8, 0}) == 0xd401);
    CHECK(words.values[1] == 1);
    words.mutex_writable = true;

    // All four byte addresses name the same condition queue and aligned condition word.
    for (std::uint64_t offset = 0; offset != 4; ++offset) {
        words.values[0] = 8;
        auto waiter = std::async(std::launch::async, [&] {
            return call(0x1c, {address, address + 4 + offset, 8, ~0ull});
        });
        settle([&] { return keys->waiters(address + 4) == 1; });
        CHECK(words.values[1] == 1);
        CHECK(call(0x1d, {address + 4 + ((offset + 1) % 4), 0}) == 0);
        CHECK(waiter.wait_for(5s) == std::future_status::ready);
        CHECK(waiter.get() == 0);
        CHECK(words.values[0] == 8);
        CHECK(words.values[1] == 0);
    }
    auto unmapped_key_waiter = std::async(std::launch::async, [&] {
        return call(0x1c, {address, 1, 8, ~0ull});
    });
    settle([&] { return keys->waiters(0) == 1; });
    CHECK(call(0x1d, {3, ~0ull}) == 0);
    CHECK(unmapped_key_waiter.wait_for(5s) == std::future_status::ready);
    CHECK(unmapped_key_waiter.get() == 0);

    // The signal itself succeeds, but a stale mutex owner fails the released
    // condition waiter after marking the owned word as contended.
    for (const auto held_owner : {Handle{7}, *owner_handle}) {
        auto waiter = std::async(std::launch::async, [&] {
            return call(0x1c, {address, address + 4, 8, ~0ull});
        });
        settle([&] { return keys->waiters(address + 4) == 1; });
        words.values[0] = held_owner;
        CHECK(call(0x1d, {address + 4, 1}) == 0);
        CHECK(words.values[0] == (held_owner | 0x40000000u));
        if (held_owner == *owner_handle) {
            CHECK(keys->mutex_waiters(address) == 1);
            CHECK(call(0x1b, {address}) == 0);
        }
        CHECK(waiter.wait_for(5s) == std::future_status::ready);
        CHECK(waiter.get() == (held_owner == *owner_handle ? 0 : 0xfa01));
        CHECK(keys->mutex_waiters(address) == 0);
    }
}
}
void guest_cancellation_survives_mutex_and_condition_waits() {
    const auto owner = process();
    HandleTable handles(4);
    const auto caller = std::make_shared<ThreadRecord>(owner, 10, 32);
    const auto other = std::make_shared<ThreadRecord>(owner, 11, 32);
    const auto caller_handle = handles.insert(caller).value();
    const auto owner_handle = handles.insert(other).value();
    auto keys = std::make_shared<ProcessWideKey>();
    Words words;
    const GuestCopyOut backing{Words::write, Words::read, &words, Words::exchange};
    StartupQueries waiting(owner, handles, caller, backing, {}, keys);
    StartupQueries signalling(owner, handles, other, backing, {}, keys);
    for (const auto svc : {0x1a, 0x1c}) {
        words.values[0] = svc == 0x1a ? owner_handle | 0x40000000u : caller_handle;
        QueryRegisters input = svc == 0x1a
            ? QueryRegisters{owner_handle, address, caller_handle}
            : QueryRegisters{address, address + 4, caller_handle, UINT64_MAX};
        auto waiter = std::async(std::launch::async, [&] {
            CHECK(waiting.dispatch(svc, input) == QueryDisposition::handled);
            return input[0];
        });
        settle([&] { return svc == 0x1a ? keys->mutex_waiters(address) == 1
                                        : keys->waiters(address + 4) == 1; });
        QueryRegisters cancel{caller_handle};
        CHECK(signalling.dispatch(0x19, cancel) == QueryDisposition::handled && cancel[0] == 0);
        CHECK(waiter.wait_for(20ms) == std::future_status::timeout);
        QueryRegisters release{svc == 0x1a ? address : address + 4, 1};
        CHECK(signalling.dispatch(svc == 0x1a ? 0x1b : 0x1d, release) ==
              QueryDisposition::handled && release[0] == 0);
        CHECK(waiter.wait_for(5s) == std::future_status::ready && waiter.get() == 0);
        CHECK(words.values[0] == caller_handle);
        QueryRegisters unlock{address};
        CHECK(waiting.dispatch(0x1b, unlock) == QueryDisposition::handled && unlock[0] == 0);
        QueryRegisters pending{0, UINT64_MAX, 0, UINT64_MAX};
        CHECK(waiting.dispatch(0x18, pending) == QueryDisposition::handled);
        CHECK(pending[0] == 0xec01 && pending[1] == UINT64_MAX);
        QueryRegisters poll{0, UINT64_MAX, 0, 0};
        CHECK(waiting.dispatch(0x18, poll) == QueryDisposition::handled && poll[0] == 0xea01);
    }
}

int main() {
    routed_waits();
    synchronization_validation_and_output_abi();
    synchronization_blocked_cancellation_abi();
    mutex_and_condition_abi();
    guest_cancellation_survives_mutex_and_condition_waits();
}
