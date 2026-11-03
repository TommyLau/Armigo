// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independent verification of the Horizon kernel-object handle table.
//
// Scenarios use synthetic objects authored for this file. They cover the
// constructor capacity limits, handle encoding and rejection rules, stale
// generations before and after the documented serial wrap, object pinning
// across close and table destruction, allocation-free steady-state
// operations, reentrant destructors, and concurrent insert/lookup/close.

#include <armigo/horizon/handle_table.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace {

using armigo::horizon::Handle;
using armigo::horizon::HandleTable;
using armigo::horizon::KernelObject;

// ---------------------------------------------------------------------------
// Minimal assertion harness
// ---------------------------------------------------------------------------

int g_checks_failed = 0;
int g_checks_total = 0;

void record_check(bool passed, const char* text, const char* file, int line) {
    ++g_checks_total;
    if (!passed) {
        ++g_checks_failed;
        std::printf("FAIL %s:%d: %s\n", file, line, text);
    }
}

#define CHECK(condition) record_check((condition), #condition, __FILE__, __LINE__)
#define CHECK_EQ(actual, expected)                                        \
    record_check((actual) == (expected), #actual " == " #expected,        \
                 __FILE__, __LINE__)

// Terminates the process if the guarded section does not finish in time, so
// a deadlock reports as a failure instead of hanging the suite.
class Watchdog {
public:
    explicit Watchdog(int seconds, int exit_code)
        : done_{false},
          thread_{[this, seconds, exit_code] {
              for (int waited = 0; waited < seconds * 20; ++waited) {
                  if (done_.load()) {
                      return;
                  }
                  std::this_thread::sleep_for(std::chrono::milliseconds{50});
              }
              std::printf("FAIL watchdog timeout: operation did not complete\n");
              std::_Exit(exit_code);
          }} {}

    ~Watchdog() {
        done_ = true;
        thread_.join();
    }

    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

private:
    std::atomic<bool> done_;
    std::thread thread_;
};

// ---------------------------------------------------------------------------
// Synthetic objects
// ---------------------------------------------------------------------------

struct PlainObject : KernelObject {};

struct AlternateObject : KernelObject {};

struct TaggedObject : KernelObject {
    int tag;
    explicit TaggedObject(int value) : tag(value) {}
};

struct CountingObject : KernelObject {
    static std::atomic<int> live;
    CountingObject() { ++live; }
    ~CountingObject() override { --live; }
};

std::atomic<int> CountingObject::live{0};

// Destructor reenters the live table: it looks up and closes the paired
// handle. If close dropped the object reference while still holding the
// table lock, these calls would deadlock and the watchdog would fail the
// run.
struct ReentrantDeleter : KernelObject {
    HandleTable* table;
    Handle other;
    std::shared_ptr<KernelObject>* observed;
    bool* paired_close_result;
    bool* destroyed;

    ReentrantDeleter(HandleTable* host, Handle paired, std::shared_ptr<KernelObject>* slot,
                     bool* close_slot, bool* destroyed_slot)
        : table(host),
          other(paired),
          observed(slot),
          paired_close_result(close_slot),
          destroyed(destroyed_slot) {}

    ~ReentrantDeleter() override {
        *observed = table->lookup(other);
        *paired_close_result = table->close(other);
        *destroyed = true;
    }
};

// ---------------------------------------------------------------------------
// Global allocation counter for the steady-state check
// ---------------------------------------------------------------------------

std::atomic<long> g_allocations{0};

}  // namespace

void* operator new(std::size_t size) {
    ++g_allocations;
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return operator new(size); }

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {

// ---------------------------------------------------------------------------
// Constructor capacity limits
// ---------------------------------------------------------------------------

void test_constructor_capacity_limits() {
    HandleTable one(1);
    CHECK_EQ(one.size(), std::size_t{0});
    CHECK_EQ(one.capacity(), std::size_t{1});

    HandleTable full(HandleTable::max_capacity);
    CHECK_EQ(full.size(), std::size_t{0});
    CHECK_EQ(full.capacity(), HandleTable::max_capacity);

    bool rejected_zero = false;
    try {
        HandleTable zero(0);
        (void)zero;
    } catch (const std::invalid_argument&) {
        rejected_zero = true;
    }
    CHECK(rejected_zero);

    bool rejected_too_large = false;
    try {
        HandleTable too_large(HandleTable::max_capacity + 1);
        (void)too_large;
    } catch (const std::invalid_argument&) {
        rejected_too_large = true;
    }
    CHECK(rejected_too_large);
}

// ---------------------------------------------------------------------------
// Insert, lookup, lookup_as, close, size basics
// ---------------------------------------------------------------------------

void test_basic_lifecycle() {
    HandleTable table(8);
    auto object = std::make_shared<TaggedObject>(7);

    auto handle = table.insert(object);
    CHECK(handle.has_value());
    CHECK_EQ(table.size(), std::size_t{1});

    auto found = table.lookup(*handle);
    CHECK(found != nullptr);
    CHECK(found.get() == object.get());

    auto typed = table.lookup_as<TaggedObject>(*handle);
    CHECK(typed != nullptr);
    CHECK(typed->tag == 7);

    CHECK(table.close(*handle));
    CHECK_EQ(table.size(), std::size_t{0});
    CHECK(table.lookup(*handle) == nullptr);
    CHECK(table.lookup_as<TaggedObject>(*handle) == nullptr);
    CHECK(!table.close(*handle));  // Closing twice fails.

    object.reset();
    CHECK(object == nullptr);
}

void test_null_insert_fails() {
    HandleTable table(4);
    CHECK(!table.insert(nullptr).has_value());
    CHECK_EQ(table.size(), std::size_t{0});
}

void test_exhaustion_fails_then_recovers() {
    HandleTable table(2);
    auto first = table.insert(std::make_shared<PlainObject>());
    auto second = table.insert(std::make_shared<PlainObject>());
    CHECK(first.has_value());
    CHECK(second.has_value());
    CHECK(*first != *second);

    auto third = table.insert(std::make_shared<PlainObject>());
    CHECK(!third.has_value());  // Exhausted.
    CHECK(third == std::nullopt);
    CHECK_EQ(table.size(), std::size_t{2});

    CHECK(table.close(*first));
    auto recycled = table.insert(std::make_shared<PlainObject>());
    CHECK(recycled.has_value());
    CHECK_EQ(table.size(), std::size_t{2});
}

// ---------------------------------------------------------------------------
// Duplicates: the same object under distinct handles
// ---------------------------------------------------------------------------

void test_duplicate_object_distinct_handles() {
    HandleTable table(4);
    auto shared = std::make_shared<CountingObject>();
    CHECK_EQ(CountingObject::live.load(), 1);

    auto first = table.insert(shared);
    auto second = table.insert(shared);
    CHECK(first.has_value() && second.has_value());
    CHECK(*first != *second);
    CHECK_EQ(table.size(), std::size_t{2});

    {
        auto a = table.lookup(*first);
        auto b = table.lookup(*second);
        CHECK(a != nullptr && b != nullptr);
        CHECK(a.get() == b.get());
        CHECK(a.get() == shared.get());
    }  // Drop the lookup pins before counting references.

    CHECK(table.close(*first));
    CHECK_EQ(CountingObject::live.load(), 1);  // Second entry still owns it.
    {
        auto survivor = table.lookup(*second);
        CHECK(survivor != nullptr && survivor.get() == shared.get());
    }  // Drop the pin before counting references.
    CHECK(table.lookup(*first) == nullptr);  // Closed handle is stale.

    CHECK(table.close(*second));
    shared.reset();
    CHECK_EQ(CountingObject::live.load(), 0);  // Released after both handles.
}

// ---------------------------------------------------------------------------
// Handle encoding rejections
// ---------------------------------------------------------------------------

void test_lookup_rejects_malformed_handles() {
    HandleTable table(4);
    auto handle = table.insert(std::make_shared<TaggedObject>(1));
    CHECK(handle.has_value());
    const Handle valid = *handle;

    CHECK(table.lookup(0) == nullptr);  // Zero handle.

    // Reserved bits 30..31, including the pseudo-handle range.
    CHECK(table.lookup(valid | 0x40000000u) == nullptr);
    CHECK(table.lookup(valid | 0x80000000u) == nullptr);
    CHECK(table.lookup(valid | 0xC0000000u) == nullptr);
    CHECK(table.lookup(0xFFFF8000u) == nullptr);  // Current-thread pseudo-handle.
    CHECK(table.lookup(0xFFFF8001u) == nullptr);  // Current-process pseudo-handle.
    CHECK(!table.close(0xFFFF8000u));

    // Zero serial with an in-range, occupied index.
    const Handle index_only = valid & 0x7FFFu;
    CHECK(index_only < table.capacity());
    CHECK(table.lookup(index_only) == nullptr);
    CHECK(!table.close(index_only));

    // Index at or beyond capacity with a plausible serial.
    CHECK(table.lookup(static_cast<Handle>(table.capacity()) | 0x8000u) == nullptr);
    CHECK(table.lookup(0x7FFFu | 0x8000u) == nullptr);
}

void test_lookup_rejects_wrong_dynamic_type() {
    HandleTable table(4);
    auto handle = table.insert(std::make_shared<AlternateObject>());
    CHECK(handle.has_value());

    CHECK(table.lookup(*handle) != nullptr);
    CHECK(table.lookup_as<AlternateObject>(*handle) != nullptr);
    CHECK(table.lookup_as<PlainObject>(*handle) == nullptr);
    CHECK(table.lookup_as<TaggedObject>(*handle) == nullptr);
    CHECK(table.lookup_as<KernelObject>(*handle) != nullptr);

    // A wrong-type value must not succeed after close either.
    CHECK(table.close(*handle));
    CHECK(table.lookup_as<AlternateObject>(*handle) == nullptr);
}

// ---------------------------------------------------------------------------
// Stale handles and the documented serial wrap
// ---------------------------------------------------------------------------

void test_stale_generation_before_wrap() {
    HandleTable table(4);
    auto first = table.insert(std::make_shared<TaggedObject>(1));
    CHECK(first.has_value());

    CHECK(table.close(*first));
    CHECK(table.lookup(*first) == nullptr);  // Stale: serial mismatch.
    CHECK(!table.close(*first));             // Stale close removes nothing.

    auto second = table.insert(std::make_shared<TaggedObject>(2));
    CHECK(second.has_value());
    CHECK(*second != *first);  // A reused slot must change its generation.
    CHECK(((*second >> 15) & 0x7FFFu) != ((*first >> 15) & 0x7FFFu));

    auto pinned = table.lookup_as<TaggedObject>(*second);
    CHECK(pinned != nullptr && pinned->tag == 2);
    CHECK(table.lookup(*first) == nullptr);  // Still stale until wrap.
}

void test_serial_wraps_and_stale_handle_revives() {
    // The per-table serial advances 1..32767 and wraps back to 1. After a
    // full cycle the original slot and serial recur, so a previously stale
    // handle value becomes valid again. This verifies the documented
    // finite-generation limit; the table does not promise that a stale
    // handle stays invalid forever.
    HandleTable table(1);
    auto original = table.insert(std::make_shared<TaggedObject>(0));
    CHECK(original.has_value());
    CHECK_EQ(*original, 0x8000u);  // Index 0, serial 1.
    CHECK(table.close(*original));
    CHECK(table.lookup(*original) == nullptr);

    for (int step = 2; step <= 32767; ++step) {
        auto handle = table.insert(std::make_shared<PlainObject>());
        CHECK(handle.has_value());
        CHECK_EQ((*handle >> 15) & 0x7FFFu, static_cast<Handle>(step));
        CHECK(table.close(*handle));
    }
    CHECK_EQ(table.size(), std::size_t{0});

    // Allocation 32768 wraps the serial to 1 in slot 0.
    auto revived = table.insert(std::make_shared<TaggedObject>(9));
    CHECK(revived.has_value());
    CHECK_EQ(*revived, *original);  // The serial cycle completed.

    auto found = table.lookup_as<TaggedObject>(*original);
    CHECK(found != nullptr && found->tag == 9);
}

// ---------------------------------------------------------------------------
// Pinning across close and table destruction
// ---------------------------------------------------------------------------

void test_pin_survives_close_and_table_destruction() {
    CountingObject::live = 0;
    std::shared_ptr<CountingObject> pin_across_close;

    {
        HandleTable table(4);
        auto handle = table.insert(std::make_shared<CountingObject>());
        CHECK(handle.has_value());
        pin_across_close = table.lookup_as<CountingObject>(*handle);
        CHECK(pin_across_close != nullptr);

        CHECK(table.close(*handle));
        CHECK_EQ(CountingObject::live.load(), 1);  // Pin keeps the object.
    }  // Table destroyed; its reference was already dropped by close.

    CHECK_EQ(CountingObject::live.load(), 1);  // Pin still owns the object.
    pin_across_close.reset();
    CHECK_EQ(CountingObject::live.load(), 0);

    // A pin taken while the table is alive also outlives table destruction
    // when the entry was never closed.
    std::shared_ptr<CountingObject> pin_across_destruction;
    {
        HandleTable table(4);
        auto handle = table.insert(std::make_shared<CountingObject>());
        CHECK(handle.has_value());
        pin_across_destruction = table.lookup_as<CountingObject>(*handle);
    }  // Table destruction releases its reference only.
    CHECK_EQ(CountingObject::live.load(), 1);
    pin_across_destruction.reset();
    CHECK_EQ(CountingObject::live.load(), 0);
}

// ---------------------------------------------------------------------------
// Reentrant destructor proves close unlocks first
// ---------------------------------------------------------------------------

void test_reentrant_destructor_runs_outside_the_lock() {
    Watchdog watchdog{20, 70};  // A deadlocked close exits with code 70.

    HandleTable table(4);
    auto paired = table.insert(std::make_shared<PlainObject>());
    CHECK(paired.has_value());

    std::shared_ptr<KernelObject> observed;
    bool paired_close_result = false;
    bool destroyed = false;
    {
        auto deleter = std::make_shared<ReentrantDeleter>(&table, *paired, &observed,
                                                          &paired_close_result, &destroyed);
        auto deleter_handle = table.insert(deleter);
        CHECK(deleter_handle.has_value());
        deleter.reset();  // The table owns the last reference.
        CHECK_EQ(table.size(), std::size_t{2});

        // If close dropped the reference while holding the table lock, the
        // destructor's lookup and close calls would deadlock.
        CHECK(table.close(*deleter_handle));
    }

    CHECK(destroyed);            // The destructor ran inside close.
    CHECK(observed != nullptr);  // Its reentrant lookup saw the paired entry.
    CHECK(paired_close_result);  // Its reentrant close succeeded.
    CHECK_EQ(table.size(), std::size_t{0});
    CHECK(!table.close(*paired));  // Already closed from the destructor.
    CHECK(table.lookup(*paired) == nullptr);
}

// ---------------------------------------------------------------------------
// Steady-state operations allocate nothing
// ---------------------------------------------------------------------------

void test_steady_state_operations_do_not_allocate() {
    constexpr std::size_t kSlots = 8;
    constexpr int kRounds = 4;

    HandleTable table(kSlots);
    std::vector<Handle> handles;
    for (std::size_t i = 0; i < kSlots; ++i) {
        auto handle = table.insert(std::make_shared<PlainObject>());
        CHECK(handle.has_value());
        handles.push_back(*handle);
    }

    // Replacement objects are created before the measured region; moving a
    // shared_ptr never allocates.
    std::vector<std::shared_ptr<KernelObject>> spares;
    spares.reserve(kSlots * kRounds);
    for (std::size_t i = 0; i < kSlots * kRounds; ++i) {
        spares.push_back(std::make_shared<PlainObject>());
    }
    std::size_t next_spare = 0;

    g_allocations = 0;
    for (int round = 0; round < kRounds; ++round) {
        for (Handle handle : handles) {
            auto pinned = table.lookup(handle);
            auto typed = table.lookup_as<KernelObject>(handle);
            CHECK(pinned != nullptr && typed != nullptr);
            CHECK_EQ(table.size(), kSlots);
        }
        for (std::size_t i = 0; i < handles.size(); ++i) {
            CHECK(table.close(handles[i]));
            CHECK_EQ(table.size(), kSlots - 1);
            auto handle_again = table.insert(std::move(spares[next_spare++]));
            CHECK(handle_again.has_value());
            handles[i] = *handle_again;
        }
    }
    CHECK_EQ(g_allocations.load(), 0);
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

void test_concurrent_churn_is_consistent() {
    constexpr int kThreads = 4;
    constexpr int kRounds = 400;
    constexpr int kObjectsPerThread = 3;
    HandleTable table(kThreads * kObjectsPerThread);

    std::atomic<int> wrong_object{0};
    std::atomic<int> insert_failures{0};
    std::vector<std::thread> workers;
    for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
        workers.emplace_back([&, thread_id] {
            for (int round = 0; round < kRounds; ++round) {
                std::vector<Handle> mine;
                for (int slot = 0; slot < kObjectsPerThread; ++slot) {
                    auto handle = table.insert(std::make_shared<TaggedObject>(thread_id));
                    if (!handle) {
                        ++insert_failures;
                        continue;
                    }
                    mine.push_back(*handle);
                }
                for (Handle handle : mine) {
                    auto found = table.lookup_as<TaggedObject>(handle);
                    if (found == nullptr || found->tag != thread_id) {
                        ++wrong_object;  // Lookup must never return another object.
                    }
                }
                for (Handle handle : mine) {
                    if (!table.close(handle)) {
                        ++wrong_object;  // Own handle must close exactly once.
                    }
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    CHECK_EQ(wrong_object.load(), 0);
    CHECK_EQ(insert_failures.load(), 0);
    CHECK_EQ(table.size(), std::size_t{0});
}

void test_concurrent_close_races_with_lookup() {
    constexpr int kThreads = 4;
    HandleTable table(4);
    auto handle = table.insert(std::make_shared<TaggedObject>(42));
    CHECK(handle.has_value());

    std::atomic<int> wrong_object{0};
    std::atomic<bool> releasing{false};
    std::vector<std::thread> readers;
    for (int i = 0; i < kThreads; ++i) {
        readers.emplace_back([&] {
            while (!releasing.load()) {
                // The entry is being closed concurrently: the result is
                // either null or the one correct object, never another one.
                auto found = table.lookup_as<TaggedObject>(*handle);
                if (found != nullptr && found->tag != 42) {
                    ++wrong_object;
                }
            }
        });
    }

    CHECK(table.close(*handle));
    releasing = true;
    for (auto& reader : readers) {
        reader.join();
    }

    CHECK_EQ(wrong_object.load(), 0);
    CHECK_EQ(table.size(), std::size_t{0});
}

}  // namespace

int main() {
    test_constructor_capacity_limits();
    test_basic_lifecycle();
    test_null_insert_fails();
    test_exhaustion_fails_then_recovers();
    test_duplicate_object_distinct_handles();
    test_lookup_rejects_malformed_handles();
    test_lookup_rejects_wrong_dynamic_type();
    test_stale_generation_before_wrap();
    test_serial_wraps_and_stale_handle_revives();
    test_pin_survives_close_and_table_destruction();
    test_reentrant_destructor_runs_outside_the_lock();
    test_steady_state_operations_do_not_allocate();
    test_concurrent_churn_is_consistent();
    test_concurrent_close_races_with_lookup();

    std::printf("%d checks, %d failed\n", g_checks_total, g_checks_failed);
    return g_checks_failed == 0 ? 0 : 1;
}
