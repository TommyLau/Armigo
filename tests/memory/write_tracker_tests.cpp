// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Scenario tests for the page-level CPU write tracker use independently
// authored fixtures: every mapped page is synthetic and owned by the test.
// Faults dispatch through the production signal handler installed by
// WriteTracker::install().

#include <armigo/memory/write_tracker.h>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

namespace {

namespace memory = armigo::memory;

std::atomic<int> g_failures{0};
std::atomic<int> g_checks{0};

#define CHECK(condition)                                                    \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(condition)) {                                                 \
            ++g_failures;                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,    \
                         #condition);                                       \
        }                                                                   \
    } while (0)

std::size_t page_size() { return static_cast<std::size_t>(::getpagesize()); }

void* map_pages(std::size_t count) {
    void* p = ::mmap(nullptr, count * page_size(), PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) {
        std::perror("mmap");
        std::exit(2);
    }
    return p;
}

void unmap_pages(void* p, std::size_t count) {
    CHECK(::munmap(p, count * page_size()) == 0);
}

struct SharedMappings {
    std::uint8_t* primary;
    std::uint8_t* alias;
    std::size_t size;
};

SharedMappings map_shared_pages(std::size_t count) {
    auto path = (std::filesystem::temp_directory_path() /
                 "armigo-write-tracker.XXXXXX").string();
    const int fd = ::mkstemp(path.data());
    if (fd < 0) {
        std::perror("mkstemp");
        std::exit(2);
    }
    ::unlink(path.c_str());
    const std::size_t size = count * page_size();
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
        std::perror("ftruncate");
        std::exit(2);
    }
    void* primary = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
    void* alias = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
    ::close(fd);
    if (primary == MAP_FAILED || alias == MAP_FAILED) {
        std::perror("mmap shared aliases");
        std::exit(2);
    }
    return {static_cast<std::uint8_t*>(primary),
            static_cast<std::uint8_t*>(alias), size};
}

void unmap_shared_pages(const SharedMappings& mappings) {
    CHECK(::munmap(mappings.primary, mappings.size) == 0);
    CHECK(::munmap(mappings.alias, mappings.size) == 0);
}

// Same-value store: only write tracking, not content, can reveal it. Relaxed
// atomics keep racing synthetic guest stores well defined while still
// compiling to an ordinary faulting byte store.
void guest_store(std::uint8_t* address) {
    __atomic_store_n(address, __atomic_load_n(address, __ATOMIC_RELAXED),
                     __ATOMIC_RELAXED);
}

std::uint64_t fault_count() {
    const auto stats = memory::WriteTracker::stats();
    return stats.faults;
}
// A store must not fault; it proves the host page is writable.
bool writable(std::uint8_t* address) {
    const auto before = fault_count();
    guest_store(address);
    return fault_count() == before;
}
// A store must fault exactly once; it proves the host page was protected.
bool protected_store(std::uint8_t* address) {
    const auto before = fault_count();
    guest_store(address);
    return fault_count() == before + 1;
}

// protect -> store faults once -> written_since true and the page is
// writable; a span without stores stays clean.
void test_protect_write_and_clean() {
    constexpr std::size_t kPages = 2;
    auto* p = static_cast<std::uint8_t*>(map_pages(kPages));
    memory::WriteTracker::add_range(p, kPages * page_size());
    CHECK(memory::WriteTracker::tracked(p, kPages * page_size()));
    CHECK(!memory::WriteTracker::tracked(p + page_size(), page_size() + 1));
    CHECK(!memory::WriteTracker::tracked(p, 0));
    const auto stamp = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(p, kPages * page_size()));
    CHECK(!memory::WriteTracker::written_since(p + page_size(), page_size(), stamp));
    CHECK(protected_store(p + page_size()));
    CHECK(memory::WriteTracker::written_since(p + page_size(), page_size(), stamp));
    CHECK(writable(p + page_size()));
    CHECK(!memory::WriteTracker::written_since(p, page_size(), stamp));
    // Zero stamp and empty spans keep their conservative answers.
    CHECK(memory::WriteTracker::written_since(p + page_size(), page_size(), 0));
    CHECK(!memory::WriteTracker::written_since(p + page_size(), 0, stamp));
    memory::WriteTracker::remove_range(p);
    unmap_pages(p, kPages);
}

// Overlapping registrations share one canonical record per host page: a write
// through any owner is observed by every owner, and removal restores
// writability only for pages the removed range alone owns.
void test_shared_records_and_removal() {
    constexpr std::size_t kPages = 5;
    auto* base = static_cast<std::uint8_t*>(map_pages(kPages));
    const auto pg = page_size();
    // A owns pages 0-1, B owns pages 1-2; page 1 is shared.
    memory::WriteTracker::add_range(base, 2 * pg);
    const auto stamp_a = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(base, 2 * pg));
    memory::WriteTracker::add_range(base + pg, 2 * pg);
    const auto stamp_b = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(base + pg, 2 * pg));
    CHECK(!memory::WriteTracker::written_since(base, 2 * pg, stamp_a));
    CHECK(!memory::WriteTracker::written_since(base + pg, 2 * pg, stamp_b));

    // Removing A keeps the shared page protected for B and untracks A's page.
    memory::WriteTracker::remove_range(base);
    CHECK(memory::WriteTracker::tracked(base + pg, 2 * pg));
    CHECK(!memory::WriteTracker::written_since(base + pg, 2 * pg, stamp_b));
    CHECK(writable(base));
    CHECK(!memory::WriteTracker::tracked(base, pg));

    // One same-value store on the shared page dirties the survivor.
    CHECK(protected_store(base + pg));
    CHECK(memory::WriteTracker::written_since(base + pg, 2 * pg, stamp_b));
    CHECK(!memory::WriteTracker::written_since(base + 2 * pg, pg, stamp_b));

    // Disjoint byte registrations on one host page share its record: a store
    // inside either registration is a coherent hit for both.
    auto* shared_page = base + 3 * pg;
    memory::WriteTracker::add_range(shared_page + 16, 16);
    memory::WriteTracker::add_range(shared_page + pg / 2, 16);
    const auto stamp_shared = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(shared_page + 16, 16));
    CHECK(protected_store(shared_page + pg / 2));
    CHECK(memory::WriteTracker::written_since(shared_page + 16, 16, stamp_shared));
    CHECK(memory::WriteTracker::written_since(shared_page + pg / 2, 16, stamp_shared));

    memory::WriteTracker::remove_range(shared_page + 16);
    memory::WriteTracker::remove_range(shared_page + pg / 2);
    memory::WriteTracker::remove_range(base + pg);
    CHECK(writable(base + pg));
    CHECK(writable(base + 2 * pg));
    CHECK(writable(shared_page));
    unmap_pages(base, kPages);
}

// Re-registration invalidates stamps that belonged to the previous storage.
void test_reregistration_bumps_generation() {
    auto* p = static_cast<std::uint8_t*>(map_pages(1));
    memory::WriteTracker::add_range(p, page_size());
    const auto first = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(p, page_size()));
    memory::WriteTracker::remove_range(p);
    memory::WriteTracker::add_range(p, page_size());
    // The old stamp predates the current registration, so the answer is true
    // even though no byte changed.
    CHECK(memory::WriteTracker::written_since(p, page_size(), first));
    const auto second = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(p, page_size()));
    CHECK(!memory::WriteTracker::written_since(p, page_size(), second));
    memory::WriteTracker::remove_range(p);
    unmap_pages(p, 1);
}

void test_note_store_publication() {
    auto* p = static_cast<std::uint8_t*>(map_pages(3));
    const auto pg = page_size();
    memory::WriteTracker::add_range(p, 3 * pg);
    const auto stamp = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(p, 3 * pg));

    // An unaligned store crossing one boundary dirties exactly those pages
    // and leaves the untouched third page clean.
    memory::WriteTracker::note_store(p + pg - 2, 4);
    CHECK(memory::WriteTracker::written_since(p, pg, stamp));
    CHECK(memory::WriteTracker::written_since(p + pg, pg, stamp));
    CHECK(!memory::WriteTracker::written_since(p + 2 * pg, pg, stamp));
    CHECK(protected_store(p));

    // Repeated publication also invalidates stamps while a page is already
    // writable, without changing that state.
    const auto writable_stamp = memory::WriteTracker::now();
    memory::WriteTracker::note_store(p + 7, 1);
    CHECK(memory::WriteTracker::written_since(p, pg, writable_stamp));
    const auto repeated_stamp = memory::WriteTracker::now();
    memory::WriteTracker::note_store(p + 9, 1);
    CHECK(memory::WriteTracker::written_since(p, pg, repeated_stamp));
    CHECK(writable(p));

    memory::WriteTracker::remove_range(p);
    unmap_pages(p, 3);
}

void test_note_store_across_registrations() {
    auto* p = static_cast<std::uint8_t*>(map_pages(2));
    const auto pg = page_size();
    memory::WriteTracker::add_range(p + 32, pg - 32);
    memory::WriteTracker::add_range(p + pg + 16, pg - 16);
    const auto stamp = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(p + 32, pg - 32));
    CHECK(memory::WriteTracker::protect_checked(p + pg + 16, pg - 16));

    // One host store can cross independently registered neighboring pages.
    memory::WriteTracker::note_store(p + pg - 1, 18);
    CHECK(memory::WriteTracker::written_since(p + 32, pg - 32, stamp));
    CHECK(memory::WriteTracker::written_since(p + pg + 16, pg - 16,
                                               stamp));
    CHECK(protected_store(p));
    CHECK(protected_store(p + pg));

    memory::WriteTracker::remove_range(p + 32);
    memory::WriteTracker::remove_range(p + pg + 16);
    unmap_pages(p, 2);
}

std::atomic<int> g_alias_copy_gate{0};
std::uint8_t* g_alias_copy_source = nullptr;
struct sigaction g_alias_copy_bus{}, g_alias_copy_segv{};

void pause_alias_copy(int sig, siginfo_t* info, void* context) {
    const auto fault = reinterpret_cast<std::uintptr_t>(info ? info->si_addr : nullptr);
    const auto source = reinterpret_cast<std::uintptr_t>(g_alias_copy_source);
    int armed = 1;
    if (source && fault >= source && fault - source < 4 &&
        g_alias_copy_gate.compare_exchange_strong(armed, 2)) {
        while (g_alias_copy_gate.load() == 2) ::sched_yield();
        return;
    }
    const auto& previous = sig == SIGBUS ? g_alias_copy_bus : g_alias_copy_segv;
    previous.sa_sigaction(sig, info, context);
}

void test_alias_store_pending_clock() {
    const auto mappings = map_shared_pages(1);
    auto* source = static_cast<std::uint8_t*>(map_pages(1));
    source[0] = 0x12;
    source[1] = 0x34;
    source[2] = 0x56;
    source[3] = 0x78;
    memory::WriteTracker::add_range(mappings.primary, mappings.size,
                                    mappings.alias);
    CHECK(memory::WriteTracker::protect_checked(mappings.primary, 4));

    struct sigaction action{};
    action.sa_sigaction = pause_alias_copy;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    CHECK(::sigaction(SIGBUS, &action, &g_alias_copy_bus) == 0);
    CHECK(::sigaction(SIGSEGV, &action, &g_alias_copy_segv) == 0);
    g_alias_copy_source = source;
    g_alias_copy_gate.store(1);
    CHECK(::mprotect(source, page_size(), PROT_NONE) == 0);

    std::atomic<bool> stored{false};
    std::thread writer([&] {
        stored.store(memory::WriteTracker::try_alias_store(
            mappings.primary,
            {reinterpret_cast<const std::byte*>(source), 4}));
    });
    while (g_alias_copy_gate.load() != 2) std::this_thread::yield();
    const auto pending_stamp = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::written_since(mappings.primary, 4,
                                               pending_stamp));
    CHECK(::mprotect(source, page_size(), PROT_READ | PROT_WRITE) == 0);
    g_alias_copy_gate.store(3);
    writer.join();
    CHECK(stored.load());
    CHECK(::sigaction(SIGBUS, &g_alias_copy_bus, nullptr) == 0);
    CHECK(::sigaction(SIGSEGV, &g_alias_copy_segv, nullptr) == 0);
    g_alias_copy_source = nullptr;

    memory::WriteTracker::remove_range(mappings.primary, mappings.alias);
    unmap_pages(source, 1);
    unmap_shared_pages(mappings);
}

void test_writable_alias_store_and_lifetime() {
    const auto mappings = map_shared_pages(2);
    const auto pg = page_size();
    constexpr std::size_t offset = 37;
    auto* logical = mappings.primary + offset;
    auto* alias = mappings.alias + offset;

    // The first same-base owner has no alias. Alias lookup must continue to
    // the second owner, and default retirement must remove only the first.
    memory::WriteTracker::add_range(logical, pg);
    memory::WriteTracker::add_range(logical, pg, alias);
    CHECK(memory::WriteTracker::has_writable_alias(logical, pg));
    CHECK(!memory::WriteTracker::has_writable_alias(logical - 1, 1));
    CHECK(!memory::WriteTracker::has_writable_alias(logical + pg, 1));

    std::array<std::byte, 4> source{
        std::byte{0x12}, std::byte{0x34}, std::byte{0x56}, std::byte{0x78}};
    const auto stamp = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(logical, source.size()));
    CHECK(memory::WriteTracker::try_alias_store(logical + 8, source));
    CHECK(std::memcmp(alias + 8, source.data(), source.size()) == 0);
    CHECK(memory::WriteTracker::written_since(logical, pg, stamp));
    CHECK(protected_store(logical + 8));
    CHECK(memory::WriteTracker::try_alias_store(nullptr, {}));
    CHECK(!memory::WriteTracker::try_alias_store(logical - 1,
                                                  std::span(source).first(1)));
    CHECK(!memory::WriteTracker::try_alias_store(logical + pg,
                                                  std::span(source).first(1)));

    // Alias publication stays coherent while protection and unprotection
    // repeatedly contend for the same canonical pages.
    std::atomic<bool> stop{false};
    std::thread rearmer([&] {
        while (!stop.load()) {
            static_cast<void>(
                memory::WriteTracker::protect_checked(logical, pg));
            memory::WriteTracker::prepare_host_write(logical, pg);
        }
    });
    for (unsigned i = 0; i < 300; i++) {
        const auto concurrent_stamp = memory::WriteTracker::now();
        CHECK(memory::WriteTracker::try_alias_store(logical + 8, source));
        CHECK(memory::WriteTracker::written_since(logical, pg,
                                                   concurrent_stamp));
    }
    stop.store(true);
    rearmer.join();

    memory::WriteTracker::remove_range(logical);
    CHECK(memory::WriteTracker::tracked(logical, pg));
    CHECK(memory::WriteTracker::has_writable_alias(logical, pg));
    memory::WriteTracker::remove_range(logical, alias);
    CHECK(!memory::WriteTracker::tracked(logical, pg));
    unmap_shared_pages(mappings);
}

// A failed protection advances the global generation: a caller that took a
// stamp before the failed call cannot trust a clean verdict afterwards.
std::atomic<int> g_hook_fail_readonly{0};
bool failing_protect_hook(void* addr, std::size_t len, int prot) {
    if (prot == PROT_READ && g_hook_fail_readonly.exchange(0) == 1) {
        errno = EACCES;
        return false;
    }
    return ::mprotect(addr, len, prot) == 0;
}
// Host-write preparation skips spans no registered range touches. It must
// still unprotect a tracked page after another range in its span is removed,
// after its own range is removed and registered again, and at the far end of
// a range that crosses several 1 MiB spans.
void test_prepare_host_write_after_registration_changes() {
    const std::size_t span_pages = (std::size_t{1} << 20) / page_size();
    auto* p = static_cast<std::uint8_t*>(map_pages(2));
    memory::WriteTracker::add_range(p, page_size());
    memory::WriteTracker::add_range(p + page_size(), page_size());
    memory::WriteTracker::remove_range(p);
    CHECK(memory::WriteTracker::protect_checked(p + page_size(), page_size()));
    memory::WriteTracker::prepare_host_write(p + page_size(), 4);
    CHECK(writable(p + page_size()));
    memory::WriteTracker::remove_range(p + page_size());

    memory::WriteTracker::add_range(p, page_size());
    memory::WriteTracker::remove_range(p);
    memory::WriteTracker::add_range(p, page_size());
    CHECK(memory::WriteTracker::protect_checked(p, page_size()));
    memory::WriteTracker::prepare_host_write(p, 4);
    CHECK(writable(p));
    memory::WriteTracker::remove_range(p);
    unmap_pages(p, 2);

    const std::size_t pages = 2 * span_pages + 2;
    auto* wide = static_cast<std::uint8_t*>(map_pages(pages));
    memory::WriteTracker::add_range(wide, pages * page_size());
    // One page in each span the range crosses, the last one included.
    for (std::size_t page = 0; page < pages; page += span_pages / 2) {
        auto* probe = wide + std::min(page, pages - 1) * page_size();
        CHECK(memory::WriteTracker::protect_checked(probe, page_size()));
        memory::WriteTracker::prepare_host_write(probe + 8, 4);
        CHECK(writable(probe));
    }
    auto* last = wide + (pages - 1) * page_size();
    CHECK(memory::WriteTracker::protect_checked(last, page_size()));
    memory::WriteTracker::prepare_host_write(last + 8, 4);
    CHECK(writable(last));
    memory::WriteTracker::remove_range(wide);
    unmap_pages(wide, pages);
}

void test_protect_failure_advances_generation() {
    memory::WriteTracker::set_protect_hook(failing_protect_hook);
    auto* p = static_cast<std::uint8_t*>(map_pages(1));
    memory::WriteTracker::add_range(p, page_size());
    CHECK(memory::WriteTracker::protect_checked(p, page_size()));
    // Already-protected pages issue no mprotect; make them writable again so
    // the failing protection has a batch to lose.
    memory::WriteTracker::prepare_host_write(p, page_size());
    g_hook_fail_readonly = 1;
    const auto before = memory::WriteTracker::now();
    CHECK(!memory::WriteTracker::protect_checked(p, page_size()));
    CHECK(memory::WriteTracker::now() > before);
    // The page stays writable after the failed protection.
    CHECK(writable(p));
    memory::WriteTracker::remove_range(p);
    unmap_pages(p, 1);
    memory::WriteTracker::set_protect_hook(nullptr);
}

std::atomic<bool> g_partial_failure{false};
bool partial_protect_hook(void* addr, std::size_t len, int prot) {
    if (prot == PROT_READ && g_partial_failure.exchange(false)) {
        // Model a kernel changing a prefix before rejecting a later page.
        if (::mprotect(addr, page_size(), PROT_READ) != 0) std::abort();
        errno = EACCES;
        return false;
    }
    return ::mprotect(addr, len, prot) == 0;
}
void test_partial_protection_failure() {
    memory::WriteTracker::set_protect_hook(partial_protect_hook);
    auto* p = static_cast<std::uint8_t*>(map_pages(2));
    memory::WriteTracker::add_range(p, 2 * page_size());
    const auto stamp = memory::WriteTracker::now();
    g_partial_failure.store(true);
    CHECK(!memory::WriteTracker::protect_checked(p, 2 * page_size()));
    CHECK(writable(p));
    CHECK(writable(p + page_size()));
    CHECK(memory::WriteTracker::written_since(p, page_size(), stamp));
    CHECK(memory::WriteTracker::written_since(p + page_size(), page_size(), stamp));
    CHECK(memory::WriteTracker::protect_checked(p, 2 * page_size()));
    CHECK(protected_store(p));
    memory::WriteTracker::remove_range(p);
    unmap_pages(p, 2);
    memory::WriteTracker::set_protect_hook(nullptr);
}

// The ordering race between "set read-only" and "unprotect before a host
// write": while a protection mprotect is delayed in flight, a concurrent
// prepare_host_write must still leave the page really writable, and the
// resulting state must record the write as dirty for stamps taken before.
std::atomic<int> g_race_armed{0}, g_race_go{0}, g_race_done{0};
std::uint8_t* g_race_page = nullptr;
bool racing_protect_hook(void* addr, std::size_t len, int prot) {
    if (prot == PROT_READ && g_race_armed.exchange(0) == 1) {
        g_race_go.store(1);
        ::usleep(100000);   // Keep the read-only mprotect in flight.
    }
    return ::mprotect(addr, len, prot) == 0;
}
void test_prepare_write_race() {
    memory::WriteTracker::set_protect_hook(racing_protect_hook);
    g_race_page = static_cast<std::uint8_t*>(map_pages(2));
    memory::WriteTracker::add_range(g_race_page, 2 * page_size());
    const auto stamp = memory::WriteTracker::now();
    std::thread preparer([] {
        while (!g_race_go.load()) ::usleep(200);
        memory::WriteTracker::prepare_host_write(g_race_page, 16);
        g_race_done.store(1);
    });
    g_race_armed.store(1);
    memory::WriteTracker::protect(g_race_page, 2 * page_size());
    for (int i = 0; i < 2000 && !g_race_done.load(); i++) ::usleep(1000);
    CHECK(g_race_done.load());
    preparer.join();
    // The prepared page must accept a syscall write and a direct store now.
    const int fd = ::open("/dev/zero", O_RDONLY);
    if (fd >= 0) {
        const auto received = ::read(fd, g_race_page, 16);
        CHECK(received == 16);
        ::close(fd);
    } else {
        CHECK(false);
    }
    guest_store(g_race_page + 1);
    CHECK(g_race_page[1] == 0);
    CHECK(memory::WriteTracker::written_since(g_race_page, 16, stamp));
    memory::WriteTracker::remove_range(g_race_page);
    unmap_pages(g_race_page, 2);
    memory::WriteTracker::set_protect_hook(nullptr);
}

std::atomic<int> g_retirement_gate{0};
void* g_retirement_page = nullptr;
bool retiring_protect_hook(void* addr, std::size_t len, int prot) {
    if (addr == g_retirement_page && prot == (PROT_READ | PROT_WRITE) &&
        g_retirement_gate.load() == 1) {
        g_retirement_gate.store(2);
        while (g_retirement_gate.load() == 2) std::this_thread::yield();
    }
    return ::mprotect(addr, len, prot) == 0;
}
void test_retirement_invalidates_cached_index() {
    memory::WriteTracker::set_protect_hook(retiring_protect_hook);
    auto* base = static_cast<std::uint8_t*>(map_pages(3));
    g_retirement_page = base;
    const auto pg = page_size();
    memory::WriteTracker::add_range(base, pg);
    memory::WriteTracker::add_range(base + 2 * pg, pg);
    CHECK(memory::WriteTracker::protect_checked(base, pg));
    g_retirement_gate.store(1);
    std::thread remover([&] { memory::WriteTracker::remove_range(base); });
    while (g_retirement_gate.load() != 2) std::this_thread::yield();
    // Build this thread's index after retirement's first version bump. The
    // index includes the still-published retiring pointer, even though the
    // query itself succeeds against a different live registration.
    CHECK(memory::WriteTracker::tracked(base + 2 * pg, pg));
    g_retirement_gate.store(3);
    remover.join();
    // Retirement must invalidate that index before freeing its entries.
    // AddressSanitizer also checks the otherwise invisible stale dereference.
    CHECK(!memory::WriteTracker::tracked(base, pg));
    CHECK(memory::WriteTracker::tracked(base + 2 * pg, pg));
    memory::WriteTracker::remove_range(base + 2 * pg);
    unmap_pages(base, 3);
    memory::WriteTracker::set_protect_hook(nullptr);
}

// Concurrent registrars, protectors, preparers, and writers stay consistent;
// the survivor of repeated retirements remains tracked and coherent.
void test_concurrency() {
    constexpr std::size_t kPages = 3;
    auto* base = static_cast<std::uint8_t*>(map_pages(kPages));
    const auto pg = page_size();
    memory::WriteTracker::add_range(base, 2 * pg);   // A: pages 0-1, permanent.
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> certified{0}, writes{0};
    auto retire = [&] {
        // Retire only while the writers run: on a loaded machine all 300
        // retirements could otherwise finish before either writer starts.
        while (certified.load() == 0) std::this_thread::yield();
        for (int i = 0; i < 300 && !stop.load(); i++) {
            memory::WriteTracker::add_range(base + pg, 2 * pg);   // B: 1-2.
            static_cast<void>(memory::WriteTracker::protect_checked(base + pg, 2 * pg));
            memory::WriteTracker::remove_range(base + pg);
        }
        stop.store(true);
    };
    auto protect = [&] {
        while (!stop.load()) {
            static_cast<void>(
                memory::WriteTracker::protect_checked(base, 2 * pg));
            static_cast<void>(memory::WriteTracker::written_since(
                base + pg, pg, memory::WriteTracker::now()));
        }
    };
    auto write = [&](int) {
        while (!stop.load()) {
            const auto stamp = memory::WriteTracker::now();
            const bool ok = memory::WriteTracker::protect_checked(base + pg, pg);
            // A covers page 1 throughout, so stores always land in a live owner.
            guest_store(base + pg);
            writes.fetch_add(1);
            if (ok) {
                certified.fetch_add(1);
                CHECK(memory::WriteTracker::written_since(base + pg, pg, stamp));
            }
            memory::WriteTracker::prepare_host_write(base + pg + 3, 1);
            guest_store(base + pg + 3);
            // Page 2 may belong to B or nobody; pre-writes are safe either way.
            memory::WriteTracker::prepare_host_write(base + 2 * pg, pg);
        }
    };
    std::vector<std::thread> threads;
    threads.emplace_back(retire);
    threads.emplace_back(protect);
    threads.emplace_back(write, 0);
    threads.emplace_back(write, 1);
    for (auto& t : threads) t.join();
    CHECK(certified.load() > 0);
    CHECK(writes.load() > 0);
    // A survived every B retirement: its pages remain tracked and re-arming
    // them protects fresh stores.
    const auto stamp = memory::WriteTracker::now();
    CHECK(memory::WriteTracker::protect_checked(base, 2 * pg));
    CHECK(!memory::WriteTracker::written_since(base, 2 * pg, stamp));
    CHECK(protected_store(base + pg));
    CHECK(memory::WriteTracker::written_since(base, 2 * pg, stamp));
    CHECK(writable(base + 2 * pg));
    memory::WriteTracker::remove_range(base);
    unmap_pages(base, kPages);
}

// The handler only resolves write faults on tracked protected pages; an
// untracked fault chains to the previously installed handler. In these child
// processes the previous handler is first a custom pre-write handler and then
// the default disposition.
std::atomic<void*> g_chain_context{nullptr};
// Apple silicon reports a store into a read-only page as SIGBUS; Linux as
// SIGSEGV. The test handler accepts both so the chaining expectation is
// platform independent.
void unprotect_on_fault(int sig, siginfo_t* info, void*) {
    if ((sig == SIGSEGV || sig == SIGBUS) && info &&
        info->si_addr == g_chain_context.load())
        ::mprotect(g_chain_context.load(), page_size(), PROT_READ | PROT_WRITE);
}
void test_chains_to_previous_handler() {
    // Child A: a custom handler installed before the tracker must see the
    // untracked fault and resolve it; the store then completes.
    pid_t custom = ::fork();
    if (custom == 0) {
        ::alarm(10);
        auto* page = static_cast<std::uint8_t*>(map_pages(1));
        ::mprotect(page, page_size(), PROT_READ);
        g_chain_context = page;
        struct sigaction previous{};
        previous.sa_sigaction = unprotect_on_fault;
        previous.sa_flags = SA_SIGINFO;
        sigemptyset(&previous.sa_mask);
        ::sigaction(SIGSEGV, &previous, nullptr);
        ::sigaction(SIGBUS, &previous, nullptr);
        memory::WriteTracker::install();
        page[0] = 42;
        ::_exit(page[0] == 42 ? 0 : 1);
    }
    int status = 0;
    ::waitpid(custom, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);

    // Child B: no previous handler means the default disposition re-raises,
    // so an untracked write fault kills the process with the fault signal.
    pid_t fatal = ::fork();
    if (fatal == 0) {
        ::alarm(10);
        auto* page = static_cast<std::uint8_t*>(map_pages(1));
        ::mprotect(page, page_size(), PROT_READ);
        struct sigaction previous{};
        previous.sa_handler = SIG_DFL;
        sigemptyset(&previous.sa_mask);
        ::sigaction(SIGBUS, &previous, nullptr);
        ::sigaction(SIGSEGV, &previous, nullptr);
        memory::WriteTracker::install();
        static_cast<volatile std::uint8_t&>(page[0]) = 7;
        ::_exit(0);   // Not reached when chaining preserves the default.
    }
    ::waitpid(fatal, &status, 0);
    CHECK(WIFSIGNALED(status) &&
          (WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS));
}

// Delay an actual store fault before the tracker pins its reader bank. This
// models a descheduled faulting thread, not an artificial call to on_fault.
// The mapping stays alive while its last tracking owner is replaced.
std::atomic<int> g_delayed_fault{0};
std::uint8_t* g_delayed_page = nullptr;
struct sigaction g_tracker_bus{}, g_tracker_segv{};
void delay_fault(int sig, siginfo_t* info, void* context) {
    int armed = 1;
    if (info && info->si_addr == g_delayed_page &&
        g_delayed_fault.compare_exchange_strong(armed, 2)) {
        while (g_delayed_fault.load() == 2) ::sched_yield();
    }
    const auto& tracker = sig == SIGBUS ? g_tracker_bus : g_tracker_segv;
    tracker.sa_sigaction(sig, info, context);
}
void fail_delayed_fault(int, siginfo_t*, void*) { ::_exit(91); }

void test_delayed_fault_with_registration_churn() {
    const pid_t child = ::fork();
    if (child == 0) {
        // A chained fault is an explicit failure, without a crash report.
        struct sigaction action{};
        action.sa_sigaction = fail_delayed_fault;
        action.sa_flags = SA_SIGINFO;
        sigemptyset(&action.sa_mask);
        ::sigaction(SIGBUS, &action, nullptr);
        ::sigaction(SIGSEGV, &action, nullptr);
        memory::WriteTracker::install();
        action.sa_sigaction = delay_fault;
        ::sigaction(SIGBUS, &action, &g_tracker_bus);
        ::sigaction(SIGSEGV, &action, &g_tracker_segv);

        const auto pg = page_size();
        auto* base = static_cast<std::uint8_t*>(map_pages(3));
        g_delayed_page = base + 2 * pg;
        memory::WriteTracker::add_range(base, 2 * pg);
        std::atomic<bool> stop{false};
        std::atomic<unsigned> stores{0}, protections{0}, preparations{0};
        std::thread writer([&] {
            while (!stop.load()) {
                guest_store(base);
                guest_store(base + pg);
                stores.fetch_add(1);
            }
        });
        std::thread protector([&] {
            while (!stop.load()) {
                memory::WriteTracker::protect_checked(base, 2 * pg);
                protections.fetch_add(1);
            }
        });
        std::thread preparer([&] {
            while (!stop.load()) {
                memory::WriteTracker::prepare_host_write(base, 2 * pg);
                preparations.fetch_add(1);
            }
        });
        while (!stores.load() || !protections.load() || !preparations.load())
            std::this_thread::yield();
        for (unsigned i = 0; i < 64; ++i) {
            // B shares one page with A and solely owns the faulting page.
            memory::WriteTracker::add_range(base + pg, 2 * pg);
            if (!memory::WriteTracker::protect_checked(g_delayed_page, pg)) ::_exit(92);
            g_delayed_fault.store(1);
            std::thread delayed([] { guest_store(g_delayed_page); });
            while (g_delayed_fault.load() != 2) std::this_thread::yield();
            memory::WriteTracker::remove_range(base + pg);
            const bool replace = (i % 2) == 0;
            if (replace) {
                memory::WriteTracker::add_range(base + pg, 2 * pg);
                memory::WriteTracker::add_range(g_delayed_page + 16, 16);
            }
            // Alternate no live record and a fresh zero-generation record.
            // Both must retry the old fault against now-writable memory.
            g_delayed_fault.store(3);
            delayed.join();
            if (replace) {
                memory::WriteTracker::remove_range(g_delayed_page + 16);
                memory::WriteTracker::remove_range(base + pg);
            }
        }
        stop.store(true);
        writer.join();
        protector.join();
        preparer.join();
        memory::WriteTracker::remove_range(base);
        const auto stats = memory::WriteTracker::stats();
        ::_exit(stats.busy_timeouts == 0 && stats.canonical_pages == 0 ? 0 : 93);
    }
    int status = 0;
    CHECK(child > 0);
    CHECK(::waitpid(child, &status, 0) == child);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        std::fprintf(stderr, "delayed fault child status: %d\n", status);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// Address history is not authority to change permissions. Genuine faults on
// retired, unmapped, freshly tracked, or never-tracked pages still chain,
// as does an alignment fault on a writable retired page.
void test_permission_fallback_chains() {
    for (unsigned scenario = 0; scenario < 5; ++scenario) {
        const pid_t child = ::fork();
        if (child == 0) {
            ::alarm(10);
            struct sigaction previous{};
            previous.sa_sigaction = fail_delayed_fault;
            previous.sa_flags = SA_SIGINFO;
            sigemptyset(&previous.sa_mask);
            ::sigaction(SIGBUS, &previous, nullptr);
            ::sigaction(SIGSEGV, &previous, nullptr);
            auto* p = static_cast<std::uint8_t*>(map_pages(1));
            memory::WriteTracker::install();
            if (scenario != 3) memory::WriteTracker::add_range(p, page_size());
            if (scenario < 2 || scenario == 4) {
                memory::WriteTracker::protect(p, page_size());
                memory::WriteTracker::remove_range(p);
            }
            if (scenario == 4) {
#if defined(__APPLE__) && defined(__aarch64__)
                // Cross a 16-byte boundary so relaxed atomic alignment on
                // newer cores cannot allow the store-release. Assembly avoids
                // a misaligned C++ lvalue.
                asm volatile("stlr %x[value], [%[address]]"
                    : : [value] "r" (std::uint64_t{7}), [address] "r" (p + 15) : "memory");
#else
                ::_exit(91);
#endif
            } else {
                if (scenario == 1) ::munmap(p, page_size());
                else ::mprotect(p, page_size(), PROT_READ);
                __atomic_store_n(p, 7, __ATOMIC_RELAXED);
            }
            ::_exit(94);
        }
        int status = 0;
        CHECK(child > 0);
        CHECK(::waitpid(child, &status, 0) == child);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 91)
            std::fprintf(stderr, "fault chaining scenario %u status: %d\n", scenario, status);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 91);
    }
}

} // namespace

int main() {
    // Chaining must be proven before any other test installs the handlers in
    // this process: the children need the pre-tracker disposition in place.
    test_chains_to_previous_handler();
    test_delayed_fault_with_registration_churn();
    test_permission_fallback_chains();
    const auto pages_before = memory::WriteTracker::stats().live_ranges;
    test_protect_write_and_clean();
    test_shared_records_and_removal();
    test_reregistration_bumps_generation();
    test_note_store_publication();
    test_note_store_across_registrations();
    test_alias_store_pending_clock();
    test_writable_alias_store_and_lifetime();
    test_prepare_host_write_after_registration_changes();
    test_protect_failure_advances_generation();
    test_partial_protection_failure();
    test_prepare_write_race();
    test_retirement_invalidates_cached_index();
    test_concurrency();
    // Every registration was removed; canonical records and owners are gone.
    const auto stats = memory::WriteTracker::stats();
    CHECK(stats.live_ranges == pages_before);
    CHECK(stats.canonical_pages == 0);
    CHECK(stats.shared_references == 0);
    CHECK(stats.busy_timeouts == 0);
    std::printf("write_tracker_tests: %d checks, %d failed\n", g_checks.load(), g_failures.load());
    return g_failures != 0;
}
