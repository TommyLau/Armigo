// ============================================================================
// Armigo Memory - Guest CPU Write Tracking
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Write faults and pre-write hooks track guest CPU writes.
// See write_tracker.h for the tracking invariants.

// The write-fault handler reads the ESR WnR bit from the faulting ucontext;
// macOS keeps those routines behind _XOPEN_SOURCE.
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif

#include <armigo/memory/write_tracker.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <thread>
#include <vector>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <new>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__aarch64__)
#include "darwin_region_info.h"
#include <mach/vm_prot.h>
#include <ucontext.h>
#define ARMIGO_TRACKER_HAS_ESR 1
#endif

namespace armigo::memory {
namespace {

enum PageState : std::uint8_t { PG_NONE = 0, PG_PROT = 1, PG_BUSY = 2 };
constexpr std::uint64_t kBusyWaitMs = 2000;
// The range table provides headroom for workloads with 3500-4100 live pools;
// backing counts per process stay well below the slot limit.
constexpr std::size_t kMaxRanges = 32768;

struct PageChunk;
struct Page {
    std::atomic<std::uint64_t> gen;
    std::atomic<std::uint8_t> state;
    std::uint8_t resume;     // State restored by an alias-store BUSY owner.
    std::uint32_t refs;      // Owning ranges; the slot mutex only.
    PageChunk* chunk;        // Allocation holding this record; slot mutex only.
};
// One allocation per registration for the records it created. A chunk is
// freed once none of its records has an owner, so a shared record outlives
// its creator. The trailing array is sized at allocation time; page[1] makes
// the type complete without a flexible-array extension.
struct PageChunk {
    std::size_t live;
    Page page[1];
};
struct Range {
    std::uintptr_t base, end;
    const void* mem;
    std::uintptr_t mem_end;
    std::byte* writable_alias;
    std::uint64_t registered;
    Page** page;             // Immutable after publication; canonical records.
    std::atomic<int> retiring;
};

std::atomic<Range*> g_ranges[kMaxRanges];
std::atomic<int> g_range_count;
// Readers validate their epoch before loading any slot. A grace period flips
// the epoch and drains only its old bank, so new unrelated readers cannot
// starve it. Sequential consistency closes the load-before-pin race.
std::atomic<unsigned> g_readers[2];
std::atomic<std::uint64_t> g_epoch;
std::atomic<std::uint64_t> g_now = 1;
// Write generations outside NotifiedStores scopes (see unnoted_clock()).
std::atomic<std::uint64_t> g_unnoted{1};
thread_local unsigned t_notified_stores = 0;
// Signal-safe: no thread-local access (guest faults never run in a scope).
inline void advance_unnoted() noexcept { g_unnoted.fetch_add(1); }
inline void advance_unnoted_unless_notified() noexcept {
    if (!t_notified_stores) g_unnoted.fetch_add(1);
}
// Covers the interval in which a host store may be in progress or its final
// page generations may not yet be visible to a dirty query.
std::atomic<unsigned> g_publications_active;
std::atomic<std::uint64_t> g_faults, g_prepared, g_adds_total;
std::atomic<std::uint64_t> g_protect_calls, g_protect_pages, g_protect_ticks;
std::atomic<std::uint64_t> g_busy_timeouts;
std::atomic<int> g_live, g_live_peak;
std::size_t g_pages_live, g_pages_shared;   // Slot mutex only.
std::atomic<std::uintptr_t> g_page_size;
std::uintptr_t g_page_shift;                // Install-time constant afterwards.
std::atomic<bool> g_installed;
std::mutex g_slot_mutex;                    // Registration and retirement only.

// A fault can be descheduled before read_enter(), so reader grace periods
// cannot retire its address identity. Keep the union of retired addresses,
// not their page records. Immutable snapshots are reclaimed by the existing
// retirement grace period. Repeated registration churn does not grow the
// union; disjoint address intervals remain until process exit. History alone
// never handles a fault: the kernel must also report the address writable.
struct RetiredSpan { std::uintptr_t base, end; };
using RetiredSpans = std::vector<RetiredSpan>;
std::atomic<RetiredSpans*> g_retired_spans;

struct sigaction g_previous_segv, g_previous_bus;

// Tests only; never mutated while tracked pages are protected.
std::atomic<WriteTracker::ProtectHook> g_protect_hook;

static_assert(std::atomic<std::uint8_t>::is_always_lock_free &&
                  std::atomic<unsigned>::is_always_lock_free &&
                  std::atomic<std::uint64_t>::is_always_lock_free,
              "write-tracker signal readers require lock-free atomics");

std::uint64_t now_ns() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ull +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

unsigned read_enter() noexcept {
    for (;;) {
        const std::uint64_t e = g_epoch.load();
        const unsigned bank = static_cast<unsigned>(e & 1);
        g_readers[bank].fetch_add(1);
        if (g_epoch.load() == e) return bank;
        g_readers[bank].fetch_sub(1);
    }
}
void read_leave(unsigned bank) noexcept { g_readers[bank].fetch_sub(1); }
void drain_readers() noexcept {
    const unsigned bank = static_cast<unsigned>(g_epoch.fetch_add(1) & 1);
    while (g_readers[bank].load()) sched_yield();
}

// Slot mutex held. Publish before unpublishing the corresponding live range.
// The caller deletes the old snapshot only after draining readers.
RetiredSpans* remember_retired(const Range* r) {
    RetiredSpans* old = g_retired_spans.load();
    if (old) {
        const auto covering = std::upper_bound(old->begin(), old->end(), r->base,
            [](std::uintptr_t base, const RetiredSpan& span) { return base < span.base; });
        if (covering != old->begin() && (covering - 1)->end >= r->end) return nullptr;
    }
    auto next = old ? *old : RetiredSpans{};
    auto first = std::lower_bound(next.begin(), next.end(), r->base,
        [](const RetiredSpan& span, std::uintptr_t base) { return span.end < base; });
    RetiredSpan merged{r->base, r->end};
    auto last = first;
    while (last != next.end() && last->base <= merged.end) {
        merged.base = std::min(merged.base, last->base);
        merged.end = std::max(merged.end, last->end);
        ++last;
    }
    first = next.erase(first, last);
    next.insert(first, merged);
    g_retired_spans.store(new RetiredSpans(std::move(next)));
    return old;
}

bool was_retired(std::uintptr_t addr) noexcept {
    const RetiredSpans* spans = g_retired_spans.load();
    if (!spans) return false;
    const auto it = std::upper_bound(spans->begin(), spans->end(), addr,
        [](std::uintptr_t a, const RetiredSpan& span) { return a < span.base; });
    return it != spans->begin() && addr < (it - 1)->end;
}

bool page_writable(std::uintptr_t addr) noexcept {
#if defined(ARMIGO_TRACKER_HAS_ESR)
    // Darwin's proc_pidinfo is a thin proc_info syscall wrapper: no allocation,
    // userspace locks, or shared MIG reply port. Use only a stack result and
    // preserve errno. Installation resolves this call before signals can use it.
    detail::NativeRegionInfo region{};
    const int saved_errno = errno;
    const int size = proc_pidinfo(getpid(), detail::native_region_info_flavor, addr,
                                 &region, sizeof(region));
    errno = saved_errno;
    return size == sizeof(region) && addr >= region.address &&
           addr - region.address < region.size &&
           (region.protection & (VM_PROT_READ | VM_PROT_WRITE)) ==
               (VM_PROT_READ | VM_PROT_WRITE);
#else
    static_cast<void>(addr);
    return false;
#endif
}

bool covers(const Range* range, std::uintptr_t a, std::size_t n) noexcept {
    return range && n && a >= range->base && a < range->end &&
           n <= static_cast<std::size_t>(range->end - a);
}
bool covers_original(const Range* range, std::uintptr_t a,
                     std::size_t n) noexcept {
    if (!range) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(range->mem);
    return n && a >= begin && a < range->mem_end &&
           n <= static_cast<std::size_t>(range->mem_end - a);
}
// Any owner of a host page resolves to its canonical record, so the fault and
// pre-write paths may use the first range containing the address.
Range* find_range(std::uintptr_t a, bool allow_retiring) noexcept {
    const int n = g_range_count.load();
    for (int i = 0; i < n; i++) {
        Range* r = g_ranges[i].load();
        if (r && a >= r->base && a < r->end &&
            (allow_retiring || !r->retiring.load())) return r;
    }
    return nullptr;
}
// Full-span queries need a range covering the whole span; a shorter
// overlapping range in an earlier slot must not hide a covering one. Only
// ranges registered no later than max_registered qualify.
// Bumped before any range record is freed: a thread's cached last hit is
// valid only for the version it was found under.
std::atomic<std::uint64_t> g_ranges_version{1};
Range* find_cover(std::uintptr_t a, std::size_t len,
                  std::uint64_t max_registered) noexcept {
    // Queries cluster on a few backings: try this thread's hit for the
    // same 1 MiB host span before scanning every registered range.
    constexpr std::size_t slots = 256;
    thread_local std::array<Range*, slots> recent{};
    thread_local std::uint64_t recent_version = 0;
    const auto version = g_ranges_version.load();
    if (recent_version != version) {
        recent.fill(nullptr);
        recent_version = version;
    }
    Range*& hit = recent[(a >> 20) & (slots - 1)];
    if (covers(hit, a, len) && !hit->retiring.load() && hit->registered <= max_registered) return hit;
    // This thread's index of the registered ranges sorted by base, with the
    // running maximum end, replaces a scan of every slot; it is rebuilt when
    // a registration or retirement changes the version.
    struct Indexed { std::uintptr_t base, max_end; Range* range; };
    thread_local std::vector<Indexed> index;
    thread_local std::uint64_t index_version = 0;
    if (index_version != version) {
        index.clear();
        const int n = g_range_count.load();
        try {
            index.reserve(std::size_t(n));
        } catch (...) {
            // Without an index, scan every slot as before.
            for (int i = 0; i < n; i++) {
                Range* r = g_ranges[i].load();
                if (covers(r, a, len) && !r->retiring.load() && r->registered <= max_registered) return r;
            }
            return nullptr;
        }
        for (int i = 0; i < n; i++)
            if (Range* r = g_ranges[i].load()) index.push_back({r->base, r->end, r});
        std::sort(index.begin(), index.end(), [](const Indexed& x, const Indexed& y) { return x.base < y.base; });
        std::uintptr_t running = 0;
        for (auto& entry : index) {
            running = std::max(running, entry.range->end);
            entry.max_end = running;
        }
        index_version = version;
    }
    auto it = std::upper_bound(index.begin(), index.end(), a,
                               [](std::uintptr_t value, const Indexed& entry) { return value < entry.base; });
    while (it != index.begin()) {
        --it;
        if (it->max_end <= a || it->max_end - a < len) break;
        Range* r = it->range;
        if (covers(r, a, len) && !r->retiring.load() && r->registered <= max_registered) {
            hit = r;
            return r;
        }
    }
    return nullptr;
}

Range* find_alias_cover(std::uintptr_t a, std::size_t len) noexcept {
    constexpr std::size_t slots = 256;
    thread_local std::array<Range*, slots> recent{};
    thread_local std::uint64_t recent_version = 0;
    const auto version = g_ranges_version.load();
    if (recent_version != version) {
        recent.fill(nullptr);
        recent_version = version;
    }
    Range*& hit = recent[(a >> 20) & (slots - 1)];
    if (hit && hit->writable_alias && covers_original(hit, a, len) &&
        !hit->retiring.load())
        return hit;

    // Reuse the ordinary covering cache when it selects the alias owner. A
    // rounded, shorter, or non-alias registration may win that lookup, so a
    // miss still scans for another exact owner.
    Range* candidate = find_cover(a, len, ~0ull);
    if (candidate && candidate->writable_alias &&
        covers_original(candidate, a, len)) {
        hit = candidate;
        return candidate;
    }
    const int n = g_range_count.load();
    for (int i = 0; i < n; i++) {
        Range* r = g_ranges[i].load();
        if (r && r->writable_alias && covers_original(r, a, len) &&
            !r->retiring.load()) {
            hit = r;
            return r;
        }
    }
    return nullptr;
}
std::size_t page_index(const Range* r, std::uintptr_t a) noexcept {
    return static_cast<std::size_t>((a - r->base) >> g_page_shift);
}

bool protect_pages(std::uintptr_t a, std::size_t len, int prot) noexcept {
    const auto hook = g_protect_hook.load(std::memory_order_relaxed);
    std::uint64_t t = now_ns();
    const bool ok = hook ? hook(reinterpret_cast<void*>(a), len, prot)
                         : mprotect(reinterpret_cast<void*>(a), len, prot) == 0;
    g_protect_ticks.fetch_add(now_ns() - t);
    g_protect_calls.fetch_add(1);
    g_protect_pages.fetch_add(len >> g_page_shift);
    return ok;
}

// Newest page generation per hashed 1 MiB host span. Collisions only merge
// spans, so a summary at or below a stamp proves every page in the span is.
constexpr unsigned kSpanShift = 20;
constexpr std::size_t kSpanSlots = std::size_t{1} << 16;
std::atomic<std::uint64_t> g_span_gen[kSpanSlots];
std::atomic<std::uint64_t>& span_gen(std::uintptr_t a) noexcept {
    return g_span_gen[(a >> kSpanShift) & (kSpanSlots - 1)];
}

// Registered ranges, live or retiring, per hashed 1 MiB host span. A range is
// counted before it is published and uncounted only after no reader can find
// it, so a zero proves that no range covers any byte of the span. Collisions
// only merge spans.
std::atomic<std::uint32_t> g_span_ranges[kSpanSlots];
void count_range_spans(std::uintptr_t base, std::uintptr_t end, bool add) noexcept {
    const std::uintptr_t first = base >> kSpanShift, last = (end - 1) >> kSpanShift;
    // A range over every slot counts each slot once.
    const std::uintptr_t spans = last - first >= kSpanSlots ? kSpanSlots : last - first + 1;
    for (std::uintptr_t span = first; span != first + spans; ++span) {
        auto& count = g_span_ranges[span & (kSpanSlots - 1)];
        if (add) count.fetch_add(1);
        else count.fetch_sub(1);
    }
}
bool spans_without_ranges(std::uintptr_t a, std::uintptr_t end) noexcept {
    const std::uintptr_t first = a >> kSpanShift, last = (end - 1) >> kSpanShift;
    if (last - first >= kSpanSlots) return false;
    for (std::uintptr_t span = first; span <= last; ++span)
        if (g_span_ranges[span & (kSpanSlots - 1)].load()) return false;
    return true;
}

void raise_generation(std::atomic<std::uint64_t>& target,
                      std::uint64_t value) noexcept {
    for (std::uint64_t seen = target.load();
         seen < value && !target.compare_exchange_weak(seen, value);) {}
}

void publish_page_store(Page* pg, std::uintptr_t addr,
                        std::uint64_t generation) noexcept {
    raise_generation(span_gen(addr), generation);
    raise_generation(pg->gen, generation);
}

bool busy_wait(Page* pg) noexcept {
    const std::uint64_t t0 = now_ns();
    for (unsigned n = 1; pg->state.load() == PG_BUSY; n++) {
        if (n & 0x3ff) continue;
        if (now_ns() - t0 > kBusyWaitMs * 1'000'000ull) {
            // Formatting and stdio are forbidden in the signal path.
            g_busy_timeouts.fetch_add(1);
            return false;
        }
        sched_yield();
    }
    return true;
}
// Caller owns PG_BUSY. Never publish writable state if mprotect failed.
bool unprotect_page(Page* pg, std::uintptr_t addr) noexcept {
    auto& span = span_gen(addr);
    // Stores may land as soon as the page is writable, before its generation
    // is bumped: first advance the clock and lift the span above every stamp
    // taken so far, so neither the unchanged-clock exit nor a span check can
    // skip this BUSY page.
    advance_unnoted();
    raise_generation(span, g_now.fetch_add(1) + 1);
    if (!protect_pages(addr, g_page_size.load(std::memory_order_relaxed),
                       PROT_READ | PROT_WRITE)) {
        pg->state.store(PG_PROT);
        return false;
    }
    const std::uint64_t gen = g_now.fetch_add(1) + 1;
    raise_generation(span, gen);
    raise_generation(pg->gen, gen);
    pg->state.store(PG_NONE);
    return true;
}
bool unprotect_one(Range* r, std::size_t p) noexcept {
    Page* pg = r->page[p];
    for (;;) {
        std::uint8_t st = PG_PROT;
        if (pg->state.compare_exchange_strong(st, PG_BUSY)) {
            if (!unprotect_page(pg, r->base + (p << g_page_shift))) return false;
            g_prepared.fetch_add(1);
            return true;
        }
        if (st != PG_BUSY) return true;
        if (!busy_wait(pg)) return false;
    }
}
bool unprotect_range(Range* r, std::uintptr_t a, std::uintptr_t stop) noexcept {
    bool ok = true;
    for (std::size_t p = page_index(r, a), q = page_index(r, stop - 1); p <= q; p++)
        if (!unprotect_one(r, p)) ok = false;
    return ok;
}

// Signal context. Only table lookups, atomics, and mprotect happen here.
bool on_fault(void* addr) noexcept;
void fault_handler(int sig, siginfo_t* info, void* context) noexcept;

bool on_fault(void* addr) noexcept {
    const unsigned bank = read_enter();
    const auto a = reinterpret_cast<std::uintptr_t>(addr);
    Range* r = find_range(a, true);
    bool handled = false;
    if (!r && was_retired(a)) {
        handled = page_writable(a);
        // Registration/protection may have raced the kernel query.
        if (!handled) r = find_range(a, true);
    }
    if (r) {
        const std::size_t p = page_index(r, a);
        Page* pg = r->page[p];
        std::uint8_t st = PG_PROT;
        if (pg->state.compare_exchange_strong(st, PG_BUSY)) {
            handled = unprotect_page(pg, r->base + (p << g_page_shift));
            if (handled) g_faults.fetch_add(1);
        } else if (st == PG_BUSY) {
            handled = busy_wait(pg) || page_writable(a);
        } else {
            // An old fault can reach a freshly registered writable record
            // with gen == 0. Generation is write history, not permission.
            // Nonzero generations are published before PG_NONE after RW
            // mprotect. Fresh records require a kernel permission check.
            handled = pg->gen.load() != 0 || page_writable(a);
            // A protector may have changed NONE while the query ran; retry
            // against that transition rather than chaining a tracker fault.
            if (!handled) handled = pg->state.load() != PG_NONE || pg->gen.load() != 0;
        }
    }
    read_leave(bank);
    return handled;
}

bool guest_write_fault(void* context) noexcept {
#if defined(ARMIGO_TRACKER_HAS_ESR)
    // Only data-abort translation/access/permission write faults belong here. In
    // particular, a writable retired address must not hide alignment faults.
    auto* uc = static_cast<ucontext_t*>(context);
    const auto esr = uc->uc_mcontext->__es.__esr;
    return ((esr >> 26) & 0x3f) == 0x24 && (esr & (1u << 6)) &&
           (esr & 0x3f) >= 0x04 && (esr & 0x3f) <= 0x0f;
#else
    static_cast<void>(context);
    return false;   // Without a write bit we must not swallow the fault.
#endif
}

void chain_previous(int sig, siginfo_t* info, void* context) noexcept {
    const struct sigaction& previous =
        sig == SIGBUS ? g_previous_bus : g_previous_segv;
    // The disposition comes first: a default action inherited across exec
    // still reports SA_SIGINFO, and returning would repeat the fault forever.
    if ((previous.sa_flags & SA_SIGINFO) && previous.sa_handler != SIG_DFL &&
        previous.sa_handler != SIG_IGN) {
        previous.sa_sigaction(sig, info, context);
        return;
    }
    if (previous.sa_handler == SIG_DFL) {
        // Restore the default disposition and re-raise. The signal stays
        // blocked until this handler returns, so delivery happens with the
        // default disposition and the process dies; nothing here re-arms.
        struct sigaction reset{};
        reset.sa_handler = SIG_DFL;
        sigemptyset(&reset.sa_mask);
        sigaction(sig, &reset, nullptr);
        raise(sig);
        return;
    }
    if (previous.sa_handler == SIG_IGN) return;
    previous.sa_handler(sig);
}

void fault_handler(int sig, siginfo_t* info, void* context) noexcept {
    if ((sig == SIGSEGV || sig == SIGBUS) && info && info->si_code > 0 &&
        guest_write_fault(context) && on_fault(info->si_addr)) return;
    chain_previous(sig, info, context);
}

// Slot mutex held. Adopt canonical records from live overlapping peers, then
// allocate fresh writable records for pages no live range covers.
void attach_pages(Range* r, int n) {
    const std::size_t count = static_cast<std::size_t>(r->end - r->base) >> g_page_shift;
    std::size_t fresh = 0;
    for (int i = 0; i < n; i++) {
        Range* peer = g_ranges[i].load();
        if (!peer || r->base >= peer->end || peer->base >= r->end) continue;
        const std::uintptr_t lo = r->base > peer->base ? r->base : peer->base;
        const std::uintptr_t hi = r->end < peer->end ? r->end : peer->end;
        for (std::uintptr_t a = lo; a < hi;
             a += g_page_size.load(std::memory_order_relaxed)) {
            Page* pg = peer->page[page_index(peer, a)];
            Page** slot = &r->page[page_index(r, a)];
            // Two records for one host page would break the canonical-state
            // invariant; it cannot happen because adoption precedes fresh
            // allocation under the slot mutex.
            if (*slot && *slot != pg) std::abort();
            *slot = pg;
        }
    }
    for (std::size_t i = 0; i < count; i++)
        if (!r->page[i]) fresh++;
    PageChunk* chunk = nullptr;
    if (fresh) {
        chunk = static_cast<PageChunk*>(
            ::operator new(sizeof(PageChunk) + (fresh - 1) * sizeof(Page)));
        chunk->live = fresh;
        for (std::size_t i = 0; i < fresh; i++)
            new (&chunk->page[i]) Page{};
    }
    for (std::size_t i = 0, k = 0; i < count; i++) {
        if (r->page[i]) {
            r->page[i]->refs++;
            g_pages_shared++;
            continue;
        }
        Page* pg = &chunk->page[k++];
        pg->chunk = chunk;
        pg->refs = 1;
        r->page[i] = pg;
    }
    g_pages_live += fresh;
}
// Slot mutex held and r unpublished with readers drained.
void detach_pages(Range* r) noexcept {
    const std::size_t count = static_cast<std::size_t>(r->end - r->base) >> g_page_shift;
    for (std::size_t i = 0; i < count; i++) {
        Page* pg = r->page[i];
        if (--pg->refs) {
            g_pages_shared--;
            continue;
        }
        g_pages_live--;
        PageChunk* c = pg->chunk;
        if (!--c->live) ::operator delete(c);
    }
    ::operator delete(r->page);
}

} // namespace

namespace {
// Idempotent handler installation with an acquire fast path safe for hot
// pre-write callers. All page-size arithmetic runs only after this returns.
void ensure_installed() noexcept {
    if (g_installed.load(std::memory_order_acquire)) return;
    WriteTracker::install();
}
} // namespace

void WriteTracker::install() {
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    if (g_installed.load()) return;
    if (!g_page_size.load()) {
        const auto pagesize = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
        g_page_shift = __builtin_ctzl(pagesize);
        g_page_size.store(pagesize, std::memory_order_release);
    }
    static_cast<void>(page_writable(reinterpret_cast<std::uintptr_t>(&mutex)));
    // Only a write fault on a tracked protected page is handled; everything
    // else chains to the handlers that were installed before us.
    struct sigaction action{};
    action.sa_sigaction = fault_handler;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, &g_previous_segv);
    sigaction(SIGBUS, &action, &g_previous_bus);
    g_installed.store(true);
}

bool WriteTracker::installed() noexcept { return g_installed.load(); }

void WriteTracker::add_range(const void* mem, std::size_t size,
                             void* writable_alias) noexcept {
    ensure_installed();
    const auto a = reinterpret_cast<std::uintptr_t>(mem);
    const std::uintptr_t page = g_page_size.load(std::memory_order_relaxed);
    if (!mem || !size || size > static_cast<std::size_t>(-1) - a ||
        a + size > static_cast<std::uintptr_t>(-1) - (page - 1))
        return;
    g_adds_total.fetch_add(1);
    std::lock_guard lock(g_slot_mutex);
    int n = g_range_count.load(), slot = -1;
    for (int i = 0; i < n; i++)
        if (!g_ranges[i].load()) {
            slot = i;
            break;
        }
    if (slot < 0) {
        if (n >= static_cast<int>(kMaxRanges)) return;   // Conservatively untracked.
        slot = n;
    }
    Range* r = new (std::nothrow) Range{};
    if (!r) std::abort();
    r->base = a & ~(page - 1);
    r->end = (a + size + page - 1) & ~(page - 1);
    r->mem = mem;
    r->mem_end = a + size;
    r->writable_alias = static_cast<std::byte*>(writable_alias);
    const std::size_t count = static_cast<std::size_t>(r->end - r->base) >> g_page_shift;
    r->page = static_cast<Page**>(
        ::operator new[](count * sizeof(Page*), std::nothrow));
    if (!r->page) std::abort();
    for (std::size_t i = 0; i < count; i++) r->page[i] = nullptr;
    attach_pages(r, n);
    // Re-registration must invalidate stamps belonging to the previous storage.
    advance_unnoted();
    r->registered = g_now.fetch_add(1) + 1;
    count_range_spans(r->base, r->end, true);
    g_ranges[slot].store(r);
    if (slot == n) g_range_count.store(n + 1);
    g_ranges_version.fetch_add(1);
    const int live = g_live.fetch_add(1) + 1;
    int peak = g_live_peak.load();
    while (live > peak && !g_live_peak.compare_exchange_weak(peak, live)) {}
}

void WriteTracker::remove_range(const void* mem, void* writable_alias) noexcept {
    std::lock_guard lock(g_slot_mutex);
    const int n = g_range_count.load();
    for (int i = 0; i < n; i++) {
        Range* r = g_ranges[i].load();
        if (!r || mem != r->mem || writable_alias != r->writable_alias) continue;
        // Stop new protect/query readers first. Faults and pre-write
        // preparation remain admitted until every solely owned page has
        // become writable.
        r->retiring.store(1);
        g_ranges_version.fetch_add(1);
        drain_readers();
        // Ownership cannot change while the slot mutex is held, and no other
        // range can reach a solely owned record, so nothing can re-protect it
        // from here on.
        const std::size_t count = static_cast<std::size_t>(r->end - r->base) >> g_page_shift;
        for (std::size_t p = 0; p < count; p++)
            if (r->page[p]->refs == 1 && !unprotect_one(r, p)) std::abort();
        RetiredSpans* old_spans = remember_retired(r);
        g_ranges[i].store(nullptr);
        // Readers admitted after the first version bump could cache this
        // retiring range in their index. Invalidate those pointers as well.
        g_ranges_version.fetch_add(1);
        drain_readers();
        count_range_spans(r->base, r->end, false);
        delete old_spans;
        detach_pages(r);
        advance_unnoted();
        g_now.fetch_add(1);
        g_live.fetch_sub(1);
        delete r;
        break;
    }
}

bool WriteTracker::tracked(const void* addr, std::size_t len) noexcept {
    const unsigned bank = read_enter();
    const bool ok = find_cover(reinterpret_cast<std::uintptr_t>(addr), len,
                               ~0ull) != nullptr;
    read_leave(bank);
    return ok;
}

bool WriteTracker::has_writable_alias(const void* addr,
                                      std::size_t len) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(addr);
    if (!len || len > static_cast<std::size_t>(-1) - a) return false;
    const unsigned bank = read_enter();
    const bool ok = find_alias_cover(a, len) != nullptr;
    read_leave(bank);
    return ok;
}

bool WriteTracker::try_alias_store(
    const void* addr, std::span<const std::byte> source) noexcept {
    if (source.empty()) return true;
    const auto a = reinterpret_cast<std::uintptr_t>(addr);
    if (!addr || source.size() > static_cast<std::size_t>(-1) - a)
        return false;
    const unsigned bank = read_enter();
    Range* r = find_alias_cover(a, source.size());
    if (!r) {
        read_leave(bank);
        return false;
    }
    g_publications_active.fetch_add(1);
    const std::uintptr_t end = a + source.size();
    const std::size_t first = page_index(r, a);
    const std::size_t last = page_index(r, end - 1);
    std::size_t acquired = first;
    for (; acquired <= last; acquired++) {
        Page* pg = r->page[acquired];
        for (;;) {
            std::uint8_t state = pg->state.load();
            if (state == PG_BUSY) {
                if (busy_wait(pg)) continue;
                for (std::size_t i = first; i < acquired; i++)
                    r->page[i]->state.store(r->page[i]->resume);
                g_publications_active.fetch_sub(1);
                read_leave(bank);
                return false;
            }
            if (pg->state.compare_exchange_strong(state, PG_BUSY)) {
                pg->resume = state;
                break;
            }
        }
    }

    // Every touched page is BUSY before the pending clock becomes visible.
    // The publication counter also covers stamps taken at that pending clock,
    // while page/span generations still await the copy's completion.
    advance_unnoted_unless_notified();
    const std::uint64_t pending = g_now.fetch_add(1) + 1;
    for (std::size_t i = first; i <= last; i++)
        raise_generation(span_gen(r->base + (i << g_page_shift)), pending);

    const auto logical = reinterpret_cast<std::uintptr_t>(r->mem);
    const std::uintptr_t page_size = g_page_size.load(std::memory_order_relaxed);
    for (std::size_t i = first; i <= last; i++) {
        const std::uintptr_t page_begin = r->base + (i << g_page_shift);
        const std::uintptr_t copy_begin = std::max(a, page_begin);
        const std::uintptr_t copy_end = std::min(end, page_begin + page_size);
        const std::size_t source_offset = static_cast<std::size_t>(copy_begin - a);
        const std::size_t alias_offset = static_cast<std::size_t>(copy_begin - logical);
        std::memcpy(r->writable_alias + alias_offset,
                    source.data() + source_offset, copy_end - copy_begin);
        const std::uint64_t generation = g_now.fetch_add(1) + 1;
        Page* pg = r->page[i];
        publish_page_store(pg, page_begin, generation);
        pg->state.store(pg->resume);
    }
    g_publications_active.fetch_sub(1);
    read_leave(bank);
    return true;
}

void WriteTracker::note_store(const void* addr, std::size_t len) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(addr);
    if (!len || len > static_cast<std::size_t>(-1) - a ||
        !g_installed.load(std::memory_order_acquire))
        return;
    const std::uintptr_t page = g_page_size.load(std::memory_order_relaxed);
    const std::uintptr_t first = a & ~(page - 1);
    const std::uintptr_t last = (a + len - 1) & ~(page - 1);
    const unsigned bank = read_enter();
    // Every owner of a host page resolves to its canonical record, so a range
    // found for one page serves the pages after it that it also covers: a
    // store over a large pool scans the registrations once, not per page.
    Range* owner = nullptr;
    const auto owner_of = [&](std::uintptr_t at) {
        if (!owner || at < owner->base || at >= owner->end) owner = find_range(at, true);
        return owner;
    };
    bool intersects = false;
    for (std::uintptr_t at = first;; at += page) {
        if (owner_of(at)) {
            intersects = true;
            break;
        }
        if (at == last) break;
    }
    if (intersects) {
        g_publications_active.fetch_add(1);
        advance_unnoted_unless_notified();
        const std::uint64_t generation = g_now.fetch_add(1) + 1;
        for (std::uintptr_t at = first;; at += page) {
            if (Range* r = owner_of(at))
                publish_page_store(r->page[page_index(r, at)], at,
                                   generation);
            if (at == last) break;
        }
        g_publications_active.fetch_sub(1);
    }
    read_leave(bank);
}

bool WriteTracker::protect_checked(const void* addr, std::size_t len) noexcept {
    ensure_installed();
    const unsigned bank = read_enter();
    Range* r = find_cover(reinterpret_cast<std::uintptr_t>(addr), len, ~0ull);
    bool ok = r != nullptr;
    if (ok) {
        const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(addr);
        const std::size_t p = page_index(r, begin);
        const std::size_t q = page_index(r, begin + len - 1);
        std::size_t run = 0;
        bool open = false;
        for (std::size_t i = p; i <= q + 1; i++) {
            std::uint8_t st = PG_NONE;
            bool took = i <= q && r->page[i]->state.compare_exchange_strong(st, PG_BUSY);
            // A BUSY page is in another thread's fault handler, which bumps
            // its write generation and then publishes it writable. Wait for
            // that, then protect it: giving up would leave a writable page
            // whose later stores no stamp observes.
            for (unsigned n = 1; i <= q && !took && st == PG_BUSY && n < (1u << 22); n++) {
                if (n % 64 == 0) std::this_thread::yield();
                st = PG_NONE;
                took = r->page[i]->state.compare_exchange_strong(st, PG_BUSY);
            }
            if (took && !open) {
                run = i;
                open = true;
            }
            if (!took && open) {
                const bool protected_ok =
                    protect_pages(r->base + (run << g_page_shift), (i - run) << g_page_shift,
                                  PROT_READ);
                for (std::size_t k = run; k < i; k++) {
                    if (protected_ok) {
                        r->page[k]->state.store(PG_PROT);
                    } else {
                        // A failed batch may already have changed a prefix.
                        // Retain BUSY until each page is restored writable and
                        // its generation published. An unrecoverable rollback
                        // cannot safely leave permission metadata behind.
                        if (!unprotect_page(r->page[k], r->base + (k << g_page_shift)))
                            std::abort();
                    }
                }
                if (!protected_ok) ok = false;
                open = false;
            }
            // A BUSY page may be becoming writable. Skipping it cannot prove
            // this range cacheable, even when its write generation is old.
            if (i <= q && !took && st == PG_BUSY) ok = false;
        }
    }
    read_leave(bank);
    // Force caches that took a stamp before this call to retry when this
    // protection could not be established.
    if (!ok) {
        advance_unnoted();
        g_now.fetch_add(1);
    }
    return ok;
}

void WriteTracker::protect(const void* addr, std::size_t len) noexcept {
    static_cast<void>(protect_checked(addr, len));
}

std::uint64_t WriteTracker::now() noexcept { return g_now.load(); }
std::uint64_t WriteTracker::unnoted_clock() noexcept { return g_unnoted.load(); }
WriteTracker::NotifiedStores::NotifiedStores() noexcept { ++t_notified_stores; }
WriteTracker::NotifiedStores::~NotifiedStores() { --t_notified_stores; }

bool WriteTracker::written_since(const void* addr, std::size_t len,
                                 std::uint64_t stamp) noexcept {
    if (!len) return false;
    if (!stamp) return true;
    const unsigned bank = read_enter();
    // Sample publication first: if it is active, its pending clock may equal
    // stamp even though bytes are moving or final generations are unpublished.
    const bool publication_active = g_publications_active.load() != 0;
    // Any covering owner registered by the stamp has held these canonical
    // records continuously since then; a later owner alone cannot vouch for
    // older storage.
    Range* r = find_cover(reinterpret_cast<std::uintptr_t>(addr), len, stamp);
    bool dirty = !r || publication_active;
    // No page fault, registration or retirement since the stamp: every page
    // generation is still at or below it, so the walk cannot find a write.
    if (!dirty && g_now.load() != stamp) {
        const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(addr);
        const std::size_t p = page_index(r, begin);
        const std::size_t q = page_index(r, begin + len - 1);
        constexpr std::uintptr_t span_mask = (std::uintptr_t{1} << kSpanShift) - 1;
        for (std::size_t i = p; i <= q; i++) {
            const std::uintptr_t at = r->base + (i << g_page_shift);
            if ((i == p || !(at & span_mask)) && span_gen(at).load() <= stamp) {
                // Resume at the first page of the next span.
                i = (((at | span_mask) + 1 - r->base) >> g_page_shift) - 1;
                continue;
            }
            const Page* pg = r->page[i];
            if (pg->state.load() == PG_BUSY || pg->gen.load() > stamp) {
                dirty = true;
                break;
            }
        }
    }
    read_leave(bank);
    return dirty;
}

void WriteTracker::prepare_host_write(const void* addr, std::size_t len) noexcept {
    auto a = reinterpret_cast<std::uintptr_t>(addr);
    if (!len || len > static_cast<std::size_t>(-1) - a) return;
    ensure_installed();
    const std::uintptr_t page = g_page_size.load(std::memory_order_relaxed);
    const std::uintptr_t end = a + len;
    // Most host writes (guest synchronization words) touch no registered
    // range: answer those without the reader epoch or a scan of every range.
    if (spans_without_ranges(a, end)) return;
    const unsigned bank = read_enter();
    while (a < end) {
        Range* r = find_range(a, true);
        if (!r) {
            const std::uintptr_t step = page - (a & (page - 1));
            if (step >= end - a) break;
            a += step;
            continue;
        }
        const std::uintptr_t stop = end < r->end ? end : r->end;
        unprotect_range(r, a, stop);
        a = stop;
    }
    read_leave(bank);
}

WriteTrackerStats WriteTracker::stats() noexcept {
    WriteTrackerStats s;
    s.faults = g_faults.load();
    s.prepared = g_prepared.load();
    s.protect_calls = g_protect_calls.load();
    s.protect_pages = g_protect_pages.load();
    s.protect_ns = g_protect_ticks.load();
    s.busy_timeouts = g_busy_timeouts.load();
    s.live_ranges = static_cast<std::uint64_t>(g_live.load());
    s.live_peak = static_cast<std::uint64_t>(g_live_peak.load());
    s.canonical_pages = g_pages_live;
    s.shared_references = g_pages_shared;
    return s;
}

void WriteTracker::set_protect_hook(ProtectHook hook) noexcept {
    g_protect_hook.store(hook, std::memory_order_relaxed);
}

} // namespace armigo::memory
