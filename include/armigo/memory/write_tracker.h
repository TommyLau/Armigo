// ============================================================================
// Armigo Memory - Guest CPU Write Tracking
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Write-protect dirty-page tracking gives every covered host page one canonical
// state and generation record shared by all live registrations. The invariants are:
// 1. Canonical pages: while any published or retiring range covers a host
//    page, every such range's immutable page slot points to the same page
//    record. Ranges are created and retired only under the slot mutex, so a
//    new range adopts the record of any live overlapping peer and allocates
//    records only for uncovered pages.
// 2. Ownership: a page record's reference count is touched only under the
//    slot mutex. Fault-path readers never touch reference counts or chunks.
// 3. Protection: a protected page is read-only, a clean page is writable, and
//    a busy page means one thread owns a transition. Because there is one
//    state per host page, no owner can observe protection after another owner
//    made the page writable; every unprotection publishes a new canonical
//    generation.
// 4. Retirement: protect/query readers are excluded and drained first. Only
//    pages whose sole owner is the retiring range are restored writable;
//    shared pages keep their state for the surviving owners. The range is
//    then unpublished, readers are drained again, and only afterwards are
//    page records freed. Retired address intervals remain as coalesced fault
//    history: a CPU can fault before entering the handler's reader bank. Such
//    a delayed fault retries only if a kernel query confirms writable memory.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace armigo::memory {

struct WriteTrackerStats {
    // Guest stores that faulted a protected tracked page.
    std::uint64_t faults = 0;
    // Pages made writable before a host write through prepare_host_write().
    std::uint64_t prepared = 0;
    // mprotect calls issued by this tracker, pages they covered, and the time
    // spent inside them in nanoseconds.
    std::uint64_t protect_calls = 0, protect_pages = 0, protect_ns = 0;
    // Transitions abandoned because a busy page stayed busy for too long.
    std::uint64_t busy_timeouts = 0;
    // Live registrations and their high-water mark.
    std::uint64_t live_ranges = 0, live_peak = 0;
    // Canonical page records and the extra owner references on them: shared
    // pages owned by several registrations appear once plus (owners - 1).
    std::uint64_t canonical_pages = 0, shared_references = 0;
};

// Process-global page-level CPU write tracker. Guest code runs natively, so
// the only way to observe its stores cheaply is write protection on host
// pages: a store to a protected tracked page faults once, the installed
// handler records a new write generation for that page and makes it writable.
// Registrations sharing a rounded host page share one canonical record, so
// any owner observes a write made through any address.
//
// The SIGSEGV/SIGBUS handlers installed here only resolve write faults on
// tracked protected pages or delayed permission faults whose tracked/retired
// address is now writable; every other fault chains to the previously
// installed handler exactly as it was configured (SA_SIGINFO, plain handler,
// SIG_IGN) or re-raises with the default disposition. Installation must be
// explicit via install() so a process that installs its own diagnostic
// handler first is chained, not replaced.
class WriteTracker {
public:
    WriteTracker() = delete;

    // Install the write-fault handlers. Idempotent; the first call chains to
    // whatever handlers were installed before it.
    static void install();
    static bool installed() noexcept;

    // Register a host range as tracked. The first call also installs the
    // fault handler and the host pre-write hook expectations. Ranges are
    // rounded outward to host pages; overlapping registrations share page
    // records. Re-registering storage bumps the global generation so stamps
    // belonging to the previous registration cannot vouch for it.
    // writable_alias, when present, names the same logical first byte through
    // a mapping that remains writable while mem is write-protected. It stays
    // valid until remove_range() returns.
    static void add_range(const void* mem, std::size_t size,
                          void* writable_alias = nullptr) noexcept;
    // Caller must stop new tracking users before removal and all users before
    // freeing the backing storage. Stores may finish while tracking retires.
    // Removal drains dirty-query readers and makes the pages this range alone
    // owns writable; host pages shared with a surviving registration keep
    // their canonical state and protection. Removal matches the exact
    // registered base and alias; the default removes only a non-alias range.
    static void remove_range(const void* mem,
                             void* writable_alias = nullptr) noexcept;

    // True when a live registration covers the whole span. A shorter
    // overlapping registration never masks a covering one; empty, untracked,
    // or boundary-straddling spans return false.
    static bool tracked(const void* addr, std::size_t len) noexcept;

    // True when one live alias-enabled registration covers every byte of the
    // nonempty span. Rounded page padding is not part of the registration.
    static bool has_writable_alias(const void* addr, std::size_t len) noexcept;

    // Copy through a registered writable alias without changing protection on
    // the original mapping, then publish the affected pages as written. Empty
    // stores succeed as no-ops; all other stores require one alias-enabled
    // registration covering the exact byte span.
    static bool try_alias_store(const void* addr,
                                std::span<const std::byte> source) noexcept;

    // Publish an already-completed host-side store to every registered page
    // intersecting the span. The caller serializes that store separately;
    // page permissions and their final tracker states are unchanged.
    static void note_store(const void* addr, std::size_t len) noexcept;

    // Make the covered pages read-only. Returns false when the range is not
    // fully tracked, a page is busy, or mprotect failed; a failed answer
    // advances the global generation so caches cannot keep stale clean
    // verdicts across it.
    static bool protect_checked(const void* addr, std::size_t len) noexcept;
    static void protect(const void* addr, std::size_t len) noexcept;

    // Current global generation. Taken before protection and before caching.
    static std::uint64_t now() noexcept;
    // Advances with every new write generation except engine stores made
    // inside a NotifiedStores scope, whose ranges the engine reports to its
    // own caches: while it holds still, no guest store, registration change or
    // unreported store reached any tracked page.
    static std::uint64_t unnoted_clock() noexcept;
    // Stores on this thread within the scope are reported by their caller.
    class NotifiedStores {
    public:
        NotifiedStores() noexcept;
        ~NotifiedStores();
        NotifiedStores(const NotifiedStores&) = delete;
        NotifiedStores& operator=(const NotifiedStores&) = delete;
    };

    // False only when a live registration covering the whole span existed at
    // the stamp and no byte of the span can have been stored since it: every
    // covered page was continuously observed and none is busy or carries a
    // newer generation. Untracked spans, a zero stamp, and stamps older than
    // the covering registration answer true.
    static bool written_since(const void* addr, std::size_t len,
                              std::uint64_t stamp) noexcept;

    // Unprotect the tracked pages of the span before a host write (a syscall
    // such as read()/pread() into a protected page returns EFAULT instead of
    // faulting). Untracked addresses are ignored; already writable pages are
    // left untouched without bumping their generation.
    static void prepare_host_write(const void* addr, std::size_t len) noexcept;

    static WriteTrackerStats stats() noexcept;

    // Test seam: replaces the mprotect call with a hook (returning true means
    // handled by the real mprotect). Only tests may set it; it is not signal
    // safe to change while tracked pages are protected.
    using ProtectHook = bool (*)(void* addr, std::size_t len, int prot);
    static void set_protect_hook(ProtectHook hook) noexcept;
};

} // namespace armigo::memory
