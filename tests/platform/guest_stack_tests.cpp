// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/guest_stack.h>

#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace {
using namespace armigo::platform;

int g_failed = 0;
int g_total = 0;

#define CHECK(condition)                                                                \
    do {                                                                                \
        ++g_total;                                                                      \
        if (!(condition)) {                                                             \
            ++g_failed;                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);            \
        }                                                                               \
    } while (false)

struct Region {
    bool found = false;  // a region contains the queried address
    std::uintptr_t base = 0;
    std::size_t size = 0;
    vm_prot_t protection = VM_PROT_NONE;
    vm_prot_t max_protection = VM_PROT_NONE;
};

Region region_at(std::uintptr_t address) {
    Region region;
    mach_vm_address_t base = address;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto status = mach_vm_region(mach_task_self(), &base, &size, VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    if (status != KERN_SUCCESS) {
        return region;
    }
    if (base > address || address - base >= size) return region;
    region.found = true;
    region.base = static_cast<std::uintptr_t>(base);
    region.size = static_cast<std::size_t>(size);
    region.protection = info.protection;
    region.max_protection = info.max_protection;
    return region;
}

bool range_mapped(std::uintptr_t base, std::size_t size) {
    for (std::uintptr_t address = base; address < base + size;) {
        const Region region = region_at(address);
        if (!region.found) return false;
        address = region.base + region.size;
    }
    return true;
}

bool range_unmapped(std::uintptr_t base, std::size_t size) {
    for (std::uintptr_t address = base; address < base + size; address += vm_page_size) {
        if (region_at(address).found) return false;
    }
    return true;
}

// Returns a free address range of `size` bytes by allocating and releasing it.
std::uintptr_t free_range(std::size_t size) {
    mach_vm_address_t address = 0;
    if (mach_vm_allocate(mach_task_self(), &address, size, VM_FLAGS_ANYWHERE) != KERN_SUCCESS) {
        return 0;
    }
    mach_vm_deallocate(mach_task_self(), address, size);
    return static_cast<std::uintptr_t>(address);
}

void test_default_layout() {
    const std::size_t page = vm_page_size;
    auto result = GuardedStack::create({.usable_size = 1});
    CHECK(result.ok());
    if (!result.ok()) return;
    const GuardedStackLayout& layout = result.stack->layout();
    CHECK(layout.page_size == page);
    CHECK(layout.usable_size == page);
    CHECK(layout.requested_usable_size == 1);
    CHECK(layout.guard_size == page);
    CHECK(layout.mapping_size == 3 * page);
    CHECK(layout.lower_guard_base == layout.mapping_base);
    CHECK(layout.usable_base == layout.mapping_base + page);
    CHECK(layout.aligned_top == layout.usable_base + layout.usable_size);
    CHECK(layout.upper_guard_base == layout.aligned_top);
    CHECK(layout.aligned_top % 16 == 0);
    CHECK(layout.usable_base % page == 0);
    CHECK(!layout.fixed);

    const Region lower = region_at(layout.lower_guard_base);
    CHECK(lower.found && lower.protection == VM_PROT_NONE && lower.max_protection == VM_PROT_NONE);
    CHECK(lower.found && lower.base == layout.lower_guard_base && lower.size == page);
    const Region interior = region_at(layout.usable_base);
    CHECK(interior.found && interior.protection == (VM_PROT_READ | VM_PROT_WRITE));
    CHECK(interior.found && interior.max_protection == (VM_PROT_READ | VM_PROT_WRITE));
    CHECK(interior.found && interior.base == layout.usable_base && interior.size == layout.usable_size);
    const Region upper = region_at(layout.upper_guard_base);
    CHECK(upper.found && upper.protection == VM_PROT_NONE && upper.max_protection == VM_PROT_NONE);
    CHECK(upper.found && upper.base == layout.upper_guard_base && upper.size == page);

    // Maximum protection really blocks reopening a guard or making the
    // interior executable.
    CHECK(::mprotect(reinterpret_cast<void*>(layout.lower_guard_base), page,
                     PROT_READ | PROT_WRITE) != 0);
    CHECK(::mprotect(reinterpret_cast<void*>(layout.usable_base), page,
                     PROT_READ | PROT_EXEC) != 0);

    const auto* bytes = reinterpret_cast<const unsigned char*>(layout.usable_base);
    bool zero = true;
    for (std::size_t index = 0; index < layout.usable_size; ++index) zero &= bytes[index] == 0;
    CHECK(zero);
    std::memset(reinterpret_cast<void*>(layout.usable_base), 0xa5, layout.usable_size);

    CHECK(result.stack->contains(layout.usable_base, layout.usable_size));
    CHECK(result.stack->contains(layout.aligned_top - 1));
    CHECK(!result.stack->contains(layout.aligned_top));
    CHECK(!result.stack->contains(layout.usable_base - 1));
    CHECK(!result.stack->contains(layout.usable_base, layout.usable_size + 1));
}

void test_rounding_and_guards() {
    const std::size_t page = vm_page_size;
    auto result = GuardedStack::create(
        {.usable_size = 2 * page + 1, .guard_size = page * 5 / 2});
    CHECK(result.ok());
    if (!result.ok()) return;
    const GuardedStackLayout& layout = result.stack->layout();
    CHECK(layout.usable_size == 3 * page);
    CHECK(layout.guard_size == 3 * page);
    CHECK(layout.mapping_size == 9 * page);
    for (std::uintptr_t address = layout.lower_guard_base; address < layout.usable_base;
         address += page) {
        CHECK(region_at(address).protection == VM_PROT_NONE);
    }
    for (std::uintptr_t address = layout.upper_guard_base;
         address < layout.mapping_base + layout.mapping_size; address += page) {
        CHECK(region_at(address).protection == VM_PROT_NONE);
    }
}

void test_invalid_requests() {
    const std::size_t page = vm_page_size;
    const auto error = [](const GuardedStackOptions& options) {
        auto result = GuardedStack::create(options);
        if (result.stack != nullptr) return GuardedStackError::none;
        return result.status.error;
    };
    CHECK(error({.usable_size = 0}) == GuardedStackError::invalid_size);
    CHECK(error({.usable_size = page, .limits = {.max_usable_bytes = kGuardedStackHardMaxUsableBytes + 1}}) ==
          GuardedStackError::invalid_limits);
    CHECK(error({.usable_size = page, .limits = {.max_guard_bytes = kGuardedStackHardMaxGuardBytes + 1}}) ==
          GuardedStackError::invalid_limits);
    CHECK(error({.usable_size = (std::size_t{256} << 20) + 1}) == GuardedStackError::resource_limit);
    CHECK(error({.usable_size = std::numeric_limits<std::size_t>::max()}) ==
          GuardedStackError::resource_limit);
    CHECK(error({.usable_size = page, .guard_size = std::numeric_limits<std::size_t>::max()}) ==
          GuardedStackError::resource_limit);
    CHECK(error({.usable_size = page, .guard_size = (std::size_t{16} << 20) + 1}) ==
          GuardedStackError::resource_limit);
    CHECK(error({.usable_size = 4 * page, .limits = {.max_usable_bytes = 2 * page}}) ==
          GuardedStackError::resource_limit);
    CHECK(error({.usable_size = page, .fixed_usable_base = page + 8}) ==
          GuardedStackError::misaligned_fixed_base);
    CHECK(error({.usable_size = page, .fixed_usable_base = 0}) == GuardedStackError::address_overflow);
    const std::uintptr_t top_page = std::numeric_limits<std::uintptr_t>::max() & ~(page - 1);
    CHECK(error({.usable_size = page, .fixed_usable_base = top_page}) ==
          GuardedStackError::address_overflow);
    // Fits arithmetically but lies far above the task's address range.
    const std::uintptr_t high = std::uintptr_t{1} << 62;
    const GuardedStackError beyond = error({.usable_size = page, .fixed_usable_base = high});
    CHECK(beyond == GuardedStackError::address_unavailable ||
          beyond == GuardedStackError::address_occupied);
    // The page-zero reservation is never replaced; the kernel reports it as
    // either occupied or below the task's lowest mappable address.
    const GuardedStackError page_zero = error({.usable_size = page, .fixed_usable_base = 16 * page});
    CHECK(page_zero == GuardedStackError::address_occupied ||
          page_zero == GuardedStackError::address_unavailable);
    // Exact limits are accepted.
    CHECK(error({.usable_size = 2 * page, .guard_size = 2 * page,
                 .limits = {.max_usable_bytes = 2 * page, .max_guard_bytes = 2 * page}}) ==
          GuardedStackError::none);
}

void test_fixed_placement() {
    const std::size_t page = vm_page_size;
    const std::size_t usable = 4 * page;
    const std::uintptr_t range = free_range(usable + 2 * page);
    CHECK(range != 0);
    if (range == 0) return;
    auto result = GuardedStack::create({.usable_size = usable, .fixed_usable_base = range + page});
    CHECK(result.ok());
    if (!result.ok()) return;
    const GuardedStackLayout& layout = result.stack->layout();
    CHECK(layout.fixed);
    CHECK(layout.usable_base == range + page);
    CHECK(layout.mapping_base == range);
    CHECK(region_at(layout.lower_guard_base).protection == VM_PROT_NONE);
    CHECK(region_at(layout.usable_base).protection == (VM_PROT_READ | VM_PROT_WRITE));

    // A second fixed stack at the same address must fail and leave the first
    // one fully intact.
    auto* word = reinterpret_cast<volatile std::uint64_t*>(layout.usable_base);
    *word = 0x1122334455667788;
    auto again = GuardedStack::create({.usable_size = usable, .fixed_usable_base = range + page});
    CHECK(!again.ok() && again.status.error == GuardedStackError::address_occupied);
    CHECK(*word == 0x1122334455667788);
    CHECK(region_at(layout.lower_guard_base).protection == VM_PROT_NONE);
    CHECK(region_at(layout.usable_base).protection == (VM_PROT_READ | VM_PROT_WRITE));
}

// An existing caller mapping with a recognizable pattern must survive every
// overlapping fixed request: interior, lower-guard-only and upper-guard-only.
void test_fixed_occupied_not_clobbered() {
    const std::size_t page = vm_page_size;
    void* mapping = ::mmap(nullptr, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(mapping != MAP_FAILED);
    if (mapping == MAP_FAILED) return;
    const auto base = reinterpret_cast<std::uintptr_t>(mapping);
    auto* bytes = static_cast<unsigned char*>(mapping);
    for (std::size_t index = 0; index < 3 * page; ++index) bytes[index] = static_cast<unsigned char>(index * 7 + 3);
    const auto intact = [&] {
        for (std::size_t index = 0; index < 3 * page; ++index) {
            if (bytes[index] != static_cast<unsigned char>(index * 7 + 3)) return false;
        }
        const Region region = region_at(base);
        return region.found && region.protection == (VM_PROT_READ | VM_PROT_WRITE) &&
               range_mapped(base, 3 * page);
    };

    // Mapping occupies [base, base + 3p); every request below uses one-page
    // guards and a one-page interior.
    struct Case {
        const char* name;
        std::uintptr_t usable_base;
    };
    const Case cases[] = {
        {"interior overlaps", base + page},
        {"only the upper guard overlaps", base - page},
        {"only the lower guard overlaps", base + 3 * page},
        {"exact overlap", base},
    };
    for (const Case& request : cases) {
        auto result = GuardedStack::create({.usable_size = page, .fixed_usable_base = request.usable_base});
        if (result.ok() || result.status.error != GuardedStackError::address_occupied) {
            std::printf("FAIL fixed request (%s) was not refused as occupied\n", request.name);
            ++g_failed;
        }
        ++g_total;
        CHECK(intact());
    }
    ::munmap(mapping, 3 * page);
}

// Runs last: a placement that replaced mappings would destroy this test's own
// code, so the pattern-based checks above report first.
void test_fixed_over_own_text() {
    const std::size_t page = vm_page_size;
    const auto text = reinterpret_cast<std::uintptr_t>(&free_range) & ~(page - 1);
    auto result = GuardedStack::create({.usable_size = page, .fixed_usable_base = text});
    // Text may occupy the task's first valid page. Its lower guard then lies
    // outside the task range, which Mach may report before the overlap.
    CHECK(!result.ok() && (result.status.error == GuardedStackError::address_occupied ||
                          result.status.error == GuardedStackError::address_unavailable));
    const Region region = region_at(text);
    CHECK(region.found && (region.protection & VM_PROT_EXECUTE) != 0);
}

void test_cleanup() {
    const std::size_t page = vm_page_size;
    std::uintptr_t mapping_base = 0;
    std::size_t mapping_size = 0;
    {
        auto result = GuardedStack::create({.usable_size = 8 * page});
        CHECK(result.ok());
        if (!result.ok()) return;
        mapping_base = result.stack->layout().mapping_base;
        mapping_size = result.stack->layout().mapping_size;
        CHECK(range_mapped(mapping_base, mapping_size));
    }
    CHECK(range_unmapped(mapping_base, mapping_size));

    // Repeated fixed creation at one address proves that every destruction
    // returned the whole range, guards included.
    const std::uintptr_t usable_base = mapping_base + page;
    for (int iteration = 0; iteration < 256; ++iteration) {
        auto result = GuardedStack::create({.usable_size = 8 * page, .fixed_usable_base = usable_base});
        if (!result.ok()) {
            CHECK(result.ok());
            break;
        }
        reinterpret_cast<volatile unsigned char*>(result.stack->usable_base())[0] = 1;
    }
    CHECK(range_unmapped(mapping_base, mapping_size));
}

void test_exclusive_use() {
    auto result = GuardedStack::create({.usable_size = 1});
    CHECK(result.ok());
    if (!result.ok()) return;
    GuardedStack& stack = *result.stack;
    CHECK(!stack.in_use());
    {
        const GuardedStackUse first = stack.try_acquire();
        CHECK(first.acquired() && first.stack() == &stack);
        CHECK(stack.in_use());
        const GuardedStackUse second = stack.try_acquire();
        CHECK(!second.acquired() && second.stack() == nullptr);
        std::atomic<bool> other_acquired{true};
        std::thread([&] { other_acquired = stack.try_acquire().acquired(); }).join();
        CHECK(!other_acquired);
        CHECK(stack.in_use());
    }
    CHECK(!stack.in_use());
    std::atomic<bool> other_acquired{false};
    std::thread([&] { other_acquired = stack.try_acquire().acquired(); }).join();
    CHECK(other_acquired);
    CHECK(!stack.in_use());
}

void test_describe() {
    for (int value = 0; value <= static_cast<int>(GuardedStackError::system_failure); ++value) {
        const char* text = describe(static_cast<GuardedStackError>(value));
        CHECK(text != nullptr && std::strcmp(text, "unknown guarded stack error") != 0);
    }
}

[[noreturn]] void run_child(const std::string& mode) {
    auto result = GuardedStack::create({.usable_size = 1});
    if (!result.ok()) _exit(2);
    GuardedStack* stack = result.stack.release();
    if (mode == "lower-guard") {
        reinterpret_cast<volatile unsigned char*>(stack->usable_base())[-1] = 1;
    } else if (mode == "upper-guard") {
        reinterpret_cast<volatile unsigned char*>(stack->aligned_top())[0] = 1;
    } else if (mode == "destroy-in-use") {
        const GuardedStackUse use = stack->try_acquire();
        delete stack;
    }
    _exit(1);
}

struct ChildOutcome {
    bool spawned = false;
    bool exited = false;
    int exit_code = -1;
    int signal = 0;
};

ChildOutcome spawn_child(const char* mode) {
    ChildOutcome outcome;
    char path[4096];
    std::uint32_t size = sizeof path;
    if (_NSGetExecutablePath(path, &size) != 0) return outcome;
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    char arg_child[] = "--child";
    std::string mode_text = mode;
    char* argv[] = {path, arg_child, mode_text.data(), nullptr};
    // Let faults terminate the child with their signal even under a sanitizer.
    std::vector<char*> environment;
    for (char** entry = environ; *entry != nullptr; ++entry) {
        if (std::strncmp(*entry, "ASAN_OPTIONS=", 13) != 0 &&
            std::strncmp(*entry, "TSAN_OPTIONS=", 13) != 0) {
            environment.push_back(*entry);
        }
    }
    char asan_options[] = "ASAN_OPTIONS=handle_segv=0:handle_sigbus=0";
    char tsan_options[] = "TSAN_OPTIONS=handle_segv=0:handle_sigbus=0";
    environment.push_back(asan_options);
    environment.push_back(tsan_options);
    environment.push_back(nullptr);
    pid_t pid = 0;
    const int error = posix_spawn(&pid, path, &actions, nullptr, argv, environment.data());
    posix_spawn_file_actions_destroy(&actions);
    if (error != 0) return outcome;
    outcome.spawned = true;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return outcome;
    if (WIFEXITED(status)) {
        outcome.exited = true;
        outcome.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        outcome.signal = WTERMSIG(status);
    }
    return outcome;
}

bool memory_fault(const ChildOutcome& outcome) {
    return outcome.spawned && !outcome.exited &&
           (outcome.signal == SIGSEGV || outcome.signal == SIGBUS);
}

void test_children() {
    const ChildOutcome lower = spawn_child("lower-guard");
    CHECK(memory_fault(lower));
    const ChildOutcome upper = spawn_child("upper-guard");
    CHECK(memory_fault(upper));
    const ChildOutcome destroyed = spawn_child("destroy-in-use");
    CHECK(destroyed.spawned && !destroyed.exited && destroyed.signal == SIGABRT);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) run_child(argv[2]);
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep reports if a defect crashes the process
    test_default_layout();
    test_rounding_and_guards();
    test_invalid_requests();
    test_fixed_placement();
    test_fixed_occupied_not_clobbered();
    test_cleanup();
    test_exclusive_use();
    test_describe();
    test_children();
    test_fixed_over_own_text();
    std::printf("%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}
