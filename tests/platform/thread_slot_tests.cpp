// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/thread_slot.h>

#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern "C" std::uint64_t armigo_test_thread_slot_load(std::uint64_t offset);
extern "C" std::uint64_t armigo_test_thread_slot_load_word(std::uint64_t offset,
                                                           std::uint64_t index);
extern "C" void armigo_test_thread_slot_store_word(std::uint64_t offset, std::uint64_t index,
                                                   std::uint64_t value);

extern char** environ;

namespace {
using namespace armigo::platform;

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition;
}

std::uint64_t address(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}

std::uint64_t offset_of(const ThreadSlotKey& key) {
    return key.native_access()->tsd_byte_offset;
}

bool key_properties() {
    ThreadSlotKey key;
    if (!check(key.status().ok() && key.valid(), "Key creation failed.")) return false;
    const auto access = key.native_access();
    if (!check(access.has_value(), "A valid key reported no native access.")) return false;
    const auto offset = access->tsd_byte_offset;
    if (!check(offset % 8 == 0 && offset <= NativeThreadSlotAccess::max_tsd_byte_offset,
               "The slot offset cannot be encoded as an unsigned-offset LDR.")) return false;
    if (!check(armigo_test_thread_slot_load(offset) == 0,
               "A new key's slot is not empty on the creating thread.")) return false;

    ThreadSlotKey other;
    return check(other.valid() && offset_of(other) != offset,
                 "Two live keys share one slot.") &&
           check(key.active_bindings() == 0, "A new key reports active bindings.");
}

bool main_thread_binding() {
    ThreadSlotKey key;
    if (!check(key.valid(), "Key creation failed.")) return false;
    const auto offset = offset_of(key);
    std::array<std::uint64_t, 4> storage{11, 22, 33, 44};
    {
        ThreadSlotBinding binding(key, storage.data());
        if (!check(binding.status().ok() && binding.active(), "Binding failed.") ||
            !check(binding.native_access().has_value() &&
                   binding.native_access()->tsd_byte_offset == offset,
                   "A binding reported a different descriptor.") ||
            !check(key.active_bindings() == 1, "The key did not count its binding.") ||
            !check(armigo_test_thread_slot_load(offset) == address(storage.data()),
                   "Native code did not observe the bound pointer.") ||
            !check(armigo_test_thread_slot_load_word(offset, 2) == 33,
                   "Native code could not read through the bound pointer.")) return false;
        armigo_test_thread_slot_store_word(offset, 3, 0x1800);
        if (!check(storage[3] == 0x1800, "Native code could not write caller storage.")) return false;
    }
    return check(key.active_bindings() == 0, "Unbinding did not release the key.") &&
           check(armigo_test_thread_slot_load(offset) == 0, "Unbinding did not clear the slot.");
}

bool independent_threads() {
    constexpr std::size_t thread_count = 8;
    ThreadSlotKey key;
    if (!check(key.valid(), "Key creation failed.")) return false;
    const auto offset = offset_of(key);

    std::array<std::uint64_t, 2> main_storage{};
    ThreadSlotBinding main_binding(key, main_storage.data());
    if (!check(main_binding.active(), "Main-thread binding failed.")) return false;

    std::vector<std::array<std::uint64_t, 2>> storage(thread_count);
    std::atomic<std::size_t> failures{0};
    std::barrier sync(static_cast<std::ptrdiff_t>(thread_count));
    std::vector<std::thread> threads;
    for (std::size_t index = 0; index < thread_count; ++index) {
        threads.emplace_back([&, index] {
            auto* own = storage[index].data();
            bool ok = armigo_test_thread_slot_load(offset) == 0 &&
                      !main_binding.native_access().has_value();
            {
                ThreadSlotBinding binding(key, own);
                ok = ok && binding.active() && binding.native_access().has_value();
                sync.arrive_and_wait();
                ok = ok && armigo_test_thread_slot_load(offset) == address(own);
                armigo_test_thread_slot_store_word(offset, 1, 0x100 + index);
                sync.arrive_and_wait();
                ok = ok && armigo_test_thread_slot_load(offset) == address(own) &&
                     armigo_test_thread_slot_load_word(offset, 1) == 0x100 + index;
                sync.arrive_and_wait();
            }
            ok = ok && armigo_test_thread_slot_load(offset) == 0;
            if (!ok) failures.fetch_add(1);
        });
    }
    for (auto& thread : threads) thread.join();

    if (!check(failures.load() == 0, "A thread observed another thread's slot.")) return false;
    for (std::size_t index = 0; index < thread_count; ++index) {
        if (!check(storage[index][1] == 0x100 + index, "Native stores reached the wrong storage."))
            return false;
    }
    return check(main_storage[1] == 0, "Other threads wrote the main thread's storage.") &&
           check(armigo_test_thread_slot_load(offset) == address(main_storage.data()),
                 "Other threads changed the main thread's slot.") &&
           check(key.active_bindings() == 1, "Thread bindings were not all released.");
}

// A reusable rendezvous built on a mutex, so that ThreadSanitizer observes the
// ordering it establishes. Data handed between threads uses this rather than
// std::barrier, whose libc++ algorithm runs in uninstrumented library code.
class Rendezvous {
public:
    explicit Rendezvous(std::size_t participants) : participants_(participants) {}

    void arrive_and_wait() {
        std::unique_lock lock(mutex_);
        const auto generation = generation_;
        if (++arrived_ == participants_) {
            arrived_ = 0;
            ++generation_;
            changed_.notify_all();
            return;
        }
        changed_.wait(lock, [&] { return generation_ != generation; });
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::size_t participants_;
    std::size_t arrived_ = 0;
    std::size_t generation_ = 0;
};

// Valid concurrent use only: every cross-thread query of a binding is ordered
// after its construction and before its destruction by a rendezvous or a join.
// Bind and unbind themselves overlap freely across threads on one key.
bool concurrent_use() {
    constexpr std::size_t thread_count = 6;
    constexpr int rounds = 200;
    ThreadSlotKey key;
    if (!check(key.valid(), "Key creation failed.")) return false;
    const auto offset = offset_of(key);

    int main_value = 0;
    ThreadSlotBinding main_binding(key, &main_value);
    if (!check(main_binding.active(), "Main-thread binding failed.")) return false;

    std::vector<ThreadSlotBinding*> held(thread_count, nullptr);
    std::vector<int> values(thread_count, 0);
    std::atomic<std::size_t> failures{0};
    std::atomic<bool> polling{true};
    Rendezvous published(thread_count + 1);
    Rendezvous released(thread_count + 1);

    std::thread poller([&] {
        bool ok = true;
        while (polling.load(std::memory_order_acquire)) {
            const auto count = key.active_bindings();
            ok = ok && count >= 1 && count <= thread_count + 1 && key.valid() &&
                 key.status().ok() && key.native_access()->tsd_byte_offset == offset &&
                 main_binding.active() && !main_binding.native_access().has_value();
        }
        if (!ok) failures.fetch_add(1);
    });

    std::vector<std::thread> threads;
    for (std::size_t index = 0; index < thread_count; ++index) {
        threads.emplace_back([&, index] {
            bool ok = true;
            for (int round = 0; round < rounds; ++round) {
                int value = round;
                ThreadSlotBinding binding(key, &value);
                ok = ok && binding.active() && binding.native_access().has_value() &&
                     armigo_test_thread_slot_load(offset) == address(&value);
            }
            ok = ok && armigo_test_thread_slot_load(offset) == 0;
            {
                ThreadSlotBinding binding(key, &values[index]);
                held[index] = &binding;
                published.arrive_and_wait();
                released.arrive_and_wait();
                ok = ok && armigo_test_thread_slot_load(offset) == address(&values[index]);
            }
            if (!ok) failures.fetch_add(1);
        });
    }

    // Between the two rendezvous points every worker binding is alive, so
    // querying it from this thread is valid; it must report active with no
    // access here.
    published.arrive_and_wait();
    bool foreign_ok = true;
    for (auto* binding : held) {
        foreign_ok = foreign_ok && binding->active() && binding->status().ok() &&
                     !binding->native_access().has_value();
    }
    released.arrive_and_wait();
    for (auto& thread : threads) thread.join();
    polling.store(false, std::memory_order_release);
    poller.join();

    return check(foreign_ok, "A foreign thread observed another thread's binding as usable.") &&
           check(failures.load() == 0, "Concurrent binding on one key misbehaved.") &&
           check(key.active_bindings() == 1, "Concurrent bindings were not all released.") &&
           check(armigo_test_thread_slot_load(offset) == address(&main_value),
                 "Concurrent bindings changed the main thread's slot.");
}

bool nested_restoration() {
    ThreadSlotKey first;
    ThreadSlotKey second;
    if (!check(first.valid() && second.valid(), "Key creation failed.")) return false;
    const auto a = offset_of(first);
    const auto b = offset_of(second);
    int outer = 0, inner = 0, other = 0;

    ThreadSlotBinding outer_binding(first, &outer);
    {
        ThreadSlotBinding inner_binding(first, &inner);
        if (!check(inner_binding.active() && armigo_test_thread_slot_load(a) == address(&inner),
                   "Nested binding was not published.")) return false;
        {
            ThreadSlotBinding other_binding(second, &other);
            if (!check(other_binding.active() &&
                       armigo_test_thread_slot_load(b) == address(&other) &&
                       armigo_test_thread_slot_load(a) == address(&inner),
                       "Bindings of different keys interfered.")) return false;
            {
                ThreadSlotBinding same_pointer(first, &inner);
                if (!check(same_pointer.active() && first.active_bindings() == 3,
                           "Rebinding the same pointer failed.")) return false;
            }
            if (!check(armigo_test_thread_slot_load(a) == address(&inner),
                       "Same-pointer nesting did not restore.")) return false;
        }
        if (!check(armigo_test_thread_slot_load(b) == 0 &&
                   armigo_test_thread_slot_load(a) == address(&inner),
                   "Restoring one key changed another.")) return false;
    }
    return check(armigo_test_thread_slot_load(a) == address(&outer) &&
                 first.active_bindings() == 1 && second.active_bindings() == 0,
                 "Nested binding did not restore the outer pointer.");
}

bool rejected_bindings() {
    ThreadSlotKey key;
    if (!check(key.valid(), "Key creation failed.")) return false;
    const auto offset = offset_of(key);
    int value = 0;
    ThreadSlotBinding binding(key, &value);
    {
        ThreadSlotBinding rejected(key, nullptr);
        if (!check(!rejected.active() &&
                   rejected.status().error == ThreadSlotError::invalid_argument &&
                   !rejected.native_access().has_value(),
                   "A null pointer was bound.")) return false;
    }
    return check(armigo_test_thread_slot_load(offset) == address(&value) &&
                 key.active_bindings() == 1,
                 "A rejected binding changed the slot or count.");
}

bool key_replacement() {
    int value = 0;
    {
        ThreadSlotKey key;
        if (!check(key.valid(), "Key creation failed.")) return false;
        std::thread([&] { ThreadSlotBinding binding(key, &value); }).join();
        ThreadSlotBinding binding(key, &value);
    }
    ThreadSlotKey replacement;
    if (!check(replacement.valid(), "Replacement key creation failed.")) return false;
    bool ok = false;
    std::thread([&] {
        const auto offset = offset_of(replacement);
        ok = armigo_test_thread_slot_load(offset) == 0;
        ThreadSlotBinding binding(replacement, &value);
        ok = ok && binding.active() && armigo_test_thread_slot_load(offset) == address(&value);
    }).join();
    // libpthread may return the deleted key number again; either way, no
    // stale value may be visible through the replacement.
    return check(ok, "A replacement key exposed stale state or failed to bind.");
}

// Child-process cases. Contract violations must abort, never continue.

int child_exhaustion() {
    std::vector<std::unique_ptr<ThreadSlotKey>> keys;
    for (;;) {
        auto key = std::make_unique<ThreadSlotKey>();
        if (!key->valid()) {
            int value = 0;
            ThreadSlotBinding binding(*key, &value);
            const bool ok = key->status().error == ThreadSlotError::key_exhausted &&
                            !key->native_access().has_value() && !binding.active() &&
                            binding.status().error == ThreadSlotError::invalid_state &&
                            !keys.empty();
            keys.clear();
            ThreadSlotKey recovered;
            return ok && recovered.valid() ? 0 : 1;
        }
        keys.push_back(std::move(key));
        if (keys.size() > 100000) return 2;
    }
}

int child_out_of_order() {
    ThreadSlotKey key;
    int a = 0, b = 0;
    auto* outer = new ThreadSlotBinding(key, &a);
    new ThreadSlotBinding(key, &b);
    delete outer;
    return 0;
}

int child_cross_key_out_of_order() {
    ThreadSlotKey first;
    ThreadSlotKey second;
    int a = 0, b = 0;
    auto* outer = new ThreadSlotBinding(first, &a);
    new ThreadSlotBinding(second, &b);
    delete outer;
    return 0;
}

int child_wrong_thread() {
    ThreadSlotKey key;
    int a = 0;
    auto* binding = new ThreadSlotBinding(key, &a);
    std::thread([binding] { delete binding; }).join();
    return 0;
}

int child_key_outlived() {
    int a = 0;
    auto* key = new ThreadSlotKey;
    new ThreadSlotBinding(*key, &a);
    delete key;
    return 0;
}

int child_thread_exit() {
    ThreadSlotKey key;
    int a = 0;
    std::thread([&] { new ThreadSlotBinding(key, &a); }).join();
    return 0;
}

struct ChildResult {
    int status = 0;
    std::string output;
};

bool run_child(const char* name, ChildResult& result) {
    char path[4096];
    std::uint32_t size = sizeof(path);
    if (::_NSGetExecutablePath(path, &size) != 0) return false;
    int pipe_fds[2];
    if (::pipe(pipe_fds) != 0) return false;
    posix_spawn_file_actions_t actions;
    ::posix_spawn_file_actions_init(&actions);
    ::posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDERR_FILENO);
    ::posix_spawn_file_actions_addclose(&actions, pipe_fds[0]);
    char case_flag[] = "--child";
    std::string case_name = name;
    char* argv[] = {path, case_flag, case_name.data(), nullptr};
    pid_t pid = 0;
    const int spawned = ::posix_spawn(&pid, path, &actions, nullptr, argv, environ);
    ::posix_spawn_file_actions_destroy(&actions);
    ::close(pipe_fds[1]);
    if (spawned != 0) {
        ::close(pipe_fds[0]);
        return false;
    }
    char buffer[512];
    ssize_t count = 0;
    while ((count = ::read(pipe_fds[0], buffer, sizeof(buffer))) > 0) result.output.append(buffer, count);
    ::close(pipe_fds[0]);
    return ::waitpid(pid, &result.status, 0) == pid;
}

// The diagnostic is the primary check. abort() normally ends in SIGABRT. An
// exiting thread may have signal delivery disabled by the time its pthread key
// destructors run; the observed Darwin abort() then falls back to a trap, so
// that case also accepts SIGTRAP.
bool expect_abort(const char* name, const char* diagnostic, bool allow_trap = false) {
    ChildResult result;
    if (!check(run_child(name, result), "Could not run a child case.")) return false;
    const int signal_number = WIFSIGNALED(result.status) ? WTERMSIG(result.status) : 0;
    const bool aborted = signal_number == SIGABRT || (allow_trap && signal_number == SIGTRAP);
    if (!aborted || result.output.find(diagnostic) == std::string::npos) {
        std::fprintf(stderr, "Child case %s did not abort as required:\n%s\n", name,
                     result.output.c_str());
        return false;
    }
    return true;
}

bool contract_violations() {
    return expect_abort("out-of-order", "not destroyed in reverse order") &&
           expect_abort("cross-key-out-of-order", "not destroyed in reverse order") &&
           expect_abort("wrong-thread", "thread that does not own it") &&
           expect_abort("key-outlived", "destroyed while bindings were active") &&
           expect_abort("thread-exit", "exited with an active binding", true);
}

bool key_exhaustion() {
    ChildResult result;
    return check(run_child("exhaustion", result), "Could not run a child case.") &&
           check(WIFEXITED(result.status) && WEXITSTATUS(result.status) == 0,
                 "Key exhaustion was not reported and recovered explicitly.");
}

int run_child_case(const char* name) {
    const std::string selected = name;
    if (selected == "exhaustion") return child_exhaustion();
    if (selected == "out-of-order") return child_out_of_order();
    if (selected == "cross-key-out-of-order") return child_cross_key_out_of_order();
    if (selected == "wrong-thread") return child_wrong_thread();
    if (selected == "key-outlived") return child_key_outlived();
    if (selected == "thread-exit") return child_thread_exit();
    return 3;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) return run_child_case(argv[2]);
    const bool ok = key_properties() && main_thread_binding() && independent_threads() &&
                    concurrent_use() && nested_restoration() && rejected_bindings() &&
                    key_replacement() && key_exhaustion() && contract_violations();
    if (ok) std::puts("thread slot tests passed");
    return ok ? 0 : 1;
}
