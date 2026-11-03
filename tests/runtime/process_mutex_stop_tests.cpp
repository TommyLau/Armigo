// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "process_runtime_fixture.h"
#include "../../src/runtime/process_runtime_test_access.h"

extern "C" {
extern const std::byte armigo_process_fixture_mutex_stop[];
extern const std::byte armigo_process_fixture_mutex_stop_end[];
extern const std::byte armigo_process_fixture_mutex_heap[];
extern const std::byte armigo_process_fixture_mutex_waiter_stack[];
extern const std::byte armigo_process_fixture_mutex_owner[];
extern const std::byte armigo_process_fixture_mutex_owner_end[];
extern const std::byte armigo_process_fixture_mutex_waiter[];
extern const std::byte armigo_process_fixture_mutex_waiter_end[];
extern const std::byte armigo_process_fixture_nested_stop[];
extern const std::byte armigo_process_fixture_nested_stop_end[];
extern const std::byte armigo_process_fixture_nested_heap[];
extern const std::byte armigo_process_fixture_nested_waiter[];
extern const std::byte armigo_process_fixture_nested_waiter_end[];
}

namespace {
loader::ExeFsDirectoryInput mutex_stop_input(std::uint64_t heap) {
    runtime::test::ModuleSpec module;
    module.name = "main";
    module.text.resize(runtime::test::rodata_offset);
    std::memcpy(module.text.data() + 0x100, armigo_process_fixture_begin,
        static_cast<std::size_t>(armigo_process_fixture_end - armigo_process_fixture_begin));
    module.data.resize(0x30000 - runtime::test::data_offset);
    const auto symbol = [](const char* name, const std::byte* begin, const std::byte* end) {
        return runtime::test::SymbolSpec{name, fixture_offset(begin),
            static_cast<std::uint64_t>(end - begin), true};
    };
    module.symbols = {
        symbol("_start", armigo_process_fixture_mutex_stop,
               armigo_process_fixture_mutex_stop_end),
        symbol("mutex_owner", armigo_process_fixture_mutex_owner,
               armigo_process_fixture_mutex_owner_end),
        symbol("mutex_waiter", armigo_process_fixture_mutex_waiter,
               armigo_process_fixture_mutex_waiter_end),
    };
    const auto heap_instruction = fixture_offset(armigo_process_fixture_mutex_heap);
    for (unsigned half = 0; half != 4; ++half) {
        runtime::test::put_word(module, heap_instruction + half * 4,
            move_wide(half != 0, 19, half,
                static_cast<std::uint16_t>(heap >> (half * 16))));
    }
    const auto waiter_stack = fixture_offset(armigo_process_fixture_mutex_waiter_stack);
    runtime::test::put_word(module, waiter_stack,
        aarch64::encode_adr(3, waiter_stack, 0x20000).word);
    loader::ExeFsDirectoryInput input;
    input.modules.push_back({"main", runtime::test::build_module(std::move(module))});
    return input;
}

loader::ExeFsDirectoryInput nested_stop_input(std::uint64_t heap) {
    runtime::test::ModuleSpec module;
    module.name = "main";
    module.text.resize(runtime::test::rodata_offset);
    std::memcpy(module.text.data() + 0x100, armigo_process_fixture_begin,
        static_cast<std::size_t>(armigo_process_fixture_end - armigo_process_fixture_begin));
    module.data.resize(0x30000 - runtime::test::data_offset);
    const auto symbol = [](const char* name, const std::byte* begin, const std::byte* end) {
        return runtime::test::SymbolSpec{name, fixture_offset(begin),
            static_cast<std::uint64_t>(end - begin), true};
    };
    module.symbols = {
        symbol("_start", armigo_process_fixture_nested_stop,
               armigo_process_fixture_nested_stop_end),
        symbol("nested_waiter", armigo_process_fixture_nested_waiter,
               armigo_process_fixture_nested_waiter_end),
    };
    const auto heap_instruction = fixture_offset(armigo_process_fixture_nested_heap);
    for (unsigned half = 0; half != 4; ++half) {
        runtime::test::put_word(module, heap_instruction + half * 4,
            move_wide(half != 0, 19, half,
                static_cast<std::uint16_t>(heap >> (half * 16))));
    }
    loader::ExeFsDirectoryInput input;
    input.modules.push_back({"main", runtime::test::build_module(std::move(module))});
    return input;
}

struct Created {
    std::unique_ptr<Fixture> fixture;
    std::unique_ptr<ProcessRuntime> runtime;
    std::uint64_t heap = 0;
};

// Prepares and creates a runtime for a fixture whose heap address is patched in
// by `build_input`, retrying while the candidate heap range is taken.
template <class BuildInput>
Created create_fixture_runtime(BuildInput build_input) {
    Created created;
    for (unsigned attempt = 0; attempt != 8 && !created.runtime; ++attempt) {
        const auto candidate_address = probe_heap_address();
        if (!candidate_address) continue;
        auto candidate = std::make_unique<Fixture>();
        candidate->input = build_input(*candidate_address);
        CHECK(write_file(candidate->exefs / "main", candidate->input.modules.front().bytes));
        const auto artifacts = prepare_modules(candidate->input, {
            .required_preparation_identity = installed_preparation_identity(),
            .max_artifact_bytes = default_prepared_artifact_limit,
            .mode = PreparationMode::synthetic_test,
        });
        CHECK(value(artifacts) != nullptr);
        if (!value(artifacts)) {
            std::fprintf(stderr, "Mutex fixture preparation: %s: %s\n",
                failure(artifacts)->operation.c_str(), failure(artifacts)->detail.c_str());
            return created;
        }
        std::filesystem::remove_all(candidate->prepared);
        CHECK(ok(write_prepared_modules(candidate->prepared, *value(artifacts))));
        auto& profile = *candidate->description.synthetic_profile;
        profile.process.heap_address = *candidate_address;
        profile.permissions.supervisor_calls = {
            0x01, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x18, 0x1a, 0x1b, 0x1c, 0x1d, 0x42};
        auto made = ProcessRuntime::create(candidate->description, dependencies());
        if (value(made)) {
            created.heap = *candidate_address;
            created.runtime = std::move(std::get<std::unique_ptr<ProcessRuntime>>(made));
            created.fixture = std::move(candidate);
        } else if (failure(made)->operation != "create_guest_memory reservations") {
            std::fprintf(stderr, "Mutex fixture creation: %s: %s\n",
                failure(made)->operation.c_str(), failure(made)->detail.c_str());
            break;
        }
    }
    return created;
}

void test_real_mutex_handoff_stop(bool pending_stop) {
    auto created = create_fixture_runtime(mutex_stop_input);
    const auto fixture = std::move(created.fixture);
    const auto runtime = std::move(created.runtime);
    const auto heap = created.heap;
    CHECK(runtime != nullptr);
    if (!runtime) return;

    std::optional<RunResult> result;
    std::thread runner([&] { result = runtime->run({}, {}); });
    const auto enrollment_deadline = std::chrono::steady_clock::now() + 2s;
    bool enrolled = false;
    do {
        enrolled = ProcessRuntimeTestAccess::mutex_waiters(*runtime, heap) == 1;
        if (enrolled) break;
        std::this_thread::sleep_for(1ms);
    } while (std::chrono::steady_clock::now() < enrollment_deadline);
    CHECK(enrolled);
    if (native_writable(heap)) store_native_word(heap + 48, 1);
    const bool handed_off = wait_for_native_word(heap + 20, 1, 2s);
    CHECK(handed_off);
    if (handed_off) {
        const auto read = [heap](unsigned offset) {
            return __atomic_load_n(reinterpret_cast<const std::uint32_t*>(heap + offset),
                                   __ATOMIC_ACQUIRE);
        };
        CHECK(read(4) != 0 && read(8) != 0 && read(4) != read(8));
        CHECK(read(0) == read(8));
        std::printf("Real ArbitrateLock handoff verified (pending stop: %u).\n",
                    pending_stop);
        if (pending_stop) runtime->request_stop();
        store_native_word(heap + 24, 1);
        const bool resumed_with_grant = wait_for_native_word(heap + 28, 1, 500ms);
        CHECK(resumed_with_grant);
        CHECK(read(0) == read(8));
        store_native_word(heap + 32, 1);
        if (!pending_stop) {
            CHECK(wait_for_native_word(heap + 36, 1, 500ms));
            CHECK(read(0) == 0);
            runtime->request_stop();
            store_native_word(heap + 40, 1);
        }
    } else {
        runtime->request_stop();
        if (native_writable(heap)) {
            store_native_word(heap + 16, 1);
            store_native_word(heap + 24, 1);
            store_native_word(heap + 32, 1);
            store_native_word(heap + 40, 1);
        }
    }
    runner.join();
    CHECK(result.has_value());
    if (!result) return;
    if (const auto* failed = failure(*result)) {
        std::fprintf(stderr, "Mutex fixture run: %s: %s\n",
                     failed->operation.c_str(), failed->detail.c_str());
    }
    CHECK(std::holds_alternative<RunCancelled>(result->outcome));
    CHECK(result->quiescent);
    CHECK(!result->teardown_failure.has_value());
    if (!result->quiescent && native_writable(heap)) {
        const auto resumed = __atomic_load_n(
            reinterpret_cast<const std::uint32_t*>(heap + 36), __ATOMIC_ACQUIRE);
        const auto mutex = __atomic_load_n(
            reinterpret_cast<const std::uint32_t*>(heap), __ATOMIC_ACQUIRE);
        std::fprintf(stderr,
            "Regression evidence: post-unlock native marker=%u, mutex=%u, quiescent=%u.\n",
            resumed, mutex, result->quiescent);
        CHECK(resumed == 0);
        // Release only the native escape gate; never modify mutex ownership.
        store_native_word(heap + 40, 1);
    }
    CHECK(ok(runtime->shutdown(std::chrono::steady_clock::now() + 2s)));
}

// A worker that owns a kernel mutex grant parks in a second wait; terminating
// the process must end it. The wait is not released by anyone: kind 0 names the
// main thread as the nested mutex's owner and the main thread never unlocks it.
// Kind 3 is not a wait: the worker polls a user-mode flag nobody will set, with
// a call that never blocks, so only the forced stop of a shutdown can end it.
// Kind 4 spins on the flag without any call: no boundary is ever reached, so only
// an asynchronous stop of the running guest code can end it.
enum class NestedWait : std::uint32_t { arbitrate_lock = 0, sleep = 1, condition_wait = 2, user_poll = 3,
                                      user_spin = 4 };

void test_termination_ends_nested_wait(NestedWait kind, const char* name) {
    auto created = create_fixture_runtime(nested_stop_input);
    const auto fixture = std::move(created.fixture);
    const auto runtime = std::move(created.runtime);
    const auto heap = created.heap;
    CHECK(runtime != nullptr);
    if (!runtime) return;
    const auto read = [heap](unsigned offset) {
        return __atomic_load_n(reinterpret_cast<const std::uint32_t*>(heap + offset),
                               __ATOMIC_ACQUIRE);
    };

    std::optional<RunResult> result;
    std::thread runner([&] { result = runtime->run({}, {}); });
    const bool granted = wait_for_native_word(heap + 12, 1, 2s);
    CHECK(granted);
    if (granted) {
        const auto main_thread = ProcessRuntimeTestAccess::inspect(*runtime).main_thread;
        store_native_word(heap + 4, main_thread);
        store_native_word(heap + 8, main_thread | 0x40000000u);
        store_native_word(heap + 24, static_cast<std::uint32_t>(kind));
        store_native_word(heap + 16, 1);
        CHECK(wait_for_native_word(heap + 28, 1, 2s));
        if (kind == NestedWait::arbitrate_lock) {
            const auto deadline = std::chrono::steady_clock::now() + 2s;
            while (ProcessRuntimeTestAccess::mutex_waiters(*runtime, heap + 8) != 1 &&
                   std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(1ms);
            CHECK(ProcessRuntimeTestAccess::mutex_waiters(*runtime, heap + 8) == 1);
        } else {
            // No enrollment probe exists for these waits; give the worker time to block.
            std::this_thread::sleep_for(100ms);
        }
    }
    runtime->request_stop();
    runner.join();
    CHECK(result.has_value());
    if (!result) return;
    CHECK(std::holds_alternative<RunCancelled>(result->outcome));
    if (!result->quiescent) {
        if (kind == NestedWait::user_poll || kind == NestedWait::user_spin) {
            // Without the forced stop the poller keeps running under its grant.
            // Release it so the runtime can still be torn down, then fail.
            CHECK(false);
            std::fprintf(stderr, "Nested %s wait survived the shutdown grace period.\n", name);
            store_native_word(heap + 64, 1);
            CHECK(ok(runtime->shutdown(std::chrono::steady_clock::now() + 2s)));
            return;
        }
        // A thread is still parked in the wait; it cannot be freed from outside,
        // so fail fast instead of blocking the destructor forever.
        std::fprintf(stderr, "Nested %s wait survived termination: not quiescent (%s).\n", name,
                     result->teardown_failure ? result->teardown_failure->detail.c_str() : "no detail");
        std::fflush(nullptr);
        std::_Exit(1);
    }
    CHECK(!result->teardown_failure.has_value());
    // The wait ended by termination, not by returning to the guest: a resumed
    // wait spins in guest code (fixture) and the run above is not quiescent.
    // Guest memory is released by then, so its marker word cannot be read.
    std::printf("Termination ended the nested %s wait.\n", name);
}
} // namespace

int main() {
    test_real_mutex_handoff_stop(true);
    test_real_mutex_handoff_stop(false);
    test_termination_ends_nested_wait(NestedWait::arbitrate_lock, "arbitrate-lock");
    test_termination_ends_nested_wait(NestedWait::sleep, "sleep");
    test_termination_ends_nested_wait(NestedWait::condition_wait, "condition-variable");
    test_termination_ends_nested_wait(NestedWait::user_poll, "user-mode poll");
    test_termination_ends_nested_wait(NestedWait::user_spin, "user-mode spin");
    std::printf("Mutex-stop failures: %u\n", failures);
    return failures ? 1 : 0;
}
