// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/gate_code.h>
#include <armigo/execution/guest_entry.h>
#include <armigo/execution/prepared_image.h>
#include <armigo/platform/host_module.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <signal.h>
#include <thread>
#include <vector>

extern "C" std::uint64_t armigo_test_monitor_gate(std::uintptr_t, std::uint64_t*);
extern "C" void armigo_test_monitor_redirect();
extern "C" void armigo_test_monitor_gate_loop(std::uintptr_t, std::uint64_t);
extern "C" std::uint64_t armigo_test_monitor_signal(
    std::uint64_t*, std::atomic<unsigned>*, std::atomic<unsigned>*);
extern "C" void armigo_test_monitor_entry(armigo::execution::GuestEntryRecord*, std::uint64_t*);
extern "C" void armigo_test_monitor_store();

namespace {
using namespace armigo;
using namespace armigo::execution;
constexpr unsigned attempts = 10'000;

bool check(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}

// Kept away from host stack/dispatch state so incidental stores do not clear
// the reservation being tested. Spurious STXR failure is always permitted.
struct alignas(2048) Word { std::uint64_t value = 0; };

void reserve(std::uint64_t* word) {
    std::uint64_t ignored;
    asm volatile("ldxr %0, [%1]" : "=&r"(ignored) : "r"(word) : "memory");
}

struct Callback {
    Word word;
    bool redirect = false;
    bool fail = false;
};

GateStatus contaminate(GateCall& call, void* opaque) {
    auto& state = *static_cast<Callback*>(opaque);
    if (state.redirect)
        call.set_x(30, reinterpret_cast<std::uintptr_t>(&armigo_test_monitor_redirect));
    const auto status = state.fail ? GateStatus::failed : GateStatus::resume;
    reserve(&state.word.value);
    return status;
}

GateStatus supervisor(std::uint16_t, GateCall&, GuestThreadContext&, void* opaque) {
    reserve(static_cast<std::uint64_t*>(opaque));
    return GateStatus::resume;
}

platform::HostModuleResult map_code(std::span<const std::byte> bytes) {
    const std::array regions{platform::ModuleRegion{
        0, bytes.size(), memory::Permission::read | memory::Permission::execute}};
    return platform::HostModule::create(bytes, regions);
}

bool gate_resumes_clear() {
    auto copy = map_code(make_local_gate_code());
    if (!check(copy.ok() && copy.module->seal().ok(), "Could not map the copied gate.")) return false;
    const std::array gates{reinterpret_cast<std::uintptr_t>(&armigo_execution_gate_entry),
                           static_cast<std::uintptr_t>(copy.module->base_address())};
    for (const auto gate : gates) {
        for (unsigned mode = 0; mode < 3; ++mode) {
            Callback state;
            state.redirect = mode == 1;
            state.fail = mode == 2;
            GateScope scope(contaminate, &state, GateFailurePolicy::resume_unchanged);
            for (unsigned attempt = 0; attempt < attempts; ++attempt) {
                state.word.value = 0;
                if (!check(armigo_test_monitor_gate(gate, &state.word.value) == 1 &&
                           state.word.value == 0,
                           "A host reservation survived a gate resume/redirect/failure return.")) return false;
            }
            if (!check(scope.failure_count() == (state.fail ? attempts : 0),
                       "The gate reservation test exercised an unexpected failure path.")) return false;
        }
    }
    return true;
}

bool prepared_svc_clears() {
    // SVC #0; STXR W2,X1,[X0]; MOV X0,X2; RET.
    const std::array<std::uint32_t, 4> code{0xd4000001u, 0xc8027c01u, 0xaa0203e0u, 0xd65f03c0u};
    std::vector<CoveredInstruction> coverage;
    for (const auto word : code) coverage.push_back({coverage.size() * 4, word, {}, {}});
    const auto plan = prepare_image(sizeof(code), coverage,
        {.module_index = 0, .module_count = 1, .scalar_x18_policy = {false, true, true}});
    if (!check(plan.ok(), "The standalone store-exclusive SVC fixture was not admitted.")) return false;
    std::vector<std::byte> bytes(plan.image->image_size());
    std::memcpy(bytes.data(), code.data(), sizeof(code));
    auto mapped = map_code(bytes);
    if (!check(mapped.ok(), "Could not map the prepared SVC fixture.")) return false;
    const std::array<std::uint64_t, 1> bases{mapped.module->base_address()};
    const auto applied = apply_prepared_image(*plan.image, mapped.module->preparation_bytes(),
                                              {bases, 0, gate_dispatcher_address()});
    if (!check(applied.ok() && mapped.module->seal().ok(), "Could not seal the prepared SVC fixture.")) return false;
    std::vector<GuestGateSite> sites;
    for (const auto& site : plan.image->gate_sites)
        sites.push_back({bases[0] + site.return_address_offset, site.instruction, {}});
    const auto table = GuestSiteTable::build(sites);
    auto stack = platform::GuardedStack::create({.usable_size = 1u << 20});
    if (!check(table.ok() && stack.ok(), "Could not initialize the prepared SVC fixture.")) return false;
    Word word;
    GuestThreadContext context;
    GuestDispatchScope scope(table.table, context, {supervisor, &word.value});
    for (unsigned attempt = 0; attempt < attempts; ++attempt) {
        word.value = 0;
        const auto result = invoke_guest_entry(*stack.stack,
            {.function = bases[0], .x0 = reinterpret_cast<std::uintptr_t>(&word.value), .x1 = 99});
        if (!check(result.returned() && result.x0 == 1 && word.value == 0,
                   "STXR after guest SVC consumed the supervisor's reservation.")) return false;
    }
    return check(scope.failure_count() == 0, "The prepared SVC test encountered a dispatch failure.");
}

bool initial_entry_clears() {
    auto stack = platform::GuardedStack::create({.usable_size = 1u << 20});
    if (!check(stack.ok(), "Could not allocate the entry reservation test stack.")) return false;
    auto use = stack.stack->try_acquire();
    if (!check(use.acquired(), "Could not acquire the entry reservation test stack.")) return false;
    Word word;
    for (unsigned attempt = 0; attempt < attempts; ++attempt) {
        word.value = 0;
        GuestEntryRecord record{
            .function = reinterpret_cast<std::uintptr_t>(&armigo_test_monitor_store),
            .stack_top = stack.stack->layout().aligned_top,
            .x0 = reinterpret_cast<std::uintptr_t>(&word.value), .x1 = 99};
        armigo_test_monitor_entry(&record, &word.value);
        if (!check(record.completion == ARMIGO_GUEST_ENTRY_RETURNED &&
                   record.result_x0 == 1 && word.value == 0,
                   "Initial guest entry imported a host exclusive reservation.")) return false;
    }
    return true;
}

alignas(2048) std::atomic<unsigned> signal_continue{0};
std::atomic<std::uint64_t*> signal_word{nullptr};
static_assert(decltype(signal_continue)::is_always_lock_free);
static_assert(decltype(signal_word)::is_always_lock_free);

void signal_reservation(int) {
    signal_continue.store(1, std::memory_order_release);
    reserve(signal_word.load(std::memory_order_relaxed));
}

bool signal_return_clears() {
    // There is no production signal-resume trampoline. Verify the OS return
    // path separately, while a signal interrupts native guest instructions.
    struct sigaction action{}, previous{};
    action.sa_handler = signal_reservation;
    sigemptyset(&action.sa_mask);
    if (!check(sigaction(SIGUSR1, &action, &previous) == 0,
               "Could not install the synthetic signal handler.")) return false;
    Word word;
    alignas(2048) std::atomic<unsigned> ready{0};
    const auto guest_thread = pthread_self();
    signal_word.store(&word.value, std::memory_order_relaxed);
    bool ok = true;
    for (unsigned attempt = 0; attempt < 100 && ok; ++attempt) {
        ready.store(0, std::memory_order_relaxed);
        signal_continue.store(0, std::memory_order_relaxed);
        word.value = 0;
        int delivery = 0;
        std::thread sender([&] {
            while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
            delivery = pthread_kill(guest_thread, SIGUSR1);
            if (delivery) signal_continue.store(1, std::memory_order_release);
        });
        const auto status = armigo_test_monitor_signal(&word.value, &ready, &signal_continue);
        sender.join();
        ok = check(delivery == 0 && status == 1 && word.value == 0,
                   "A signal handler's reservation survived resumption of native guest code.");
    }
    signal_word.store(nullptr, std::memory_order_relaxed);
    return check(sigaction(SIGUSR1, &previous, nullptr) == 0,
                 "Could not restore the signal handler.") && ok;
}

GateStatus noop(GateCall&, void*) { return GateStatus::resume; }

bool benchmark() {
    auto baseline = make_local_gate_template();
    auto cleared = baseline;
    unsigned removed = 0;
    for (std::size_t offset = 0; offset < baseline.dispatcher_offset; offset += 4) {
        std::uint32_t word;
        std::memcpy(&word, baseline.code.data() + offset, 4);
        if (word != 0xd5033f5fu) continue;
        word = 0xd503201fu; // NOP preserves all addresses and alignment.
        std::memcpy(baseline.code.data() + offset, &word, 4);
        ++removed;
    }
    if (!check(removed == 1, "Benchmark expected exactly one gate-return CLREX.")) return false;
    for (auto* code : {&baseline, &cleared})
        if (!bind_local_gate_dispatcher(code->code, code->dispatcher_offset, gate_dispatcher_address())) return false;
    std::array mappings{map_code(baseline.code), map_code(cleared.code)};
    for (auto& mapping : mappings)
        if (!mapping.ok() || !mapping.module->seal().ok()) return false;
    std::array<unsigned, 2> successes{};
    {
        Callback state;
        GateScope scope(contaminate, &state);
        for (unsigned variant = 0; variant < mappings.size(); ++variant) {
            for (unsigned attempt = 0; attempt < attempts; ++attempt) {
                state.word.value = 0;
                successes[variant] += armigo_test_monitor_gate(
                    mappings[variant].module->base_address(), &state.word.value) == 0;
            }
        }
    }
    // This control is observational: architectures permit spurious failure,
    // so the NOP variant is not required to produce a successful store.
    std::printf("Reservation control: NOP %u/%u stores succeeded; CLREX %u/%u.\n",
                successes[0], attempts, successes[1], attempts);
    if (!check(successes[1] == 0, "The benchmark gate exported a host reservation.")) return false;
    GateScope scope(noop, nullptr);
    constexpr std::uint64_t iterations = 1'000'000;
    constexpr unsigned samples = 21;
    std::array<std::vector<double>, 2> times;
    for (auto& mapping : mappings)
        armigo_test_monitor_gate_loop(mapping.module->base_address(), 100'000);
    for (unsigned sample = 0; sample < samples; ++sample) {
        for (unsigned turn = 0; turn < 2; ++turn) {
            const auto variant = (sample + turn) % 2;
            const auto start = std::chrono::steady_clock::now();
            armigo_test_monitor_gate_loop(mappings[variant].module->base_address(), iterations);
            const auto elapsed = std::chrono::steady_clock::now() - start;
            times[variant].push_back(std::chrono::duration<double, std::nano>(elapsed).count() / iterations);
        }
    }
    std::vector<double> deltas;
    for (unsigned sample = 0; sample < samples; ++sample)
        deltas.push_back(times[1][sample] - times[0][sample]);
    for (auto& values : times) std::sort(values.begin(), values.end());
    std::sort(deltas.begin(), deltas.end());
    std::printf("Warm gate: %u alternating pairs, %llu calls/sample; NOP median %.3f ns, "
                "CLREX median %.3f ns, paired delta median %.3f ns (range %.3f..%.3f ns).\n",
                samples, static_cast<unsigned long long>(iterations), times[0][samples / 2],
                times[1][samples / 2], deltas[samples / 2], deltas.front(), deltas.back());
    return scope.failure_count() == 0;
}
} // namespace

int main(int argc, char** argv) {
    if (!gate_resumes_clear() || !prepared_svc_clears() || !initial_entry_clears() ||
        !signal_return_clears()) return 1;
    std::puts("Host reservations cleared in 70,000 linked/copied/redirect/failure/SVC returns.");
    std::puts("Host reservations cleared in 10,000 initial guest entries.");
    std::puts("Signal-handler reservations cleared in 100 native guest resumptions.");
    if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) return benchmark() ? 0 : 1;
    return argc == 1 ? 0 : 1;
}
