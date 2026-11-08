// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>
#include <armigo/execution/gate_code.h>
#include <armigo/execution/gate_patch.h>
#include <armigo/execution/guest_fault.h>
#include <armigo/aarch64/instruction.h>
#include <armigo/horizon/arbitration.h>
#include <armigo/platform/guest_stack.h>
#include <armigo/runtime/thread_runtime.h>
#include <armigo/runtime/kernel_queries.h>

#include "../loader/nso_fixture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

extern "C" {
extern const std::byte armigo_thread_runtime_code_begin[];
extern const std::byte armigo_thread_runtime_code_end[];
extern const std::byte armigo_thread_runtime_entry[];
extern const std::byte armigo_thread_runtime_tpidr[];
extern const std::byte armigo_thread_runtime_tpidrro[];
extern const std::byte armigo_thread_runtime_first_svc[];
extern const std::byte armigo_thread_runtime_second_svc[];
extern const std::byte armigo_thread_runtime_exit_svc[];
extern const std::byte armigo_thread_runtime_processor_loop[];
extern const std::byte armigo_thread_runtime_processor_svc[];
extern const std::byte armigo_thread_runtime_scheduling_entry[];
extern const std::byte armigo_thread_runtime_get_priority_svc[];
extern const std::byte armigo_thread_runtime_set_priority_svc[];
extern const std::byte armigo_thread_runtime_get_core_svc[];
extern const std::byte armigo_thread_runtime_set_core_svc[];
extern const std::byte armigo_thread_runtime_current_processor_svc[];
extern const std::byte armigo_thread_runtime_current_priority_svc[];
extern const std::byte armigo_thread_runtime_original_entry[];
extern const std::byte armigo_thread_runtime_original_first_svc[];
extern const std::byte armigo_thread_runtime_original_exit_svc[];
extern const std::byte armigo_thread_runtime_original_worker_entry[];
extern const std::byte armigo_thread_runtime_original_worker_svc[];
}

namespace {
using namespace armigo;
using namespace armigo::execution;
using namespace armigo::runtime;
using namespace std::chrono_literals;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Check failed at line %u: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

[[noreturn]] void bail(unsigned line, const char* detail) {
    std::fprintf(stderr, "Fatal test failure at line %u: %s\n", line, detail);
    std::fflush(stderr);
    std::abort();
}

template<class Predicate>
void require_within(Predicate&& predicate, const char* detail, unsigned line) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) bail(line, detail);
        std::this_thread::yield();
    }
}

#define REQUIRE_WITHIN(condition) \
    require_within([&] { return (condition); }, #condition, __LINE__)

bool ok(const LaunchStatus& result) {
    return std::holds_alternative<std::monostate>(result);
}

bool invalid(const LaunchStatus& result) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure && failure->code == LaunchError::invalid_input;
}

template<class T>
bool failed(const LaunchResult<T>& result, LaunchError code) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure != nullptr && failure->code == code;
}

template<class T>
bool failed(const LaunchResult<T>& result) {
    return std::holds_alternative<LaunchFailure>(result);
}

void put(std::vector<std::byte>& image, std::size_t offset,
         std::uint64_t value, unsigned width = 8) {
    for (unsigned index = 0; index < width; ++index) {
        image.at(offset + index) = std::byte((value >> (index * 8)) & 0xffu);
    }
}

std::size_t code_offset(const std::byte* symbol) {
    return 0x100 + static_cast<std::size_t>(symbol - armigo_thread_runtime_code_begin);
}

std::vector<std::byte> make_nso() {
    using namespace armigo::loader::test;
    FixtureSpec spec;
    spec.segments[0] = {std::vector<std::byte>(0x1000), 0, false, true};
    spec.segments[1] = {std::vector<std::byte>(0x1000), 0x4000, false, true};
    spec.segments[2] = {std::vector<std::byte>(0x1000), 0x8000, false, true};
    auto& text = spec.segments[0].content;
    auto& ro = spec.segments[1].content;
    auto& data = spec.segments[2].content;
    const auto code_size = static_cast<std::size_t>(
        armigo_thread_runtime_code_end - armigo_thread_runtime_code_begin);
    std::memcpy(text.data() + 0x100, armigo_thread_runtime_code_begin, code_size);

    put(text, 4, 0x4000, 4);
    put(ro, 0, 0x30444f4d, 4);
    put(ro, 4, 0x40, 4);
    put(ro, 8, 0x5000, 4);
    put(ro, 12, 0x5000, 4);
    put(ro, 16, 0xf00, 4);
    put(ro, 20, 0xf00, 4);
    put(ro, 24, 0x4100, 4);

    std::size_t dynamic = 0x40;
    auto tag = [&](std::uint64_t key, std::uint64_t value) {
        put(ro, dynamic, key);
        put(ro, dynamic + 8, value);
        dynamic += 16;
    };
    tag(4, 0x4180);
    tag(5, 0x4200);
    tag(10, 7);
    tag(6, 0x4300);
    tag(11, 24);
    tag(36, 0x4500);
    tag(35, 8);
    tag(37, 8);
    tag(0, 0);
    put(ro, 0x180, 1, 4);
    put(ro, 0x184, 2, 4);
    put(ro, 0x188, 1, 4);
    constexpr char strings[] = "\0entry";
    std::memcpy(ro.data() + 0x200, strings, sizeof(strings));
    put(ro, 0x300 + 24, 1, 4);
    put(ro, 0x300 + 24 + 4, 0x12, 1);
    put(ro, 0x300 + 24 + 6, 1, 2);
    put(ro, 0x300 + 24 + 8, 0x100);
    put(ro, 0x300 + 24 + 16, code_size);
    put(ro, 0x500, 0x8008);
    put(data, 8, 0x8010);
    return build_nso(spec);
}

constexpr std::size_t kSiteCount = 15;
struct PreparationState {
    std::vector<std::byte> gate;
    std::array<GateSitePatch, kSiteCount> patches;
    std::size_t appended_offset = 0;
    std::size_t landing_offset = 0;
    std::array<std::uint64_t, 5> roots{};
    bool applied = false;
};

bool prepare_module(const ModulePreparationView& view, void* raw) noexcept {
    auto& state = *static_cast<PreparationState*>(raw);
    if (view.module_index != 0 || view.appended_code_offset != state.appended_offset ||
        view.appended_code.size() < state.gate.size() + kSiteCount * 16) return false;
    std::memcpy(view.appended_code.data(), state.gate.data(), state.gate.size());
    for (const auto& patch : state.patches) {
        if (patch.instruction_address + 4 > view.image.size() ||
            patch.trampoline_address < view.appended_code_offset ||
            patch.trampoline_address + 16 >
                view.appended_code_offset + view.appended_code.size()) return false;
        std::uint32_t before = 0;
        std::memcpy(&before, view.image.data() + patch.instruction_address, 4);
        if (before != patch.expected_instruction) return false;
        std::memcpy(view.image.data() + patch.instruction_address,
                    &patch.replacement_instruction, 4);
        std::memcpy(view.appended_code.data() +
                        (patch.trampoline_address - view.appended_code_offset),
                    patch.trampoline.data(), 16);
    }
    for (std::size_t index = 0; index < state.roots.size(); ++index) {
        const auto landing_offset = state.landing_offset + index * 8;
        const auto landing = view.base_address + landing_offset;
        const auto branch = aarch64::encode_b(
            landing + 4, view.base_address + state.roots[index]);
        if (!branch.ok() || landing_offset < view.appended_code_offset ||
            landing_offset - view.appended_code_offset >
                view.appended_code.size() - 8) return false;
        const std::uint32_t words[2] = {0xaa1f03f0, branch.word};
        std::memcpy(view.appended_code.data() +
                        (landing_offset - view.appended_code_offset),
                    words, sizeof(words));
    }
    state.applied = true;
    return true;
}

struct LoadedCode {
    std::shared_ptr<ModuleSet> modules;
    std::shared_ptr<const GuestSiteTable> sites;
    std::shared_ptr<platform::ThreadSlotKey> thread_slot;
    std::uint64_t entry = 0;
    std::uint64_t processor_loop = 0;
    std::uint64_t scheduling_entry = 0;
    std::uint64_t original_entry = 0;
    std::uint64_t original_worker_entry = 0;
    std::uint64_t process_landing = 0;
    std::vector<PreparedEntryLanding> original_landings;
    std::uint64_t data = 0;
};

std::optional<LoadedCode> load_code() {
    const auto image = make_nso();
    PreparationState state;
    state.gate = make_local_gate_code();
    const auto page = platform::HostModule::native_page_size();
    if (!page) return {};
    state.appended_offset = (std::size_t{0x9000} + *page - 1) & ~(*page - 1);
    const auto gate_bytes = (state.gate.size() + 15) & ~std::size_t{15};
    const std::array<const std::byte*, kSiteCount> labels = {
        armigo_thread_runtime_tpidr,
        armigo_thread_runtime_tpidrro,
        armigo_thread_runtime_first_svc,
        armigo_thread_runtime_second_svc,
        armigo_thread_runtime_exit_svc,
        armigo_thread_runtime_processor_svc,
        armigo_thread_runtime_get_priority_svc,
        armigo_thread_runtime_set_priority_svc,
        armigo_thread_runtime_get_core_svc,
        armigo_thread_runtime_set_core_svc,
        armigo_thread_runtime_current_processor_svc,
        armigo_thread_runtime_current_priority_svc,
        armigo_thread_runtime_original_first_svc,
        armigo_thread_runtime_original_exit_svc,
        armigo_thread_runtime_original_worker_svc,
    };
    for (std::size_t index = 0; index < labels.size(); ++index) {
        const auto offset = code_offset(labels[index]);
        std::uint32_t instruction = 0;
        std::memcpy(&instruction, labels[index], 4);
        const auto patch = make_gate_site_patch(
            instruction, offset, state.appended_offset + gate_bytes + index * 16,
            {state.appended_offset, state.gate.size()});
        if (!patch.ok()) return {};
        state.patches[index] = *patch.patch;
    }
    state.landing_offset = state.appended_offset + gate_bytes + kSiteCount * 16;
    state.roots = {
        code_offset(armigo_thread_runtime_entry),
        code_offset(armigo_thread_runtime_processor_loop),
        code_offset(armigo_thread_runtime_scheduling_entry),
        code_offset(armigo_thread_runtime_original_entry),
        code_offset(armigo_thread_runtime_original_worker_entry),
    };
    std::sort(state.roots.begin(), state.roots.end());
    const ModulePreparation preparation{
        gate_bytes + kSiteCount * 16 + state.roots.size() * 8,
        prepare_module, &state, true, state.roots};
    ModuleLoadOptions options;
    options.preparation = std::span(&preparation, 1);
    const std::array sources = {NsoSource{"thread-runtime", image}};
    auto loaded = ModuleSet::load_nso(sources, options);
    if (!loaded.ok() || !state.applied) return {};
    auto modules = std::make_shared<ModuleSet>(std::move(*loaded.modules));
    std::array<GuestGateSite, kSiteCount> sites;
    for (std::size_t index = 0; index < sites.size(); ++index) {
        sites[index] = state.patches[index].dispatch_site;
        sites[index].return_address += modules->mapping(0).base_address();
    }
    auto built = GuestSiteTable::build(sites);
    if (!built.ok()) return {};
    const auto entry = modules->entry_address(0, "entry");
    if (!entry) return {};
    LoadedCode result;
    result.modules = std::move(modules);
    result.sites = std::move(built.table);
    result.thread_slot = std::make_shared<platform::ThreadSlotKey>();
    result.entry = *entry;
    result.processor_loop = result.modules->mapping(0).base_address() +
                            code_offset(armigo_thread_runtime_processor_loop);
    result.scheduling_entry = result.modules->mapping(0).base_address() +
                              code_offset(armigo_thread_runtime_scheduling_entry);
    result.original_entry = result.modules->mapping(0).base_address() +
                            code_offset(armigo_thread_runtime_original_entry);
    result.original_worker_entry = result.modules->mapping(0).base_address() +
                                   code_offset(armigo_thread_runtime_original_worker_entry);
    for (std::size_t index = 0; index < state.roots.size(); ++index) {
        result.original_landings.push_back({
            result.modules->mapping(0).base_address() + state.roots[index],
            result.modules->mapping(0).base_address() + state.landing_offset + index * 8});
    }
    const auto main_landing = std::lower_bound(result.original_landings.begin(),
        result.original_landings.end(), result.original_entry,
        [](const auto& candidate, std::uint64_t root) {
            return candidate.logical_root < root;
        });
    if (main_landing == result.original_landings.end() ||
        main_landing->logical_root != result.original_entry) return {};
    result.process_landing = main_landing->landing;
    result.data = result.modules->mapping(0).base_address() + 0x8000;
    if (!result.thread_slot->valid()) return {};
    return result;
}

LaunchFailure memory_failure(const char* operation) {
    return {.stage = LaunchStage::memory, .code = LaunchError::unsupported,
            .operation = operation, .detail = "Synthetic test memory operation is unavailable"};
}

class TestPin final : public PinnedGuestMemory {
public:
    TestPin(platform::GuardedStack& stack, horizon::AddressRange range,
            memory::Permission permissions, std::atomic<unsigned>& live,
            std::intptr_t native_bias = 0)
        : stack_(stack), range_(range), permissions_(permissions), live_(live),
          native_bias_(native_bias) {
        live_.fetch_add(1, std::memory_order_relaxed);
    }
    ~TestPin() override { live_.fetch_sub(1, std::memory_order_relaxed); }

    horizon::AddressRange range() const noexcept override { return range_; }
    memory::Permission permissions() const noexcept override { return permissions_; }
    std::span<const std::byte> bytes() const noexcept override {
        return {reinterpret_cast<const std::byte*>(stack_.usable_base()) + native_bias_,
                static_cast<std::size_t>(range_.size)};
    }
    std::span<std::byte> writable_bytes() noexcept override {
        if (!memory::has_permission(permissions_, memory::Permission::write)) return {};
        return {reinterpret_cast<std::byte*>(stack_.usable_base()) + native_bias_,
                static_cast<std::size_t>(range_.size)};
    }

private:
    platform::GuardedStack& stack_;
    horizon::AddressRange range_;
    memory::Permission permissions_;
    std::atomic<unsigned>& live_;
    std::intptr_t native_bias_;
};

class TestMemory final : public GuestMemory {
public:
    explicit TestMemory(const LaunchProfile& profile) {
        add_stack();
        const horizon::MemoryAccounting accounting{
            0, 0, 0, stacks_.front()->usable_size(), 2 * 1024 * 1024,
            stacks_.front()->usable_size()};
        process_ = horizon::ProcessRecord::create(profile.process, accounting).process;
        if (!process_) throw std::runtime_error("Synthetic process construction failed");
    }

    std::shared_ptr<horizon::ProcessRecord> process() const noexcept override {
        return process_;
    }
    LaunchResult<GuestMemoryPin> allocate(
        std::uint64_t, horizon::MemoryType, memory::Permission) override {
        return memory_failure("test_allocate");
    }
    LaunchStatus release(horizon::AddressRange) override {
        return memory_failure("test_release");
    }
    LaunchResult<horizon::QueryDisposition> dispatch(
        std::uint16_t, horizon::QueryRegisters&) override {
        return memory_failure("test_dispatch");
    }
    LaunchResult<GuestMemoryPin> pin(
        horizon::AddressRange range, memory::Permission access) override {
        pin_calls.fetch_add(1, std::memory_order_relaxed);
        platform::GuardedStack* selected = nullptr;
        for (const auto& stack : stacks_) {
            if (range.address == stack->usable_base() && range.size == stack->usable_size()) {
                selected = stack.get();
                break;
            }
        }
        if (!selected || access != (memory::Permission::read | memory::Permission::write) ||
            refuse_pin) return memory_failure("test_pin");
        const auto permissions = read_only_pin ? memory::Permission::read : access;
        return GuestMemoryPin{std::make_shared<TestPin>(
            *selected, range, permissions, live_pins, native_bias)};
    }
    horizon::GuestCopyOut copy_out() noexcept override { return {}; }
    horizon::GuestBuffers ipc_buffers() noexcept override { return {}; }
    horizon::GuestMemoryMapper service_mapper() noexcept override { return {}; }

    horizon::AddressRange stack_range() const noexcept {
        return {stacks_.front()->usable_base(), stacks_.front()->usable_size()};
    }
    GuestMemoryPin initial_pin() {
        return std::get<GuestMemoryPin>(pin(
            stack_range(), memory::Permission::read | memory::Permission::write));
    }
    GuestMemoryPin additional_pin() {
        auto& stack = add_stack();
        const horizon::AddressRange range{stack.usable_base(), stack.usable_size()};
        return std::get<GuestMemoryPin>(pin(
            range, memory::Permission::read | memory::Permission::write));
    }

    std::atomic<unsigned> live_pins{0};
    std::atomic<unsigned> pin_calls{0};
    bool refuse_pin = false;
    bool read_only_pin = false;
    std::intptr_t native_bias = 0;

private:
    platform::GuardedStack& add_stack() {
        auto stack = platform::GuardedStack::create({.usable_size = 64 * 1024});
        if (!stack.ok()) throw std::runtime_error("Synthetic stack allocation failed");
        stacks_.push_back(std::move(stack.stack));
        return *stacks_.back();
    }

    std::vector<std::unique_ptr<platform::GuardedStack>> stacks_;
    std::shared_ptr<horizon::ProcessRecord> process_;
};

struct EntryReport {
    std::uint64_t stack_pointer = 0;
    std::uint64_t thread_pointer = 0;
    std::uint64_t read_only_thread_pointer = 0;
    std::uint64_t first_result = 0;
    std::uint64_t second_result = 0;
    std::uint64_t segment_control = 0;
};
struct SegmentControl {
    std::atomic<std::uint32_t> entered{0};
    std::atomic<std::uint32_t> release{0};
    std::uint64_t signal_address = 0;
};
struct SchedulingReport {
    std::uint64_t target_handle = 0;
    std::uint64_t requested_priority = 0;
    std::uint64_t requested_ideal_core = 0;
    std::uint64_t requested_affinity_mask = 0;
    std::uint64_t get_priority_result = 0;
    std::uint64_t observed_priority = 0;
    std::uint64_t set_priority_result = 0;
    std::uint64_t get_core_result = 0;
    std::uint64_t observed_ideal_core = 0;
    std::uint64_t observed_affinity_mask = 0;
    std::uint64_t set_core_result = 0;
    std::uint64_t current_processor = 0;
    std::uint64_t current_priority_result = 0;
    std::uint64_t current_priority = 0;
};
struct OriginalWorkerReport {
    std::uint64_t link_register = UINT64_MAX;
    std::uint64_t stack_pointer = 0;
};
static_assert(offsetof(EntryReport, segment_control) == 40);
static_assert(offsetof(SegmentControl, release) == 4);
static_assert(offsetof(SegmentControl, signal_address) == 8);
static_assert(offsetof(SchedulingReport, get_priority_result) == 32);
static_assert(offsetof(SchedulingReport, current_priority) == 104);
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);

struct BindingOwner {
    explicit BindingOwner(std::atomic<unsigned>& destroyed) : destroyed(destroyed) {}
    ~BindingOwner() { destroyed.fetch_add(1, std::memory_order_relaxed); }
    std::atomic<unsigned>& destroyed;
};

struct StopJoinCallback;

struct BindingState {
    static constexpr std::uint64_t arbitration_address = 0x7100004000;
    std::mutex mutex;
    std::condition_variable changed;
    horizon::Arbitration arbitration;
    std::stop_token stop;
    ThreadRuntime* runtime = nullptr;
    platform::ThreadSlotKey* thread_slot = nullptr;
    std::shared_ptr<horizon::HandleTable> handles;
    std::shared_ptr<GuestClock> clock;
    std::atomic<unsigned> bind_calls{0};
    std::atomic<unsigned> owner_destroyed{0};
    std::atomic<unsigned> first_calls{0};
    std::atomic<std::uint64_t> first_x0{0};
    std::atomic<std::uint64_t> first_x1{0};
    std::atomic<unsigned> second_calls{0};
    std::atomic<unsigned> original_worker_calls{0};
    std::array<std::uint64_t, 31> original_gprs{};
    std::array<GateVector, 32> original_vectors{};
    std::uint64_t original_sp = 0;
    std::uint64_t original_nzcv = UINT64_MAX;
    std::uint64_t original_fpsr = UINT64_MAX;
    std::uint64_t original_fpcr = UINT64_MAX;
    std::uint64_t original_tpidr = UINT64_MAX;
    std::uint64_t original_tpidrro = 0;
    alignas(16) std::array<std::byte, 0x200> original_tlr{};
    std::atomic<unsigned> exit_forwarded{0};
    std::atomic<unsigned> barrier_arrivals{0};
    std::atomic<unsigned> arbitration_arrivals{0};
    std::atomic<unsigned> arbitration_wait_checks{0};
    std::atomic<unsigned> arbitration_wait_successes{0};
    std::atomic<unsigned> arbitration_wait_cancellations{0};
    std::atomic<unsigned> arbitration_signal_successes{0};
    std::atomic<std::size_t> arbitration_signal_waiting{0};
    std::atomic<unsigned> exit_current_successes{0};
    std::atomic<unsigned> bound_context_observations{0};
    std::atomic<unsigned> worker_join_rejections{0};
    std::atomic<unsigned> stop_join_rejections{0};
    std::atomic<unsigned> cleanup_join_rejections{0};
    std::atomic<unsigned> startup_sleep_calls{0};
    std::atomic<unsigned> blocked_binders{0};
    std::atomic<unsigned> scheduling_boundaries{0};
    std::atomic<unsigned> scheduling_deferrals{0};
    bool block_bind = false;
    bool fail_bind = false;
    bool throw_bind = false;
    bool block_first = false;
    bool use_arbitration = false;
    bool stop_after_grant = false;
    bool use_exit_current = false;
    bool join_in_binder = false;
    bool join_from_stop_callback = false;
    bool stop_join_ran = false;
    bool use_startup_sleep = false;
    bool use_scheduling_boundary = false;
    bool defer_cancelled_scheduling_once = false;
    bool original_binding = false;
    unsigned barrier_target = 0;
    bool release = false;
};

struct StopJoinCallback {
    BindingState* state = nullptr;
    void operator()() const noexcept {
        if (state && state->runtime && invalid(state->runtime->join(std::nullopt))) {
            state->stop_join_rejections.fetch_add(1, std::memory_order_relaxed);
        }
        if (state) {
            {
                std::lock_guard lock(state->mutex);
                state->stop_join_ran = true;
            }
            state->changed.notify_all();
        }
    }
};

struct StopJoinOwner {
    StopJoinOwner(BindingState& state, std::stop_token stop)
        : state(state), callback(stop, StopJoinCallback{&state}) {}
    ~StopJoinOwner() { state.owner_destroyed.fetch_add(1, std::memory_order_relaxed); }
    BindingState& state;
    std::stop_callback<StopJoinCallback> callback;
};

GateStatus supervise(std::uint16_t immediate, GateCall& call,
                     GuestThreadContext& context, void* raw) {
    auto& state = *static_cast<BindingState*>(raw);
    std::stop_token stop;
    {
        std::lock_guard lock(state.mutex);
        stop = state.stop;
    }
    if (context.bound() && state.thread_slot && state.thread_slot->active_bindings() != 0) {
        state.bound_context_observations.fetch_add(1, std::memory_order_relaxed);
    }
    if (immediate == 0xa) {
        state.exit_forwarded.fetch_add(1, std::memory_order_relaxed);
        return GateStatus::failed;
    }
    if (immediate == 0x33) {
        for (unsigned index = 0; index < state.original_gprs.size(); ++index)
            state.original_gprs[index] = call.x(index);
        for (unsigned index = 0; index < state.original_vectors.size(); ++index)
            state.original_vectors[index] = call.v(index);
        state.original_sp = call.sp();
        state.original_nzcv = call.nzcv();
        state.original_fpsr = call.fpsr();
        state.original_fpcr = call.fpcr();
        state.original_tpidr = context.thread_pointer();
        state.original_tpidrro = context.read_only_thread_pointer();
        state.original_worker_calls.fetch_add(1, std::memory_order_release);
        return GateStatus::resume;
    }
    if (immediate == 0x31) {
        state.first_x0.store(call.x(0), std::memory_order_relaxed);
        state.first_x1.store(call.x(1), std::memory_order_relaxed);
        state.first_calls.fetch_add(1, std::memory_order_release);
        if (state.use_arbitration) {
            const auto arrival =
                state.arbitration_arrivals.fetch_add(1, std::memory_order_relaxed) + 1;
            if (arrival == 1) {
                const auto outcome = state.arbitration.wait_transaction(
                    BindingState::arbitration_address, -1,
                    [](void* raw) noexcept {
                        auto& shared = *static_cast<BindingState*>(raw);
                        shared.arbitration_wait_checks.fetch_add(1, std::memory_order_relaxed);
                        return horizon::ArbitrateOutcome::succeeded;
                    },
                    &state, stop, 24);
                if (outcome == horizon::ArbitrateOutcome::cancelled) {
                    state.arbitration_wait_cancellations.fetch_add(
                        1, std::memory_order_relaxed);
                    return GateStatus::exit_thread;
                }
                if (outcome != horizon::ArbitrateOutcome::succeeded) {
                    return GateStatus::failed;
                }
                state.arbitration_wait_successes.fetch_add(1, std::memory_order_relaxed);
                if (state.stop_after_grant) state.runtime->request_stop();
                call.set_x(0, 0xa110ca7e);
                return GateStatus::resume;
            }
            const auto deadline = std::chrono::steady_clock::now() + 5s;
            while (state.arbitration.waiters(BindingState::arbitration_address) != 1) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    return GateStatus::failed;
                }
                std::this_thread::yield();
            }
            const auto outcome = state.arbitration.signal_transaction(
                BindingState::arbitration_address, 1,
                [](void* raw, std::size_t waiting) noexcept {
                    auto& shared = *static_cast<BindingState*>(raw);
                    shared.arbitration_signal_waiting.store(waiting,
                                                            std::memory_order_relaxed);
                    return horizon::ArbitrateOutcome::succeeded;
                },
                &state);
            if (outcome != horizon::ArbitrateOutcome::succeeded) {
                return GateStatus::failed;
            }
            state.arbitration_signal_successes.fetch_add(1, std::memory_order_relaxed);
            call.set_x(0, 0x5190a1ed);
            return GateStatus::resume;
        }
        const auto arrivals = state.barrier_arrivals.fetch_add(1, std::memory_order_relaxed) + 1;
        if (state.barrier_target != 0) {
            std::stop_callback wake(stop, [&] {
                std::lock_guard lock(state.mutex);
                state.changed.notify_all();
            });
            std::unique_lock lock(state.mutex);
            if (arrivals >= state.barrier_target) state.changed.notify_all();
            state.changed.wait(lock, [&] {
                return state.barrier_arrivals.load(std::memory_order_relaxed) >=
                           state.barrier_target || stop.stop_requested();
            });
        }
        if (state.block_first) {
            std::stop_callback wake(stop, [&] {
                std::lock_guard lock(state.mutex);
                state.changed.notify_all();
            });
            std::unique_lock lock(state.mutex);
            // A stop callback that joins must run before this thread exits:
            // the exit releases the binding owner, which would deregister it
            // unrun if the stop request had not reached it yet.
            state.changed.wait(lock, [&] {
                return state.release || (stop.stop_requested() &&
                                         (!state.join_from_stop_callback || state.stop_join_ran));
            });
        }
        if (state.use_exit_current) {
            if (state.runtime && ok(state.runtime->exit_current())) {
                state.exit_current_successes.fetch_add(1, std::memory_order_relaxed);
                return GateStatus::exit_thread;
            }
            return GateStatus::failed;
        }
        if (stop.stop_requested()) return GateStatus::exit_thread;
        call.set_x(0, 0x31003100);
        return GateStatus::resume;
    }
    if (immediate == 0x32) {
        state.second_calls.fetch_add(1, std::memory_order_relaxed);
        if (stop.stop_requested()) return GateStatus::exit_thread;
        call.set_x(0, 0x32003200);
        return GateStatus::resume;
    }
    return GateStatus::failed;
}

GateStatus scheduling_boundary(std::uint16_t, GateCall&,
                               GuestThreadContext&, void* raw) {
    auto& state = *static_cast<BindingState*>(raw);
    state.scheduling_boundaries.fetch_add(1, std::memory_order_relaxed);
    std::stop_token stop;
    {
        std::lock_guard lock(state.mutex);
        stop = state.stop;
    }
    if (!stop.stop_requested()) return GateStatus::resume;
    if (state.defer_cancelled_scheduling_once &&
        state.scheduling_deferrals.load(std::memory_order_relaxed) == 0) {
        state.scheduling_deferrals.store(1, std::memory_order_relaxed);
        return GateStatus::resume;
    }
    return GateStatus::exit_thread;
}

struct StartupSleepOwner {
    StartupSleepOwner(BindingState& state,
                      const std::shared_ptr<horizon::ThreadRecord>& record,
                      std::stop_token stop)
        : state(state), queries(record->process(), *state.handles, record, {}, {}, {}, stop),
          ipc(*state.handles, nullptr), supervisor(queries, ipc, {}, state.clock.get()),
          startup(supervisor.binding()) {}
    ~StartupSleepOwner() {
        state.owner_destroyed.fetch_add(1, std::memory_order_relaxed);
    }

    static GateStatus route(std::uint16_t immediate, GateCall& call,
                            GuestThreadContext& context, void* raw) {
        auto& self = *static_cast<StartupSleepOwner*>(raw);
        if (immediate != 0x31) return GateStatus::failed;
        self.state.startup_sleep_calls.fetch_add(1, std::memory_order_relaxed);
        call.set_x(0, static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(60s).count()));
        return self.startup.handler(0x0b, call, context, self.startup.context);
    }

    BindingState& state;
    horizon::StartupQueries queries;
    horizon::IpcDispatcher ipc;
    StartupSupervisor supervisor;
    GuestSupervisor startup;
};

LaunchResult<ThreadServiceBinding> bind_thread(
    BindingState& state, std::shared_ptr<horizon::ThreadRecord> record,
    GuestThreadContext& context, std::stop_token stop) {
    state.bind_calls.fetch_add(1, std::memory_order_relaxed);
    if (state.throw_bind) throw std::runtime_error("Synthetic binder exception");
    if (state.fail_bind) {
        return LaunchFailure{.stage = LaunchStage::services,
            .code = LaunchError::host_failure, .operation = "synthetic_bind",
            .detail = "Synthetic binder refusal"};
    }
    {
        std::lock_guard lock(state.mutex);
        state.stop = stop;
    }
    if (state.join_in_binder && state.runtime && invalid(state.runtime->join(std::nullopt))) {
        state.worker_join_rejections.fetch_add(1, std::memory_order_relaxed);
    }
    if (state.original_binding) {
        context.set_thread_pointer(0);
        context.set_read_only_thread_pointer(
            reinterpret_cast<std::uintptr_t>(state.original_tlr.data()));
        context.set_guest_x18(0x5a5a5a5a5a5a5a5bull);
    } else {
        context.set_thread_pointer(0x1000 + record->id() * 2);
        context.set_read_only_thread_pointer(0x1001 + record->id() * 2);
    }
    if (state.use_startup_sleep) {
        auto owner = std::make_shared<StartupSleepOwner>(state, record, stop);
        return ThreadServiceBinding{
            .supervisor = {StartupSleepOwner::route, owner.get()},
            .owner = std::move(owner),
        };
    }
    std::shared_ptr<void> owner;
    if (state.join_from_stop_callback) {
        owner = std::make_shared<StopJoinOwner>(state, stop);
    } else {
        owner = std::make_shared<BindingOwner>(state.owner_destroyed);
    }
    if (state.block_bind) {
        std::stop_callback wake(stop, [&] {
            std::lock_guard lock(state.mutex);
            state.changed.notify_all();
        });
        std::unique_lock lock(state.mutex);
        state.blocked_binders.fetch_add(1, std::memory_order_release);
        state.changed.wait(lock, [&] { return stop.stop_requested(); });
    }
    return ThreadServiceBinding{
        .supervisor = {supervise, &state},
        .owner = std::move(owner),
        .scheduling_boundary = state.use_scheduling_boundary
            ? GuestSupervisor{scheduling_boundary, &state} : GuestSupervisor{},
    };
}

LaunchProfile make_profile() {
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.process.heap_address = 0x1000000000;
    profile.process.alias_address = 0x2000000000;
    profile.process.stack_address = 0x6000000000;
    profile.process.allowed_core_mask = 0x3;
    profile.permissions.handle_table_capacity = 16;
    profile.permissions.highest_priority = 16;
    profile.permissions.lowest_priority = 40;
    profile.permissions.supervisor_calls = {
        0xa, 0xb, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x31, 0x32, 0x33};
    profile.main_thread = {24, 0, 64 * 1024};
    return profile;
}

PreparedLaunch make_launch(const LoadedCode& code) {
    PreparedLaunch launch;
    launch.modules = code.modules;
    launch.sites = code.sites;
    launch.thread_slot = code.thread_slot;
    launch.plan.module_order = {0};
    launch.plan.entry = {0, code_offset(armigo_thread_runtime_entry)};
    launch.plan.entry_abi = LaunchEntryAbi::synthetic_exact_sp;
    launch.plan.relocation_owner = LaunchOwnership::host_loader;
    launch.plan.initialization_owner = LaunchOwnership::guest_bootstrap;
    launch.entry_address = code.entry;
    return launch;
}

struct Environment {
    explicit Environment(const LoadedCode& code)
        : profile(make_profile()), memory(std::make_shared<TestMemory>(profile)),
          handles(std::make_shared<horizon::HandleTable>(
              profile.permissions.handle_table_capacity)), launch(make_launch(code)) {
        binding.thread_slot = code.thread_slot.get();
        auto made_clock = create_guest_clock();
        if (const auto* value = std::get_if<std::shared_ptr<GuestClock>>(&made_clock)) {
            clock = *value;
        } else {
            throw std::runtime_error("Synthetic guest clock construction failed");
        }
        binding.handles = handles;
        binding.clock = clock;
    }

    ThreadRuntimeDependencies dependencies() {
        return {profile, launch, memory, handles, clock,
            [this](std::shared_ptr<horizon::ThreadRecord> record,
                   GuestThreadContext& context, std::stop_token stop) {
                return bind_thread(binding, std::move(record), context, stop);
            }};
    }

    LaunchProfile profile;
    std::shared_ptr<TestMemory> memory;
    std::shared_ptr<horizon::HandleTable> handles;
    std::shared_ptr<GuestClock> clock;
    PreparedLaunch launch;
    BindingState binding;
};

void configure_original(Environment& environment, const LoadedCode& code) {
    environment.launch.plan.entry_abi = LaunchEntryAbi::original_process;
    environment.launch.plan.relocation_owner = LaunchOwnership::guest_bootstrap;
    environment.launch.plan.initialization_owner = LaunchOwnership::guest_bootstrap;
    environment.launch.entry_address = code.original_entry;
    environment.launch.process_entry_address = code.process_landing;
    environment.launch.original_entry_landings = code.original_landings;
    environment.binding.original_binding = true;
}

std::shared_ptr<ThreadRuntime> make_runtime(Environment& environment) {
    auto made = create_thread_runtime(environment.dependencies());
    CHECK(std::holds_alternative<std::shared_ptr<ThreadRuntime>>(made));
    if (auto* runtime = std::get_if<std::shared_ptr<ThreadRuntime>>(&made)) {
        environment.binding.runtime = runtime->get();
        return *runtime;
    }
    return {};
}

ThreadLaunch thread_launch(Environment& environment, const LoadedCode& code,
                           EntryReport& report, GuestMemoryPin stack) {
    (void)environment;
    const auto range = stack->range();
    return {
        .entry = {.function = code.entry,
                  .x0 = reinterpret_cast<std::uintptr_t>(&report),
                  .x1 = 0,
                  .stack_pointer = range.address + range.size - 64},
        .stack = std::move(stack),
        .priority = 24,
        .ideal_core = 0,
        .affinity_mask = 0x3,
    };
}

ThreadLaunch scheduling_launch(Environment& environment, const LoadedCode& code,
                              SchedulingReport& report, GuestMemoryPin stack) {
    (void)environment;
    const auto range = stack->range();
    return {
        .entry = {.function = code.scheduling_entry,
                  .x0 = reinterpret_cast<std::uintptr_t>(&report),
                  .x1 = 0,
                  .stack_pointer = range.address + range.size - 64},
        .stack = std::move(stack),
        .priority = 24,
        .ideal_core = 0,
        .affinity_mask = 0x3,
    };
}

CreatedThread require_created(const LaunchResult<CreatedThread>& result, unsigned line) {
    if (const auto* created = std::get_if<CreatedThread>(&result)) return *created;
    bail(line, "thread creation failed");
}

#define REQUIRE_CREATED(result) require_created((result), __LINE__)

void test_factory_validation(const LoadedCode& code) {
    Environment environment(code);
    auto dependencies = environment.dependencies();
    dependencies.memory.reset();
    CHECK(failed(create_thread_runtime(std::move(dependencies)), LaunchError::invalid_input));

    dependencies = environment.dependencies();
    dependencies.handles = std::make_shared<horizon::HandleTable>(15);
    CHECK(failed(create_thread_runtime(std::move(dependencies)), LaunchError::invalid_input));

    dependencies = environment.dependencies();
    dependencies.launch.plan.entry_abi = LaunchEntryAbi::original_process;
    CHECK(failed(create_thread_runtime(std::move(dependencies)), LaunchError::unsupported));

    dependencies = environment.dependencies();
    {
        Environment original_environment(code);
        configure_original(original_environment, code);
        dependencies = original_environment.dependencies();
        CHECK(std::holds_alternative<std::shared_ptr<ThreadRuntime>>(
            create_thread_runtime(std::move(dependencies))));

        dependencies = original_environment.dependencies();
        dependencies.launch.original_entry_landings.front().landing += 4;
        CHECK(failed(create_thread_runtime(std::move(dependencies)),
                     LaunchError::permission_denied));
    }

    dependencies = environment.dependencies();
    dependencies.launch.sites.reset();
    CHECK(failed(create_thread_runtime(std::move(dependencies)), LaunchError::invalid_input));

    dependencies = environment.dependencies();
    dependencies.launch.entry_address = code.data;
    CHECK(failed(create_thread_runtime(std::move(dependencies)), LaunchError::permission_denied));

    const GuestGateSite unowned_site{
        .return_address = code.data + 4,
        .instruction = aarch64::encode_svc(0x31).word,
    };
    const auto unowned_table = GuestSiteTable::build(std::span(&unowned_site, 1));
    CHECK(unowned_table.ok());
    dependencies = environment.dependencies();
    dependencies.launch.sites = unowned_table.table;
    CHECK(failed(create_thread_runtime(std::move(dependencies)), LaunchError::permission_denied));
}

void test_original_process_creation(const LoadedCode& code) {
    Environment environment(code);
    configure_original(environment, code);
    auto runtime = make_runtime(environment);
    if (!runtime) return;

    EntryReport report;
    auto launch = thread_launch(
        environment, code, report, environment.memory->initial_pin());
    launch.entry.function = code.original_entry;
    launch.entry.x0 = 0x1111111111111111;
    launch.entry.x1 = 0x2222222222222222;
    launch.entry.transfer_function = code.process_landing + 4;
    launch.entry.original_process = false;
    launch.original_process = true;

    auto invalid = launch;
    invalid.entry.function = code.entry;
    CHECK(failed(runtime->create(invalid), LaunchError::invalid_input));
    invalid = launch;
    invalid.original_process = false;
    CHECK(failed(runtime->create(invalid), LaunchError::invalid_input));
    invalid.entry.function = code.original_worker_entry;
    CHECK(failed(runtime->create(invalid), LaunchError::invalid_input));

    const auto created = REQUIRE_CREATED(runtime->create(launch));
    auto repeated = launch;
    repeated.stack = environment.memory->additional_pin();
    repeated.entry.stack_pointer = repeated.stack->range().address + repeated.stack->range().size;
    CHECK(failed(runtime->create(repeated), LaunchError::invalid_input));
    OriginalWorkerReport worker_report;
    EntryReport unused_worker_entry_report;
    auto worker_launch = thread_launch(
        environment, code, unused_worker_entry_report,
        environment.memory->additional_pin());
    const auto worker_sp = worker_launch.entry.stack_pointer;
    worker_launch.entry.function = code.original_worker_entry;
    worker_launch.entry.x0 = reinterpret_cast<std::uintptr_t>(&worker_report);
    worker_launch.entry.x1 = 0xffffffffffffffffull;
    worker_launch.entry.transfer_function = code.process_landing;
    worker_launch.entry.original_process = false;
    constexpr std::uint64_t worker_fpcr = 0x02c00000;
    constexpr std::uint64_t worker_fpsr = 0x08000003;
    worker_launch.entry.fpcr = worker_fpcr;
    worker_launch.entry.fpsr = worker_fpsr;
    worker_launch.original_process = false;
    const auto worker = REQUIRE_CREATED(runtime->create(worker_launch));
    worker_launch.entry.fpcr = 0;
    worker_launch.entry.fpsr = 0;

    CHECK(ok(runtime->start(created.handle)));
    CHECK(ok(runtime->start(worker.handle)));
    CHECK(ok(runtime->join(std::nullopt)));
    CHECK(created.record->exited() && worker.record->exited());
    CHECK(environment.binding.first_calls.load(std::memory_order_acquire) == 1);
    CHECK(environment.binding.first_x0.load(std::memory_order_relaxed) == 0);
    CHECK(environment.binding.first_x1.load(std::memory_order_relaxed) == created.handle);
    CHECK(environment.binding.original_worker_calls.load(std::memory_order_acquire) == 1);
    CHECK(worker_report.link_register == 0);
    CHECK(environment.binding.original_gprs[0] ==
          reinterpret_cast<std::uintptr_t>(&worker_report));
    for (unsigned index = 1; index <= 17; ++index)
        CHECK(environment.binding.original_gprs[index] == 0);
    CHECK((environment.binding.original_gprs[18] & 1) == 1);
    for (unsigned index = 19; index <= 29; ++index)
        CHECK(environment.binding.original_gprs[index] == 0);
    CHECK(worker_report.stack_pointer == worker_sp);
    CHECK(environment.binding.original_sp == worker_sp - 16);
    CHECK(environment.binding.original_nzcv == 0);
    CHECK(environment.binding.original_fpsr == worker_fpsr);
    CHECK(environment.binding.original_fpcr == worker_fpcr);
    CHECK(std::all_of(environment.binding.original_vectors.begin(),
        environment.binding.original_vectors.end(), [](const auto& value) {
            return value == GateVector{};
        }));
    CHECK(environment.binding.original_tpidr == 0);
    CHECK(environment.binding.original_tpidrro ==
          reinterpret_cast<std::uintptr_t>(environment.binding.original_tlr.data()));
    CHECK(std::all_of(environment.binding.original_tlr.begin(),
        environment.binding.original_tlr.end(), [](std::byte value) {
            return value == std::byte{};
        }));
    const auto snapshot = std::get<ThreadSnapshot>(runtime->snapshot(created.handle));
    CHECK(snapshot.state == ThreadState::exited && !snapshot.failure);
}

void test_original_root_reuse(const LoadedCode& code) {
    Environment environment(code);
    configure_original(environment, code);
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport report;
    auto launch = thread_launch(environment, code, report, environment.memory->initial_pin());
    launch.entry.function = code.original_entry;
    launch.original_process = true;
    const auto main = REQUIRE_CREATED(runtime->create(launch));
    launch.stack = environment.memory->additional_pin();
    launch.entry.stack_pointer = launch.stack->range().address + launch.stack->range().size;
    launch.original_process = false;
    const auto worker = REQUIRE_CREATED(runtime->create(launch));
    CHECK(worker.handle != main.handle);
    runtime->request_stop();
    CHECK(ok(runtime->join(std::nullopt)));
}

void test_create_validation(const LoadedCode& code) {
    Environment environment(code);
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport report;
    auto pin = environment.memory->initial_pin();
    auto launch = thread_launch(environment, code, report, pin);

    auto invalid = launch;
    invalid.priority = 15;
    CHECK(failed(runtime->create(invalid), LaunchError::permission_denied));
    invalid = launch;
    invalid.ideal_core = 1;
    invalid.affinity_mask = 1;
    CHECK(failed(runtime->create(invalid), LaunchError::permission_denied));
    invalid = launch;
    invalid.affinity_mask = 4;
    CHECK(failed(runtime->create(invalid), LaunchError::permission_denied));
    invalid = launch;
    invalid.entry.function = code.entry + 2;
    CHECK(failed(runtime->create(invalid)));
    invalid = launch;
    invalid.entry.function = code.data;
    CHECK(failed(runtime->create(invalid), LaunchError::permission_denied));
    invalid = launch;
    invalid.entry.function = code.entry + 4;
    CHECK(failed(runtime->create(invalid), LaunchError::permission_denied));
    invalid = launch;
    invalid.entry.function = code.modules->mapping(0).base_address() + 0x1000;
    CHECK(failed(runtime->create(invalid), LaunchError::permission_denied));
    invalid = launch;
    invalid.entry.function = code.modules->mapping(0).base_address() +
        code.modules->appended_code(0)->offset;
    CHECK(failed(runtime->create(invalid), LaunchError::permission_denied));
    invalid = launch;
    invalid.entry.stack_pointer = pin->range().address - 16;
    CHECK(failed(runtime->create(invalid), LaunchError::invalid_input));
    // A full handle table is the guest's 0xd201, with nothing published.
    {
        std::vector<horizon::Handle> fillers;
        while (const auto filler = environment.handles->insert(std::make_shared<horizon::Event>()))
            fillers.push_back(*filler);
        const auto full = runtime->create(launch);
        const auto* full_error = std::get_if<LaunchFailure>(&full);
        CHECK(full_error && full_error->code == LaunchError::resource_limit && full_error->guest_result == 0xd201);
        for (const auto filler : fillers) CHECK(environment.handles->close(filler));
    }

    auto created = REQUIRE_CREATED(runtime->create(launch));
    CHECK(environment.handles->lookup_as<horizon::ThreadRecord>(created.handle) == created.record);
    CHECK(created.record->priority() == 24 && created.record->affinity_mask() == 3);
    CHECK(environment.memory->pin_calls.load() >= 2);
    const auto missing_start = runtime->start(0x1234);
    const auto* missing_error = std::get_if<LaunchFailure>(&missing_start);
    CHECK(missing_error && missing_error->guest_result == 0xe401);
    const auto bad_priority = runtime->set_priority(created.handle, 15);
    const auto* priority_error = std::get_if<LaunchFailure>(&bad_priority);
    CHECK(priority_error && priority_error->guest_result == 0xe001);
    CHECK(failed(runtime->create(launch), LaunchError::invalid_input));
    runtime->request_stop();
    CHECK(ok(runtime->join(std::nullopt)));
}

void test_native_execution_and_queries(const LoadedCode& code) {
    Environment environment(code);
    environment.binding.block_first = true;
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport report;
    auto initial = environment.memory->initial_pin();
    const auto stack_range = initial->range();
    const auto expected_sp = stack_range.address + stack_range.size - 64;
    auto created = REQUIRE_CREATED(runtime->create(
        thread_launch(environment, code, report, initial)));
    initial.reset();
    CHECK(environment.memory->live_pins.load() == 1);
    CHECK(std::get<ThreadSnapshot>(runtime->snapshot(created.handle)).state == ThreadState::created);
    CHECK(ok(runtime->set_priority(created.handle, 26)));
    CHECK(ok(runtime->set_core_mask(created.handle, 1, 0x2)));
    CHECK(created.record->priority() == 26 && created.record->affinity_mask() == 2);
    CHECK(!ok(runtime->exit_current()));
    CHECK(ok(runtime->start(created.handle)));
    REQUIRE_WITHIN(environment.binding.first_calls.load(std::memory_order_acquire) == 1);
    CHECK(report.stack_pointer == expected_sp);
    CHECK(report.thread_pointer == 0x1000 + created.record->id() * 2);
    CHECK(report.read_only_thread_pointer == 0x1001 + created.record->id() * 2);
    CHECK(created.record->current_processor() == 1);
    const auto running = std::get<ThreadSnapshot>(runtime->snapshot(created.handle));
    CHECK(running.state == ThreadState::running && running.priority == 26 &&
          running.ideal_core == 1 && running.affinity_mask == 2);
    const auto duplicate_start = runtime->start(created.handle);
    const auto* start_error = std::get_if<LaunchFailure>(&duplicate_start);
    CHECK(start_error && start_error->guest_result == 0xfa01);
    {
        std::lock_guard lock(environment.binding.mutex);
        environment.binding.release = true;
    }
    environment.binding.changed.notify_all();
    CHECK(ok(runtime->join(std::nullopt)));
    CHECK(report.first_result == 0x31003100 && report.second_result == 0x32003200);
    CHECK(environment.binding.second_calls.load() == 1 &&
          environment.binding.exit_forwarded.load() == 0);
    CHECK(environment.binding.bound_context_observations.load() == 2);
    CHECK(created.record->exited() && !created.record->current_processor());
    CHECK(environment.binding.owner_destroyed.load() == 1);
    CHECK(environment.memory->live_pins.load() == 0);
    const auto exited = std::get<ThreadSnapshot>(runtime->snapshot(created.handle));
    CHECK(exited.state == ThreadState::exited && !exited.failure);
    CHECK(ok(runtime->set_core_mask(created.handle, 0, 1)));
    const auto rescheduled = std::get<ThreadSnapshot>(runtime->snapshot(created.handle));
    CHECK(rescheduled.ideal_core == 0 && rescheduled.affinity_mask == 1);
    CHECK(!created.record->current_processor());
}

void test_native_scheduling_calls(const LoadedCode& code) {
    Environment environment(code);
    environment.binding.block_first = true;
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport foreign_report;
    auto foreign = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, foreign_report, environment.memory->initial_pin())));
    CHECK(ok(runtime->start(foreign.handle)));
    REQUIRE_WITHIN(environment.binding.first_calls.load(std::memory_order_acquire) == 1);

    auto run = [&](SchedulingReport& report) {
        auto caller = REQUIRE_CREATED(runtime->create(scheduling_launch(
            environment, code, report, environment.memory->additional_pin())));
        CHECK(ok(runtime->start(caller.handle)));
        CHECK(caller.record->wait_for_exit(5'000'000'000));
        CHECK(environment.handles->close(caller.handle));
    };

    SchedulingReport invalid_handle;
    invalid_handle.target_handle = 0x1234;
    invalid_handle.requested_priority = 99;
    invalid_handle.requested_ideal_core = 2;
    invalid_handle.requested_affinity_mask = 4;
    run(invalid_handle);
    CHECK(invalid_handle.get_priority_result == 0xe401 &&
          invalid_handle.set_priority_result == 0xe401);
    CHECK(invalid_handle.get_core_result == 0xe401 &&
          invalid_handle.set_core_result == 0xe401);

    SchedulingReport invalid_value;
    invalid_value.target_handle = foreign.handle;
    invalid_value.requested_priority = 99;
    invalid_value.requested_ideal_core = 2;
    invalid_value.requested_affinity_mask = 4;
    run(invalid_value);
    CHECK(invalid_value.get_priority_result == 0 && invalid_value.observed_priority == 24);
    CHECK(invalid_value.set_priority_result == 0xe001);
    CHECK(invalid_value.get_core_result == 0 && invalid_value.observed_ideal_core == 0 &&
          invalid_value.observed_affinity_mask == 3);
    CHECK(invalid_value.set_core_result == 0xe201);
    CHECK(invalid_value.current_processor == 0 &&
          invalid_value.current_priority_result == 0 && invalid_value.current_priority == 24);

    SchedulingReport accepted;
    accepted.target_handle = foreign.handle;
    accepted.requested_priority = 26;
    accepted.requested_ideal_core = 1;
    accepted.requested_affinity_mask = 2;
    run(accepted);
    CHECK(accepted.get_priority_result == 0 && accepted.observed_priority == 24);
    CHECK(accepted.set_priority_result == 0 && accepted.get_core_result == 0 &&
          accepted.observed_ideal_core == 0 && accepted.observed_affinity_mask == 3);
    CHECK(accepted.set_core_result == 0 && foreign.record->priority() == 26);
    const auto scheduling = foreign.record->scheduling_state();
    CHECK(scheduling.ideal_core == 1 && scheduling.affinity_mask == 2 &&
          scheduling.current_processor == 1);

    constexpr std::uint64_t upper = 0xface123400000000ull;
    const auto core_request = [&](std::int32_t ideal, std::uint64_t mask,
                                  std::uint32_t result, std::int32_t expected_ideal,
                                  std::uint64_t expected_mask) {
        const auto before = foreign.record->scheduling_state();
        SchedulingReport request;
        request.target_handle = upper | foreign.handle;
        request.requested_priority = 26;
        request.requested_ideal_core = upper | static_cast<std::uint32_t>(ideal);
        request.requested_affinity_mask = mask;
        run(request);
        CHECK(request.get_priority_result == 0 && request.observed_priority == (upper | 26));
        CHECK(request.get_core_result == 0 &&
              request.observed_ideal_core == (upper | static_cast<std::uint32_t>(before.ideal_core)) &&
              request.observed_affinity_mask == before.affinity_mask);
        CHECK(request.set_core_result == result);
        const auto after = foreign.record->scheduling_state();
        CHECK(after.ideal_core == expected_ideal && after.affinity_mask == expected_mask);
        CHECK(after.current_processor && (expected_mask & (1ull << *after.current_processor)));
    };
    core_request(-1, 3, 0, -1, 3);
    core_request(-3, 1, 0, -1, 1);
    core_request(-2, UINT64_MAX, 0, 0, 1);
    core_request(-3, 2, 0xe801, 0, 1);
    core_request(-4, 0, 0xe801, 0, 1);
    core_request(-4, 4, 0xe201, 0, 1);

    {
        std::lock_guard lock(environment.binding.mutex);
        environment.binding.release = true;
    }
    environment.binding.changed.notify_all();
    CHECK(ok(runtime->join(std::nullopt)) && foreign.record->exited());
}

void test_scheduling_stop_boundaries(const LoadedCode& code) {
    for (const bool custom_boundary : {false, true}) {
        Environment environment(code);
        environment.binding.use_scheduling_boundary = custom_boundary;
        environment.binding.defer_cancelled_scheduling_once = custom_boundary;
        auto runtime = make_runtime(environment);
        if (!runtime) return;
        SegmentControl control;
        EntryReport report;
        report.segment_control = reinterpret_cast<std::uintptr_t>(&control);
        auto launch = thread_launch(environment, code, report, environment.memory->initial_pin());
        launch.entry.function = code.processor_loop;
        auto created = REQUIRE_CREATED(runtime->create(launch));
        CHECK(ok(runtime->start(created.handle)));
        REQUIRE_WITHIN(control.entered.load(std::memory_order_acquire) == 1);
        if (custom_boundary) {
            REQUIRE_WITHIN(environment.binding.scheduling_boundaries.load(
                               std::memory_order_relaxed) != 0);
        }
        runtime->request_stop();
        CHECK(ok(runtime->join(std::chrono::steady_clock::now() + 5s)));
        CHECK(created.record->exited() && !runtime->failure());
        if (custom_boundary) {
            CHECK(environment.binding.scheduling_deferrals.load(
                      std::memory_order_relaxed) == 1);
        }
    }
}

void test_closed_worker_reaping(const LoadedCode& code) {
    Environment environment(code);
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    constexpr unsigned iterations = 32;
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        EntryReport report;
        auto created = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, report, environment.memory->initial_pin())));
        CHECK(ok(runtime->start(created.handle)));
        CHECK(created.record->wait_for_exit(5'000'000'000));
        CHECK(environment.handles->close(created.handle));
        std::weak_ptr<horizon::ThreadRecord> retired = created.record;
        auto retained = std::move(created.record);
        std::this_thread::sleep_for(30ms);
        CHECK(!retired.expired());
        retained.reset();
        REQUIRE_WITHIN(retired.expired());
        CHECK(environment.handles->size() == 0 && environment.memory->live_pins.load() == 0);
    }

    EntryReport last_report;
    auto last = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, last_report, environment.memory->initial_pin())));
    CHECK(ok(runtime->start(last.handle)));
    CHECK(last.record->wait_for_exit(5'000'000'000));
    std::weak_ptr<horizon::ThreadRecord> last_record = last.record;
    const auto last_handle = last.handle;
    last.record.reset();
    std::this_thread::sleep_for(30ms);
    CHECK(!last_record.expired() && environment.handles->size() == 1);
    CHECK(environment.handles->close(last_handle));
    REQUIRE_WITHIN(last_record.expired());
    CHECK(environment.handles->size() == 0 && environment.memory->live_pins.load() == 0);

    runtime->request_stop();
    CHECK(ok(runtime->join(std::nullopt)));
    CHECK(environment.binding.owner_destroyed.load() == iterations + 1);
}

void test_parallel_context_isolation(const LoadedCode& code) {
    Environment environment(code);
    environment.binding.barrier_target = 2;
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport first_report, second_report;
    auto first_pin = environment.memory->initial_pin();
    auto second_pin = environment.memory->additional_pin();
    auto first_launch = thread_launch(environment, code, first_report, first_pin);
    auto second_launch = thread_launch(environment, code, second_report, second_pin);
    auto first = REQUIRE_CREATED(runtime->create(first_launch));
    auto second = REQUIRE_CREATED(runtime->create(second_launch));
    CHECK(ok(runtime->start(first.handle)));
    CHECK(ok(runtime->start(second.handle)));
    REQUIRE_WITHIN(environment.binding.first_calls.load() == 2);
    CHECK(ok(runtime->join(std::nullopt)));
    CHECK(first_report.thread_pointer == 0x1000 + first.record->id() * 2);
    CHECK(second_report.thread_pointer == 0x1000 + second.record->id() * 2);
    CHECK(first_report.read_only_thread_pointer == first_report.thread_pointer + 1);
    CHECK(second_report.read_only_thread_pointer == second_report.thread_pointer + 1);
    CHECK(first_report.thread_pointer != second_report.thread_pointer);
    CHECK(first_report.first_result == 0x31003100 &&
          second_report.first_result == 0x31003100);
    CHECK(first.record->exited() && second.record->exited());
}

void test_native_shared_arbitration(const LoadedCode& code) {
    for (const bool stop_after_grant : {false, true}) {
        Environment environment(code);
        environment.binding.use_arbitration = true;
        environment.binding.stop_after_grant = stop_after_grant;
        auto runtime = make_runtime(environment);
        if (!runtime) return;
        EntryReport waiter_report, signaler_report;
        auto waiter = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, waiter_report, environment.memory->initial_pin())));
        auto signaler = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, signaler_report, environment.memory->additional_pin())));
        CHECK(ok(runtime->start(waiter.handle)));
        CHECK(ok(runtime->start(signaler.handle)));
        CHECK(ok(runtime->join(std::nullopt)));
        CHECK(environment.binding.arbitration_wait_checks.load() == 1);
        CHECK(environment.binding.arbitration_wait_successes.load() == 1);
        CHECK(environment.binding.arbitration_wait_cancellations.load() == 0);
        CHECK(environment.binding.arbitration_signal_successes.load() == 1);
        CHECK(environment.binding.arbitration_signal_waiting.load() == 1);
        CHECK(environment.binding.arbitration.waiters(
                  BindingState::arbitration_address) == 0);
        CHECK((waiter_report.first_result == 0xa110ca7e &&
               signaler_report.first_result == 0x5190a1ed) ||
              (waiter_report.first_result == 0x5190a1ed &&
               signaler_report.first_result == 0xa110ca7e));
        CHECK(waiter.record->exited() && signaler.record->exited());
    }
    {
        Environment environment(code);
        environment.binding.use_arbitration = true;
        auto runtime = make_runtime(environment);
        if (!runtime) return;
        EntryReport report;
        auto waiter = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, report, environment.memory->initial_pin())));
        CHECK(ok(runtime->start(waiter.handle)));
        REQUIRE_WITHIN(environment.binding.arbitration.waiters(
                           BindingState::arbitration_address) == 1);
        runtime->request_stop();
        CHECK(ok(runtime->join(std::nullopt)));
        CHECK(environment.binding.arbitration_wait_checks.load() == 1);
        CHECK(environment.binding.arbitration_wait_successes.load() == 0);
        CHECK(environment.binding.arbitration_wait_cancellations.load() == 1);
        CHECK(environment.binding.arbitration.waiters(
                  BindingState::arbitration_address) == 0);
        CHECK(environment.binding.second_calls.load() == 0 && waiter.record->exited());
    }
}

// Contract: core bindings never serialize native guest execution. The spinner
// can complete only after the peer writes its flag, without either invoking SVC.
void test_native_parallel_progress(const LoadedCode& code) {
    for (const std::int32_t peer_core : {1, 0}) {
        Environment environment(code);
        environment.binding.block_first = true;
        auto runtime = make_runtime(environment);
        if (!runtime) return;
        SegmentControl spinner_control, peer_control;
        peer_control.signal_address = reinterpret_cast<std::uintptr_t>(&spinner_control.release);
        EntryReport spinner_report, peer_report;
        spinner_report.segment_control = reinterpret_cast<std::uintptr_t>(&spinner_control);
        peer_report.segment_control = reinterpret_cast<std::uintptr_t>(&peer_control);
        auto spinner_launch = thread_launch(
            environment, code, spinner_report, environment.memory->initial_pin());
        spinner_launch.ideal_core = 0;
        spinner_launch.affinity_mask = 1;
        auto peer_launch = thread_launch(
            environment, code, peer_report, environment.memory->additional_pin());
        peer_launch.ideal_core = peer_core;
        peer_launch.affinity_mask = 1ull << peer_core;
        auto spinner = REQUIRE_CREATED(runtime->create(spinner_launch));
        auto peer = REQUIRE_CREATED(runtime->create(peer_launch));
        CHECK(ok(runtime->start(spinner.handle)));
        REQUIRE_WITHIN(spinner_control.entered.load(std::memory_order_acquire) == 1);
        CHECK(ok(runtime->start(peer.handle)));
        REQUIRE_WITHIN(peer_control.entered.load(std::memory_order_acquire) == 1);
        CHECK(code.thread_slot->active_bindings() == 2);
        CHECK(spinner.record->current_processor() == 0);
        CHECK(peer.record->current_processor() == peer_core);
        CHECK(environment.binding.first_calls.load() == 0);
        CHECK(spinner_control.release.load(std::memory_order_acquire) == 0);
        // Foreign rebinding remains coherent while both guests are in native code,
        // including when both become assigned to the same virtual processor.
        CHECK(ok(runtime->set_core_mask(spinner.handle, 1, 2)));
        CHECK(spinner.record->current_processor() == 1);
        CHECK(ok(runtime->set_core_mask(spinner.handle, 0, 1)));
        CHECK(spinner.record->current_processor() == 0);
        // Only the peer guest can release the spinner. The host releases the peer.
        peer_control.release.store(1, std::memory_order_release);
        REQUIRE_WITHIN(environment.binding.first_calls.load() == 2);
        CHECK(spinner_control.release.load(std::memory_order_acquire) == 1);
        {
            std::lock_guard lock(environment.binding.mutex);
            environment.binding.release = true;
        }
        environment.binding.changed.notify_all();
        CHECK(ok(runtime->join(std::nullopt)));
        CHECK(spinner_report.second_result == 0x32003200 && peer_report.second_result == 0x32003200);
        CHECK(spinner_report.thread_pointer != peer_report.thread_pointer);
        CHECK(spinner.record->exited() && peer.record->exited());
        CHECK(!runtime->failure());
    }
}

void test_cancel_during_binding(const LoadedCode& code) {
    Environment environment(code);
    environment.binding.block_bind = true;
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    SegmentControl control;
    EntryReport report;
    report.segment_control = reinterpret_cast<std::uintptr_t>(&control);
    auto created = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, report, environment.memory->initial_pin())));
    CHECK(ok(runtime->start(created.handle)));
    REQUIRE_WITHIN(environment.binding.blocked_binders.load(std::memory_order_acquire) == 1);
    runtime->request_stop();
    CHECK(ok(runtime->join(std::chrono::steady_clock::now() + 5s)));
    CHECK(created.record->exited() && control.entered.load(std::memory_order_acquire) == 0);
    CHECK(!runtime->failure() && environment.memory->live_pins.load() == 0);
    CHECK(environment.binding.owner_destroyed.load() == 1);
}

void test_stop_retains_native_workers(const LoadedCode& code) {
    Environment environment(code);
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    SegmentControl control;
    EntryReport report;
    report.segment_control = reinterpret_cast<std::uintptr_t>(&control);
    auto created = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, report, environment.memory->initial_pin())));
    CHECK(ok(runtime->start(created.handle)));
    REQUIRE_WITHIN(control.entered.load(std::memory_order_acquire) == 1);
    runtime->request_stop();
    CHECK(failed(runtime->join(std::chrono::steady_clock::now() + 1ms),
                 LaunchError::deadline_expired));
    CHECK(!created.record->exited() && environment.memory->live_pins.load() == 1);
    CHECK(environment.binding.owner_destroyed.load() == 0);
    control.release.store(1, std::memory_order_release);
    CHECK(ok(runtime->join(std::nullopt)));
    CHECK(created.record->exited() && !runtime->failure());
    CHECK(environment.memory->live_pins.load() == 0);
    CHECK(environment.binding.owner_destroyed.load() == 1);
}

void test_timed_join_restores_reaping(const LoadedCode& code) {
    Environment environment(code);
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    SegmentControl control;
    EntryReport report;
    report.segment_control = reinterpret_cast<std::uintptr_t>(&control);
    auto created = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, report, environment.memory->initial_pin())));
    CHECK(ok(runtime->start(created.handle)));
    REQUIRE_WITHIN(control.entered.load(std::memory_order_acquire) == 1);
    CHECK(failed(runtime->join(std::chrono::steady_clock::now() + 1ms),
                 LaunchError::deadline_expired));
    control.release.store(1, std::memory_order_release);
    CHECK(created.record->wait_for_exit(5'000'000'000));
    std::weak_ptr<horizon::ThreadRecord> retired = created.record;
    CHECK(environment.handles->close(created.handle));
    created.record.reset();
    REQUIRE_WITHIN(retired.expired());
    CHECK(environment.binding.owner_destroyed.load() == 1 &&
          environment.memory->live_pins.load() == 0);
    runtime->request_stop();
    CHECK(ok(runtime->join(std::nullopt)));
}

void test_native_startup_sleep_cancellation(const LoadedCode& code) {
    Environment environment(code);
    environment.binding.use_startup_sleep = true;
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport report;
    auto created = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, report, environment.memory->initial_pin())));
    CHECK(ok(runtime->start(created.handle)));
    REQUIRE_WITHIN(environment.binding.startup_sleep_calls.load() == 1);
    runtime->request_stop();
    CHECK(ok(runtime->join(std::nullopt)));
    const auto snapshot = std::get<ThreadSnapshot>(runtime->snapshot(created.handle));
    CHECK(snapshot.state == ThreadState::exited && !snapshot.failure);
    CHECK(created.record->exited() && report.first_result == 0);
    CHECK(environment.binding.owner_destroyed.load() == 1);
}

void test_close_stop_and_deadline(const LoadedCode& code) {
    Environment environment(code);
    environment.binding.block_first = true;
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport report;
    auto created = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, report, environment.memory->initial_pin())));
    CHECK(ok(runtime->start(created.handle)));
    REQUIRE_WITHIN(environment.binding.first_calls.load() == 1);
    CHECK(environment.handles->close(created.handle));
    CHECK(!environment.handles->lookup(created.handle) && !created.record->exited());
    const auto deadline = std::chrono::steady_clock::now() + 5ms;
    CHECK(failed(runtime->join(deadline), LaunchError::deadline_expired));
    CHECK(!created.record->exited());
    runtime->request_stop();
    CHECK(ok(runtime->join(std::nullopt)));
    CHECK(created.record->exited() && environment.binding.second_calls.load() == 0);
    CHECK(environment.binding.owner_destroyed.load() == 1 &&
          environment.memory->live_pins.load() == 0);
}

void test_stop_before_entry(const LoadedCode& code) {
    Environment environment(code);
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport report;
    auto created = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, report, environment.memory->initial_pin())));
    runtime->request_stop();
    CHECK(failed(runtime->start(created.handle), LaunchError::cancelled));
    CHECK(ok(runtime->join(std::nullopt)));
    CHECK(created.record->exited() && environment.binding.bind_calls.load() == 0);
    CHECK(report.stack_pointer == 0 && environment.memory->live_pins.load() == 0);
}

void test_binder_failures(const LoadedCode& code) {
    for (const bool throws : {false, true}) {
        Environment environment(code);
        environment.binding.fail_bind = !throws;
        environment.binding.throw_bind = throws;
        auto runtime = make_runtime(environment);
        if (!runtime) continue;
        EntryReport report;
        auto created = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, report, environment.memory->initial_pin())));
        CHECK(ok(runtime->start(created.handle)));
        CHECK(ok(runtime->join(std::nullopt)));
        const auto snapshot = std::get<ThreadSnapshot>(runtime->snapshot(created.handle));
        CHECK(snapshot.state == ThreadState::exited && snapshot.failure.has_value());
        CHECK(runtime->failure().has_value() && report.stack_pointer == 0);
        CHECK(created.record->exited() && environment.memory->live_pins.load() == 0);
    }
}

void test_exit_current_scope(const LoadedCode& code) {
    Environment environment(code);
    environment.binding.use_exit_current = true;
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    CHECK(!ok(runtime->exit_current()));
    EntryReport report;
    auto created = REQUIRE_CREATED(runtime->create(thread_launch(
        environment, code, report, environment.memory->initial_pin())));
    CHECK(ok(runtime->start(created.handle)) && ok(runtime->join(std::nullopt)));
    CHECK(environment.binding.exit_current_successes.load() == 1);
    CHECK(environment.binding.second_calls.load() == 0 && created.record->exited());
}

void test_join_reentry_from_worker_and_stop(const LoadedCode& code) {
    {
        Environment environment(code);
        environment.binding.join_in_binder = true;
        auto runtime = make_runtime(environment);
        if (!runtime) return;
        EntryReport report;
        auto created = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, report, environment.memory->initial_pin())));
        CHECK(ok(runtime->start(created.handle)) && ok(runtime->join(std::nullopt)));
        CHECK(environment.binding.worker_join_rejections.load() == 1);
        CHECK(created.record->exited());
    }
    {
        Environment environment(code);
        environment.binding.block_first = true;
        environment.binding.join_from_stop_callback = true;
        auto runtime = make_runtime(environment);
        if (!runtime) return;
        EntryReport report;
        auto created = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, report, environment.memory->initial_pin())));
        CHECK(ok(runtime->start(created.handle)));
        REQUIRE_WITHIN(environment.binding.first_calls.load() == 1);
        runtime->request_stop();
        // The join from the stop callback is rejected exactly once, once that
        // callback has completed (bounded wait, never a fixed sleep).
        REQUIRE_WITHIN([&] {
            std::lock_guard lock(environment.binding.mutex);
            return environment.binding.stop_join_ran;
        }());
        CHECK(environment.binding.stop_join_rejections.load() == 1);
        CHECK(ok(runtime->join(std::nullopt)) && created.record->exited());
        CHECK(environment.binding.owner_destroyed.load() == 1);
    }
}

struct DestructorJoinCapture {
    explicit DestructorJoinCapture(BindingState& state) : state(state) {}
    ~DestructorJoinCapture() {
        if (state.runtime && invalid(state.runtime->join(std::nullopt))) {
            state.cleanup_join_rejections.fetch_add(1, std::memory_order_relaxed);
        }
    }
    BindingState& state;
};

struct BlockingCleanupState {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool release = false;
};

struct BlockingCleanupCapture {
    explicit BlockingCleanupCapture(std::shared_ptr<BlockingCleanupState> state)
        : state(std::move(state)) {}
    ~BlockingCleanupCapture() {
        std::unique_lock lock(state->mutex);
        state->entered = true;
        state->changed.notify_all();
        state->changed.wait(lock, [&] { return state->release; });
    }
    std::shared_ptr<BlockingCleanupState> state;
};

void test_join_dependency_cleanup(const LoadedCode& code) {
    {
        Environment environment(code);
        auto capture = std::make_shared<DestructorJoinCapture>(environment.binding);
        auto dependencies = environment.dependencies();
        dependencies.bind_services = [&environment, capture](
            std::shared_ptr<horizon::ThreadRecord> record,
            GuestThreadContext& context, std::stop_token stop) {
            return bind_thread(environment.binding, std::move(record), context, stop);
        };
        capture.reset();
        auto made = create_thread_runtime(std::move(dependencies));
        CHECK(std::holds_alternative<std::shared_ptr<ThreadRuntime>>(made));
        if (!std::holds_alternative<std::shared_ptr<ThreadRuntime>>(made)) return;
        auto runtime = std::get<std::shared_ptr<ThreadRuntime>>(std::move(made));
        environment.binding.runtime = runtime.get();
        EntryReport report;
        auto created = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, report, environment.memory->initial_pin())));
        CHECK(ok(runtime->start(created.handle)) && ok(runtime->join(std::nullopt)));
        CHECK(environment.binding.cleanup_join_rejections.load() == 1);
    }
    {
        Environment environment(code);
        auto cleanup = std::make_shared<BlockingCleanupState>();
        auto capture = std::make_shared<BlockingCleanupCapture>(cleanup);
        auto dependencies = environment.dependencies();
        dependencies.bind_services = [&environment, capture](
            std::shared_ptr<horizon::ThreadRecord> record,
            GuestThreadContext& context, std::stop_token stop) {
            return bind_thread(environment.binding, std::move(record), context, stop);
        };
        capture.reset();
        auto made = create_thread_runtime(std::move(dependencies));
        CHECK(std::holds_alternative<std::shared_ptr<ThreadRuntime>>(made));
        if (!std::holds_alternative<std::shared_ptr<ThreadRuntime>>(made)) return;
        auto runtime = std::get<std::shared_ptr<ThreadRuntime>>(std::move(made));
        environment.binding.runtime = runtime.get();
        EntryReport report;
        auto created = REQUIRE_CREATED(runtime->create(thread_launch(
            environment, code, report, environment.memory->initial_pin())));
        CHECK(ok(runtime->start(created.handle)));
        std::atomic<bool> first_done{false}, second_done{false};
        std::atomic<bool> first_ok{false}, second_ok{false};
        std::thread first([&] {
            first_ok = ok(runtime->join(std::nullopt));
            first_done = true;
        });
        {
            std::unique_lock lock(cleanup->mutex);
            if (!cleanup->changed.wait_for(lock, 5s, [&] { return cleanup->entered; })) {
                bail(__LINE__, "dependency cleanup did not begin");
            }
        }
        std::thread second([&] {
            second_ok = ok(runtime->join(std::nullopt));
            second_done = true;
        });
        std::this_thread::sleep_for(20ms);
        CHECK(!first_done.load() && !second_done.load());
        {
            std::lock_guard lock(cleanup->mutex);
            cleanup->release = true;
        }
        cleanup->changed.notify_all();
        first.join();
        second.join();
        CHECK(first_ok.load() && second_ok.load());
    }
}

void test_authoritative_repin_failures(const LoadedCode& code) {
    {
        Environment environment(code);
        auto runtime = make_runtime(environment);
        EntryReport report;
        auto pin = environment.memory->initial_pin();
        environment.memory->refuse_pin = true;
        CHECK(failed(runtime->create(thread_launch(environment, code, report, pin))));
        CHECK(environment.handles->size() == 0);
    }
    {
        Environment environment(code);
        environment.memory->native_bias = 16;
        auto runtime = make_runtime(environment);
        EntryReport report;
        auto pin = environment.memory->initial_pin();
        CHECK(failed(runtime->create(thread_launch(environment, code, report, pin))));
        CHECK(environment.handles->size() == 0);
    }
    {
        Environment environment(code);
        environment.memory->read_only_pin = true;
        auto runtime = make_runtime(environment);
        EntryReport report;
        auto pin = environment.memory->initial_pin();
        CHECK(failed(runtime->create(thread_launch(environment, code, report, pin))));
        CHECK(environment.handles->size() == 0);
    }
}

} // namespace

void test_guest_fault_report(const LoadedCode& code) {
    // The entry's first load reads [x0 + 40]: x0 = 0x10 faults on the zero page.
    install_guest_fault_handler();
    Environment environment(code);
    auto runtime = make_runtime(environment);
    if (!runtime) return;
    EntryReport report;
    auto created = [&] {
        auto launch = thread_launch(environment, code, report, environment.memory->initial_pin());
        launch.entry.x0 = 0x10;
        return REQUIRE_CREATED(runtime->create(launch));
    }();
    CHECK(ok(runtime->start(created.handle)));
    CHECK(ok(runtime->join(std::nullopt)));
    const auto snapshot = std::get<ThreadSnapshot>(runtime->snapshot(created.handle));
    CHECK(snapshot.state == ThreadState::exited && snapshot.failure.has_value());
    const auto failure = runtime->failure();
    CHECK(failure.has_value());
    if (!failure) return;
    const auto load = code.modules->mapping(0).base_address() + code_offset(armigo_thread_runtime_entry) + 4;
    char location[64];
    std::snprintf(location, sizeof location, "SIGSEGV at thread-runtime+0x%llx",
                  static_cast<unsigned long long>(code_offset(armigo_thread_runtime_entry) + 4));
    CHECK(failure->code == LaunchError::guest_fault && failure->stage == LaunchStage::threads);
    CHECK(failure->guest_pc == load);
    CHECK(failure->thread_id == created.record->id());
    CHECK(failure->detail.find(location) != std::string::npos);
    CHECK(failure->detail.find("address 0x38") != std::string::npos);
    CHECK(failure->detail.find("No supervisor call yet.") != std::string::npos);
    CHECK(failure->detail.find(" x19=0x10 ") != std::string::npos);
    CHECK(report.stack_pointer == 0);
    CHECK(created.record->exited() && environment.memory->live_pins.load() == 0);
}

int main() {
    const auto code = load_code();
    if (!code) {
        std::fprintf(stderr, "Failed to build the sealed synthetic thread module.\n");
        return 1;
    }
    test_factory_validation(*code);
    test_create_validation(*code);
    test_original_process_creation(*code);
    test_original_root_reuse(*code);
    test_native_execution_and_queries(*code);
    test_native_scheduling_calls(*code);
    test_scheduling_stop_boundaries(*code);
    test_closed_worker_reaping(*code);
    test_parallel_context_isolation(*code);
    test_native_shared_arbitration(*code);
    test_native_parallel_progress(*code);
    test_cancel_during_binding(*code);
    test_stop_retains_native_workers(*code);
    test_timed_join_restores_reaping(*code);
    test_native_startup_sleep_cancellation(*code);
    test_close_stop_and_deadline(*code);
    test_stop_before_entry(*code);
    test_binder_failures(*code);
    test_exit_current_scope(*code);
    test_join_reentry_from_worker_and_stop(*code);
    test_join_dependency_cleanup(*code);
    test_authoritative_repin_failures(*code);
    test_guest_fault_report(*code);
    if (failures == 0) {
        std::puts("Native thread runtime lifecycle and validation tests passed.");
    }
    return failures == 0 ? 0 : 1;
}
