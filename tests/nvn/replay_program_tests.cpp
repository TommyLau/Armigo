// ============================================================================
// Armigo NVN - Driver Backends
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "programs.h"
#include "entry_table.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

using namespace armigo::nvn;
namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #value); ++failures; } } while (false)
template<Entry E> auto api_function() {
    return reinterpret_cast<typename EntrySignature<E>::type>(
        armigo_nvn_bootstrap_loader(entry_name(E).data()));
}
struct ResolverState { unsigned calls = 0; bool fail = false; } resolver_state;
bool resolve(void* context, const DeviceState*, std::span<const NVNshaderData> input,
    std::array<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>, 6>& result,
    std::string& error) {
    auto& state = *static_cast<ResolverState*>(context);
    ++state.calls;
    if (state.fail) { error = "Synthetic package miss"; return false; }
    for (std::size_t i = 0; i < input.size(); ++i) {
        auto stage = std::make_shared<armigo::platform::CompiledGraphicsStage>();
        stage->entry_point = "synthetic";
        result[i] = stage;
    }
    return true;
}
}
int main() {
    select_api_version({55, 15});
    register_program_entries();
    const auto initialize = api_function<Entry::nvnProgramInitialize>();
    const auto finalize = api_function<Entry::nvnProgramFinalize>();
    const auto set = api_function<Entry::nvnProgramSetShaders>();
    const auto extended = api_function<Entry::nvnProgramSetShadersExt>();
    NVNprogram program{};
    CHECK(initialize(&program, nullptr));
    NVNshaderData descriptors[2]{{123, nullptr}, {456, nullptr}};
    CHECK(!set(&program, 1, descriptors));
    set_program_resolver(resolve, &resolver_state,
        [](void*, const NVNshaderDataExt& descriptor, std::string&) {
            return descriptor.extra == descriptor.control;
        });
    CHECK(set(&program, 2, descriptors));
    auto& live = *reinterpret_cast<ProgramState*>(&program);
    auto first = retain_program(live);
    CHECK(first && first->stages[0] && first->stages[1]);
    CHECK(first && first->state.snapshot == &first->state);
    CHECK(resolver_state.calls == 1);
    const auto first_serial = live.serial;
    CHECK(retain_program(live) == first && resolver_state.calls == 1);
    resolver_state.fail = true;
    CHECK(!set(&program, 1, descriptors));
    CHECK(live.serial == first_serial && retain_program(live) == first);
    resolver_state.fail = false;
    CHECK(!set(&program, -1, descriptors));
    CHECK(!set(&program, 7, descriptors));
    CHECK(!set(&program, 1, nullptr));
    NVNshaderDataExt extended_descriptor{789, nullptr, nullptr};
    CHECK(extended(&program, 1, &extended_descriptor));
    CHECK(live.serial != first_serial && first->stages[1]);
    int metadata = 0, override = 1;
    extended_descriptor.control = &metadata;
    extended_descriptor.extra = &override;
    const auto prior = live.serial;
    CHECK(!extended(&program, 1, &extended_descriptor));
    CHECK(live.serial == prior);
    extended_descriptor.extra = &metadata;
    CHECK(extended(&program, 1, &extended_descriptor));
    auto second = retain_program(live);
    const auto snapshot = live;
    finalize(&program);
    CHECK(live.tag == 0);
    CHECK(retain_program(snapshot) == second);
    CHECK(retain_program(first->state) == first);
    second.reset();
    CHECK(!retain_program(snapshot));
    CHECK(initialize(&program, nullptr));
    CHECK(set(&program, 0, nullptr));
    CHECK(retain_program(live) && !retain_program(live)->stages[0]);
    finalize(&program);
    // Program constants: the range starts at the control's +0x700 offset
    // into the shader data and has the control's +0x6fc byte size, whatever
    // the code size at +0x704 says; a zero size records no constants.
    {
        CHECK(initialize(&program, nullptr));
        alignas(16) std::uint8_t control[0x800]{};
        const auto put = [&](std::size_t offset, std::uint32_t value) { std::memcpy(control + offset, &value, 4); };
        put(0x6fc, 0x40);
        put(0x700, 0x300);
        put(0x704, 0x1000);
        put(0x714, 1);
        NVNshaderData fragment{0x10000, control};
        CHECK(set(&program, 1, &fragment));
        const auto program_state = retain_program(live);
        CHECK(program_state && program_state->constants[1].address == 0x10300 &&
              program_state->constants[1].size == 0x40);
        put(0x6fc, 0);
        CHECK(set(&program, 1, &fragment));
        const auto empty = retain_program(live);
        CHECK(empty && empty->constants[1].size == 0);
        finalize(&program);
    }
    // Resolution runs outside the program lock: while one thread's resolver
    // is still loading a program, another thread retains an existing program
    // and sets shaders on a second one without waiting for it.
    {
        CHECK(initialize(&program, nullptr));
        CHECK(set(&program, 1, descriptors));
        const auto existing = retain_program(live);
        struct Blocking {
            std::mutex mutex;
            std::condition_variable changed;
            bool entered = false, released = false;
        } blocking;
        set_program_resolver([](void* context, const DeviceState*, std::span<const NVNshaderData> input,
                                std::array<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>, 6>& result,
                                std::string&) {
            auto& self = *static_cast<Blocking*>(context);
            std::unique_lock lock(self.mutex);
            if (input.size() == 2) {
                self.entered = true;
                self.changed.notify_all();
                self.changed.wait_for(lock, std::chrono::seconds(5), [&] { return self.released; });
            }
            for (std::size_t i = 0; i < input.size(); ++i)
                result[i] = std::make_shared<armigo::platform::CompiledGraphicsStage>();
            return true;
        }, &blocking);
        NVNprogram slow{}, quick{};
        CHECK(initialize(&slow, nullptr));
        CHECK(initialize(&quick, nullptr));
        bool slow_set = false;
        std::thread loader([&] { slow_set = set(&slow, 2, descriptors); });
        {
            std::unique_lock lock(blocking.mutex);
            CHECK(blocking.changed.wait_for(lock, std::chrono::seconds(5), [&] { return blocking.entered; }));
        }
        const auto start = std::chrono::steady_clock::now();
        CHECK(retain_program(live) == existing);
        CHECK(set(&quick, 1, descriptors));
        CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
        // The slow program is still being resolved: nothing is published yet.
        CHECK(reinterpret_cast<ProgramState*>(&slow)->serial == 0);
        {
            std::lock_guard lock(blocking.mutex);
            blocking.released = true;
        }
        blocking.changed.notify_all();
        loader.join();
        CHECK(slow_set);
        const auto& slow_state = *reinterpret_cast<ProgramState*>(&slow);
        const auto resolved = retain_program(slow_state);
        CHECK(resolved && resolved->stages[0] && resolved->stages[1]);
        CHECK(retain_program(*reinterpret_cast<ProgramState*>(&quick)));
        finalize(&slow);
        finalize(&quick);
        finalize(&program);
        set_program_resolver(nullptr, nullptr);
    }
    // Deferred stages: SetShaders publishes the program while a stage is
    // still pending; recording retains it without waiting, and the first
    // retain_program waits for the stage and fills it in for every holder.
    // A pending stage that fails stays empty.
    {
        using StagePointer = std::shared_ptr<const armigo::platform::CompiledGraphicsStage>;
        struct Gate final : PendingStage {
            std::mutex mutex;
            std::condition_variable changed;
            bool released = false;
            unsigned waits = 0;
            StagePointer stage;
            StagePointer wait() override {
                std::unique_lock lock(mutex);
                ++waits;
                changed.notify_all();
                changed.wait(lock, [&] { return released; });
                return stage;
            }
            void release() {
                { std::lock_guard lock(mutex); released = true; }
                changed.notify_all();
            }
        };
        struct Deferred {
            std::shared_ptr<Gate> gate;
            bool both = false;
        } deferred;
        const auto ready = std::make_shared<armigo::platform::CompiledGraphicsStage>();
        set_deferred_program_resolver([](void* context, const DeviceState*, std::span<const NVNshaderData> input,
                                         std::array<StagePointer, 6>& result,
                                         std::array<std::shared_ptr<PendingStage>, 6>& pending, std::string&) {
            auto& self = *static_cast<Deferred*>(context);
            result[0] = std::make_shared<armigo::platform::CompiledGraphicsStage>();
            if (input.size() == 2) pending[1] = self.gate;
            if (self.both) pending[0] = self.gate;
            return true;
        }, &deferred);
        deferred.gate = std::make_shared<Gate>();
        deferred.gate->stage = ready;
        CHECK(initialize(&program, nullptr));
        CHECK(set(&program, 2, descriptors));
        CHECK(deferred.gate->waits == 0);
        const auto recorded = retain_program_snapshot(live);
        CHECK(recorded && recorded->pending && deferred.gate->waits == 0);
        // The look-ahead's retain neither waits nor returns the program yet.
        bool translating = false;
        CHECK(!retain_ready_program(live, translating) && translating && deferred.gate->waits == 0);
        StagePointer seen[2];
        std::thread replay_a([&] { if (auto p = retain_program(live)) seen[0] = p->stages[1]; });
        std::thread replay_b([&] { if (auto p = retain_program(live)) seen[1] = p->stages[1]; });
        {
            std::unique_lock lock(deferred.gate->mutex);
            CHECK(deferred.gate->changed.wait_for(lock, std::chrono::seconds(5),
                                                  [&] { return deferred.gate->waits == 1; }));
        }
        deferred.gate->release();
        replay_a.join();
        replay_b.join();
        CHECK(seen[0] == ready && seen[1] == ready);
        CHECK(deferred.gate->waits == 1);
        CHECK(recorded->stages[0] && recorded->stages[1] == ready && recorded->failed_stages == 0);
        CHECK(retain_program(live) == recorded);
        CHECK(retain_ready_program(live, translating) == recorded && !translating);
        // A failed pending stage leaves the program without that stage.
        deferred.gate = std::make_shared<Gate>();
        CHECK(set(&program, 2, descriptors));
        deferred.gate->release();
        const auto failed = retain_program(live);
        CHECK(failed && failed->stages[0] && !failed->stages[1] && failed->failed_stages == 2u);
        // A stage both prepared and pending is refused.
        deferred.both = true;
        const auto before = live.serial;
        CHECK(!set(&program, 1, descriptors));
        CHECK(live.serial == before);
        finalize(&program);
        set_program_resolver(nullptr, nullptr);
    }
    set_program_resolver(nullptr, nullptr);
    return failures ? 1 : 0;
}
