// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include <armigo/common/diagnostics.h>
// The renderer queue records typed commands and replays them in submission order.

#include "engine_renderer.h"
#include "engine_pipeline.h"
#include "native_pipeline_state.h"
#include "device_context.h"
#include "buffer_cache.h"
#include "texture_cache.h"
#include "twod_blit.h"
#include <armigo/platform/metal_color_clear.h>
#include "compute_dispatch.h"
#include <armigo/graphics/color_clear.h>
#include <armigo/graphics/resource_memory_registry.h>
#include <armigo/memory/write_tracker.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <list>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <utility>

// Guest effects follow GPU completion; Armigo serializes their write-back and
// publication with its registry.
// Deferred write-back batches encode draws into one lazily committed frame
// command buffer per batch, committed and awaited where results become
// observable (coherence/readback boundaries), instead of committing and
// waiting once per draw.
namespace armigo::platform::metal {
namespace {
using namespace graphics;

RendererError failure(const char* detail, CommandStatus status = CommandStatus::Unsupported) {
    return RendererError({status, detail});
}
template<class T> T take(RendererResult<T> value) {
    if (const auto* error = std::get_if<RendererError>(&value)) throw *error;
    return std::get<T>(std::move(value));
}
void check(RendererStatus status) { static_cast<void>(take(std::move(status))); }
std::uint64_t deadline(std::uint64_t timeout) {
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return std::uint64_t(now) + timeout;
}
bool same_owner(const std::shared_ptr<void>& a, const std::shared_ptr<void>& b) {
    return !a.owner_before(b) && !b.owner_before(a);
}
bool same_mapping(const Mapping& a, const Mapping& b) {
    return a.gpu_base == b.gpu_base && a.mapped_size == b.mapped_size &&
        a.buffer_offset == b.buffer_offset && a.guest_address == b.guest_address &&
        a.backing_size == b.backing_size && a.page_size == b.page_size &&
        same_owner(a.owner, b.owner);
}
struct AllocationPin final : ResourceLease {
    explicit AllocationPin(std::shared_ptr<void> value) : owner(std::move(value)) {}
    std::shared_ptr<void> owner;
};

// Only the serialized callback installs active. No borrowed CPU callback is
// copied into retained work. Backing pins and bounds are independent of GPU VA;
// unmap writeback must never translate through a replacement mapping.
class ScopedMemory final : public ResourceMemoryBackend {
public:
    struct Entry { Mapping mapping; BackingId backing; MappingId id; };
    struct Allocation { Mapping mapping; BackingId backing; };
    std::vector<Entry> entries;
    std::vector<Allocation> allocations;
    MemoryCallbacks active{};
    std::uint64_t next_backing = 0, next_mapping = 0;
    // Generation of the address-space view the entries last matched; zero
    // after any driver notification changes them.
    std::uint64_t matched_view = 0;
    // Other recent views known to hold exactly the registered mappings, so
    // callers alternating between views skip the comparison.
    std::array<std::uint64_t, 8> matched_views{};
    void forget_views() noexcept { matched_view = 0; matched_views = {}; }
    bool matches(std::uint64_t generation) const noexcept {
        return generation && (generation == matched_view ||
            std::find(matched_views.begin(), matched_views.end(), generation) != matched_views.end());
    }
    void remember(std::uint64_t generation) noexcept {
        if (!generation) return;
        std::rotate(matched_views.rbegin(), matched_views.rbegin() + 1, matched_views.rend());
        matched_views[0] = matched_view;
        matched_view = generation;
    }
    std::uint64_t timeout = 0;
    std::optional<SubmissionId> held_prefix;

    RendererStatus read_cpu(BackingRange range, std::span<std::byte> out) override {
        return transfer(range, out, {}, false);
    }
    RendererStatus write_cpu(BackingRange range, std::span<const std::byte> in) override {
        return transfer(range, {}, in, true);
    }
    RendererStatus wait_gpu(CompletionToken token) override {
        const auto end = deadline(timeout);
        for (;;) {
            const auto state = token.snapshot();
            if (state.error) return *state.error;
            if (state.phase == SubmissionPhase::GpuComplete ||
                state.phase == SubmissionPhase::Publishing ||
                state.phase == SubmissionPhase::Complete) return renderer_success();
            if (held_prefix && state.id.queue == held_prefix->queue &&
                state.id.sequence > held_prefix->sequence) return commit_hold_wait();
            if (state.phase == SubmissionPhase::Accepted)
                return failure("Engine renderer producer has not been submitted", CommandStatus::Stopped);
            if (deadline(0) >= end)
                return failure("Engine renderer producer feedback timed out", CommandStatus::Stopped);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
private:
    RendererStatus transfer(BackingRange range, std::span<std::byte> out,
                            std::span<const std::byte> in, bool write) {
        if (!active.read && !active.write)
            return failure("Engine renderer CPU access is outside its synchronous callback", CommandStatus::Stopped);
        if (range.size != (write ? in.size() : out.size()))
            return failure("Engine renderer backing transfer size mismatch", CommandStatus::MemoryFault);
        const auto allocation = std::find_if(allocations.begin(), allocations.end(),
            [&](const Allocation& value) { return value.backing == range.backing; });
        if (allocation == allocations.end() || !allocation->mapping.owner ||
            range.offset > allocation->mapping.backing_size ||
            range.size > allocation->mapping.backing_size - range.offset)
            return failure("Engine renderer backing transfer is outside its pinned allocation", CommandStatus::MemoryFault);
        const auto address = allocation->mapping.guest_address + range.offset;
        const bool transferred = write
            ? active.write && active.write(active.context, address, in)
            : active.read && active.read(active.context, address, out);
        if (!transferred)
            return failure("Engine renderer backing transfer failed", CommandStatus::MemoryFault);
        return renderer_success();
    }
};

struct Recovery { std::shared_ptr<ResourceMemoryRegistry> failed_memory; };
class Effects final : public CompletionActions {
public:
    Effects(std::shared_ptr<ResourceMemoryRegistry> memory, std::weak_ptr<Recovery> recovery)
        : memory_(std::move(memory)), recovery_(std::move(recovery)) {}
    RendererStatus make_coherent() override { return renderer_success(); }
    RendererStatus prepare_publication() override { return renderer_success(); }
    void publish() noexcept override {}
    RendererStatus preserve_failed_authority() override {
        const auto recovery = recovery_.lock();
        if (!recovery) return failure("Engine renderer recovery owner is unavailable", CommandStatus::Stopped);
        recovery->failed_memory = memory_;
        return renderer_success();
    }
private:
    std::shared_ptr<ResourceMemoryRegistry> memory_;
    std::weak_ptr<Recovery> recovery_;
};

struct Resources {
    std::shared_ptr<ScopedMemory> backend;
    std::shared_ptr<ResourceMemoryRegistry> memory;
    std::shared_ptr<MetalRenderer> renderer;
    std::shared_ptr<SubmissionQueue> submissions;
    // Shared with the renderer services; 2D blits resolve through this cache.
    std::shared_ptr<graphics::TextureCache> textures;
    std::shared_ptr<Recovery> recovery = std::make_shared<Recovery>();
    // Coordinator-owned: queued cancellation must not release an exclusion
    // concurrently with preparation on the backend thread.
    std::shared_ptr<ResourceLease> deferred_store;
    // Emergency teardown retention requires no allocation. No guest callbacks
    // remain active in a quarantined domain, and it can never publish effects.
    std::shared_ptr<Resources> quarantined_next;
};
void quarantine(std::shared_ptr<Resources> resources) noexcept {
    static std::mutex mutex;
    static std::shared_ptr<Resources> retained;
    std::lock_guard lock(mutex);
    resources->quarantined_next = std::move(retained);
    retained = std::move(resources);
}

// Why an open batch was committed; ARMIGO_TRACE_ENGINE_FLUSH tallies them.
enum class FlushReason : unsigned {
    Shutdown, Blit, CoherenceFull, CoherenceRanged, MemoryChanged, Mapping, Budget, Count
};

// ARMIGO_TRACE_ENGINE_FLUSH: a summary line every 1000 batch commits (count
// and CPU time per reason); ARMIGO_TRACE_SLOW_CALLS adds a line for any engine
// call over two seconds.
bool trace_flush_enabled() {
    static const bool enabled = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
    return enabled;
}

void trace_flush(FlushReason reason, std::chrono::steady_clock::duration elapsed) {
    if (!trace_flush_enabled()) return;
    static std::mutex trace_mutex;
    std::lock_guard trace_lock(trace_mutex);
    static std::uint64_t count = 0;
    static std::array<std::uint64_t, unsigned(FlushReason::Count)> counts{}, ns{};
    ++count;
    ++counts[unsigned(reason)];
    ns[unsigned(reason)] += std::uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    if (count % 1000) return;
    static constexpr const char* names[] = {"shutdown", "blit", "coherence_full", "coherence_ranged",
                                            "memory_changed", "mapping", "budget"};
    armigo::diag::opt_in( "engine_flush count=%llu", (unsigned long long)count);
    for (unsigned index = 0; index != unsigned(FlushReason::Count); ++index)
        armigo::diag::opt_in( " %s=%llu/%llums", names[index], (unsigned long long)counts[index],
                     (unsigned long long)(ns[index] / 1000000));
    armigo::diag::opt_in( "\n");
}

struct SlowCall {
    const char* name;
    unsigned kind;
    // Timed only on request (ARMIGO_TRACE_SLOW_CALLS): coherence runs about
    // once per draw.
    static bool enabled() {
        static const bool value = std::getenv("ARMIGO_TRACE_SLOW_CALLS") != nullptr;
        return value;
    }
    std::chrono::steady_clock::time_point started =
        enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~SlowCall() {
        if (!enabled()) return;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (ms >= 2000) armigo::diag::opt_in( "engine_slow_call call=%s kind=%u ms=%lld\n", name, kind,
                                     (long long)ms);
    }
};

class EngineBackend : public std::enable_shared_from_this<EngineBackend> {
public:
    // Owned requests between the front end and this backend. A front end
    // that runs far ahead only grows the backend's working set (each request
    // holds a register snapshot); ARMIGO_ENGINE_QUEUE_DEPTH overrides it.
    static constexpr std::size_t default_queue_capacity = 4096;
    const std::size_t queue_capacity = [] {
        const char* value = std::getenv("ARMIGO_ENGINE_QUEUE_DEPTH");
        const auto depth = value ? std::strtoull(value, nullptr, 10) : 0;
        return depth >= 16 && depth <= default_queue_capacity ? std::size_t(depth) : default_queue_capacity;
    }();
    static constexpr std::size_t write_capacity = InlineUpload::max_upload_bytes;
    MetalEngineOptions options;
    EnginePipelineFactory pipeline_factory;
    std::shared_ptr<SubmissionTestControl> test_control;
    std::shared_ptr<DeviceContext> device;
    std::shared_ptr<PipelineStateCache> pipelines;
    std::shared_ptr<GraphicsStageProvider> stages;
    std::shared_ptr<Resources> resources;
    std::optional<RendererError> stopped;
    CommandResult literal_stop;
    MetalDrawStatistics legacy_draw_statistics;
    MetalClearStatistics legacy_clear_statistics;
    // Compute launches use the shared synchronous Metal 4 compute backend.
    MetalComputeContext compute_context;
    MetalComputeStatistics compute_statistics;
    std::uint64_t domain = 0, sequence = 0;
    std::uint64_t previous_buffers = 0, previous_passes = 0;
    // Receipts of batched work accepted but not yet converted to completion
    // statistics, tagged with the request kind for the completed counters.
    std::vector<std::pair<CompletionToken, RenderKind>> outstanding;
    // Draws and clears encoded into the current unflushed batch.
    std::uint64_t batched = 0;
    // Backend state belongs to the consumer. Producers use queue_mutex and do
    // not wait for registry, cache, pipeline or Metal preparation.
    std::mutex mutex;
    std::mutex queue_mutex;
    // Serializes producers and their retained-view cache, never backend work.
    std::mutex producer_mutex;
    std::vector<std::unique_ptr<RenderStop>> render_pool;
    EngineBackend() {
        render_pool.reserve(queue_capacity);
        for (std::size_t i = 0; i != queue_capacity; ++i)
            free_pending.emplace_back(PendingRender{});
    }
    std::atomic<bool> terminal = false;
    std::atomic<unsigned> active_callbacks = 0;
    std::atomic<bool> lifecycle_released = false;
    std::atomic<bool> shutdown_started = false;
    enum class PendingRenderKind { Draw, Clear, Dispatch, Blit };
    struct PendingRender {
        std::unique_ptr<RenderStop> stop;
        std::shared_ptr<const GpuAddressSpace> memory;
        PendingRenderKind kind;
    };
    struct PendingEffect {
        GuestEffect effect;
        std::shared_ptr<const GpuAddressSpace> memory;
        bool committed = false, coherent = false;
        std::uint64_t expires = 0;
        CompletionToken prefix;
        std::shared_ptr<GuestBoundary> boundary;
    };
    struct PendingWrite {
        std::uint64_t address;
        std::vector<std::byte> bytes;
        std::shared_ptr<const GpuAddressSpace> memory;
    };
    struct BarrierResult {
        std::condition_variable ready;
        bool done = false;
        CommandResult result;
    };
    struct PendingCoherence {
        std::shared_ptr<const GpuAddressSpace> memory;
        std::uint64_t gpu_address;
        std::uint64_t size;
        bool prepare_write;
        std::shared_ptr<BarrierResult> completion;
        bool bypass = false;
    };
    struct PendingResolve {
        std::uint64_t guest_address;
        std::uint64_t size;
        std::shared_ptr<BarrierResult> completion;
    };
    struct PendingMemoryChange {
        std::vector<Mapping> removed;
        std::vector<std::shared_ptr<void>> released;
        MemoryCallbacks callbacks;
        std::shared_ptr<BarrierResult> completion;
    };
    using Pending = std::variant<PendingRender, PendingEffect, PendingWrite, PendingCoherence,
                                 PendingResolve, PendingMemoryChange>;
    // Preallocated list nodes preserve cheap front insertion for disjoint reads
    // and effect retries without allocating on the producer/consumer hot path.
    std::list<Pending> pending, free_pending;
    void push_pending(Pending request, std::list<Pending>::iterator position) noexcept {
        auto node = free_pending.begin();
        *node = std::move(request);
        pending.splice(position, free_pending, node);
    }
    void push_pending(Pending request) noexcept { push_pending(std::move(request), pending.end()); }
    // Includes the active request. Queue ownership, including this pointer and
    // the payload budget, is protected by queue_mutex while work executes.
    const Pending* active_request = nullptr;
    std::size_t write_bytes = 0;
    // Share one immutable view across queued work, but never keep allocation
    // pins alive after the queue and mapping registry have released them.
    std::weak_ptr<const GpuAddressSpace> retained_view;
    std::condition_variable wake;
    bool closing = false;
    bool worker_active = false;
    bool threaded = false;
    struct PumpWake {
        std::mutex mutex;
        std::condition_variable ready;
        bool requested = false, closing = false;
        void notify() {
            std::lock_guard lock(mutex);
            requested = true;
            ready.notify_one();
        }
    };
    std::shared_ptr<PumpWake> pump_wake = std::make_shared<PumpWake>();
    std::thread pump;

    struct CallbackScope {
        EngineBackend& backend;
        std::shared_ptr<EngineBackend> owner;
        explicit CallbackScope(EngineBackend& value)
            : backend(value), owner(value.shared_from_this()) {
            backend.active_callbacks.fetch_add(1, std::memory_order_acq_rel);
        }
        ~CallbackScope() {
            if (backend.active_callbacks.fetch_sub(1, std::memory_order_acq_rel) == 1 &&
                backend.lifecycle_released.load(std::memory_order_acquire))
                backend.shutdown();
        }
    };

    void release_lifecycle() noexcept {
        lifecycle_released.store(true, std::memory_order_release);
        if (active_callbacks.load(std::memory_order_acquire) == 0) shutdown();
    }

    ~EngineBackend() {
        shutdown();
        finish_shutdown_batch();
    }

    void shutdown() noexcept {
        if (!threaded) return;
        bool expected = false;
        if (!shutdown_started.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel, std::memory_order_acquire)) return;
        const bool worker = pump.joinable() && pump.get_id() == std::this_thread::get_id();
        std::list<Pending> cancelled;
        {
            std::unique_lock lock(queue_mutex);
            if (closing) {
                if (!worker && pump.joinable()) {
                    lock.unlock();
                    pump.join();
                }
                return;
            }
            closing = true;
            if (!worker)
                cancelled = cancel_pending(CommandResult{CommandStatus::Stopped,
                                                         "Engine renderer stopped during shutdown"});
            wake.notify_all();
        }
        {
            std::lock_guard lock(pump_wake->mutex);
            pump_wake->closing = true;
            pump_wake->ready.notify_one();
        }
        // Destroy queued publications only after the worker has been told to
        // stop: their captures may release the callback lifecycle recursively.
        cancelled.clear();
        if (pump.joinable()) {
            // Publication may release the final callback owner from this worker.
            // Its thread-held backend owner keeps the current stack valid.
            if (worker) pump.detach();
            else pump.join();
        }
    }

    void finish_shutdown_batch() noexcept {
        // Teardown cancels unpublished effects and unencoded renders. No driver
        // callback is allowed from destruction; committed GPU work retires below.
        pending.clear();
        retained_view.reset();
        if (!resources || !resources->renderer) return;
        // After callback return there is deliberately no CPU capability. A
        // terminal error may retire safely but must never retry guest writeback.
        try {
            resources->backend->held_prefix.reset();
            resources->deferred_store.reset();
            if (resources->renderer->commit_held())
                check(resources->renderer->abort_commit_hold(stopped ? stopped->command() :
                    CommandResult{CommandStatus::Stopped, "Engine renderer cancelled unpublished work"}));
            auto status = resources->renderer->drain(deadline(options.completion_timeout_ns));
            if (std::holds_alternative<RendererError>(status))
                status = resources->renderer->drain(deadline(options.completion_timeout_ns));
            static_cast<void>(retire_completed());
            record_statistics();
            if (std::holds_alternative<RendererError>(status)) quarantine(std::move(resources));
        } catch (...) { quarantine(std::move(resources)); }
    }

    void start_pump(std::shared_ptr<EngineBackend> owner) {
        threaded = true;
        if (resources && resources->renderer) resources->renderer->set_engine_stores_reported(true);
        pump = std::thread([owner = std::move(owner), signal = pump_wake] {
            std::unique_lock signal_lock(signal->mutex);
            bool more = false;
            while (!signal->closing) {
                const auto notified = [&] { return signal->closing || signal->requested; };
                if (more) signal->ready.wait_for(signal_lock, std::chrono::milliseconds(1), notified);
                else signal->ready.wait(signal_lock, notified);
                if (signal->closing) break;
                signal->requested = false;
                signal_lock.unlock();
                @autoreleasepool {
                    std::unique_lock lock(owner->queue_mutex);
                    static_cast<void>(owner->finish_pending(lock, false));
                    more = !owner->pending.empty() || owner->idle_poll_needed();
                }
                signal_lock.lock();
            }
        });
    }

    static CommandResult ordered_effect(void* raw, const GpuAddressSpace& memory,
                                        GuestEffect effect) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        std::unique_lock producer_lock(self.producer_mutex);
        std::unique_lock lock(self.queue_mutex);
        if (self.terminal.load(std::memory_order_acquire)) return self.terminal_result(lock);
        @autoreleasepool {
            try {
                if (const auto available = self.wait_queue_slot(lock); !available.ok()) return available;
                lock.unlock();
                if (effect.size > effect.payload.size())
                    throw failure("Engine guest effect payload is too large");
                if (effect.size && effect.publish)
                    throw failure("Engine guest effect must have one publication operation");
                auto owned = self.retain_memory(memory);
                // Validate the entire store without touching guest bytes. The
                // pinned immutable view keeps this translation valid until publish.
                if (effect.size) {
                    const auto mappings = owned->mappings();
                    const auto found = std::find_if(mappings.begin(), mappings.end(), [&](const auto& mapping) {
                        return effect.address >= mapping.gpu_base &&
                            effect.address - mapping.gpu_base < mapping.mapped_size &&
                            effect.size <= mapping.mapped_size - (effect.address - mapping.gpu_base);
                    });
                    if (found == mappings.end()) throw failure("Engine guest effect store is unmapped",
                                                               CommandStatus::MemoryFault);
                }
                lock.lock();
                if (self.terminal.load(std::memory_order_acquire)) return self.terminal_result(lock);
                if (self.closing) return {CommandStatus::Stopped, "Engine renderer is shutting down"};
                self.push_pending(PendingEffect{std::move(effect), std::move(owned)});
                if (!self.worker_active) self.pump_wake->notify();
                return {};
            } catch (RendererError& error) {
                return self.latch_producer_error(lock, std::move(error));
            } catch (...) {
                return self.latch_producer_literal(lock, "Engine guest effect allocation failed",
                                                   CommandStatus::BudgetExceeded);
            }
        }
    }

    static CommandResult ordered_write(void* raw, const GpuAddressSpace& memory,
                                       std::uint64_t address,
                                       std::span<const std::byte> bytes) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        if (!self.threaded) {
            std::unique_lock lock(self.mutex);
            if (self.stopped) return self.stopped->command();
            if (!self.literal_stop.ok()) return self.literal_stop;
            if (bytes.empty()) return {};
            const auto ready = self.cohere_locked(memory, address, bytes.size(), false);
            if (!ready.ok()) return ready;
            const auto host = host_address(memory, address, bytes.size());
            if (!host) return self.stop_literal("Engine host write range is not backed",
                                                CommandStatus::MemoryFault);
            host_write(nullptr, *host, bytes);
            return {};
        }
        std::unique_lock producer_lock(self.producer_mutex);
        std::unique_lock lock(self.queue_mutex);
        if (self.terminal.load(std::memory_order_acquire)) return self.terminal_result(lock);
        if (self.closing) return {CommandStatus::Stopped, "Engine renderer is shutting down"};
        if (bytes.empty()) return {};
        try {
            if (bytes.size() > write_capacity)
                throw failure("Engine queued write exceeds the byte budget", CommandStatus::BudgetExceeded);
            if (!host_address(memory, address, bytes.size()))
                throw failure("Engine queued write range is not backed", CommandStatus::MemoryFault);
            if (const auto available = self.wait_queue_slot(lock, bytes.size()); !available.ok())
                return available;
            // Apply backpressure before allocating, but never copy payloads
            // while holding the consumer queue lock.
            lock.unlock();
            PendingWrite request{address, {bytes.begin(), bytes.end()}, self.retain_memory(memory)};
            lock.lock();
            if (self.terminal.load(std::memory_order_acquire)) return self.terminal_result(lock);
            if (self.closing) return {CommandStatus::Stopped, "Engine renderer is shutting down"};
            self.push_pending(std::move(request));
            self.write_bytes += bytes.size();
            if (!self.worker_active) self.pump_wake->notify();
            return {};
        } catch (RendererError& error) {
            return self.latch_producer_error(lock, std::move(error));
        } catch (...) {
            return self.latch_producer_literal(lock, "Engine queued write allocation failed",
                                               CommandStatus::BudgetExceeded);
        }
    }

    static CommandResult draw(void* raw, const RenderStop& stop, const GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        return self.execute(stop, memory, false);
    }
    static CommandResult status(void* raw) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        if (!self.terminal.load(std::memory_order_acquire)) return {};
        std::lock_guard lock(self.mutex);
        if (self.stopped) return self.stopped->command();
        return self.literal_stop;
    }
    static CommandResult clear(void* raw, const RenderStop& stop, const GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        return self.execute(stop, memory, true);
    }
    // Persistent engines encode compute launches into the open batch like
    // draws because dispatch shares the frame command buffer. The
    // legacy engine keeps its synchronous Metal 4 compute backend.
    static CommandResult dispatch(void* raw, const RenderStop& stop,
                                  const GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        if (self.options.mode == MetalEngineMode::Persistent) {
            return self.execute(stop, memory, false);
        }
        std::lock_guard lock(self.mutex);
        if (self.stopped) return self.stopped->command();
        if (!self.literal_stop.ok()) return self.literal_stop;
        @autoreleasepool {
            try {
                const auto before = self.compute_statistics.completed_dispatches;
                const auto executed = execute_metal_compute(&self.compute_context, stop, memory);
                if (!executed.ok()) throw RendererError(executed);
                if (self.options.statistics) {
                    self.options.statistics->completed_dispatches +=
                        self.compute_statistics.completed_dispatches - before;
                }
                return {};
            } catch (RendererError& error) {
                self.stopped.emplace(std::move(error));
                return self.stopped->command();
            } catch (...) {
                return self.stop_literal("Engine renderer compute dispatch failed",
                                         CommandStatus::Stopped);
            }
        }
    }
    // 2D blits execute outside the submission queue on their own synchronous
    // command buffer, but share this backend's registry domain, texture cache
    // and ordered submission identity. An open draw batch commits and drains
    // first, because the blit waits for its source's producers.
    static CommandResult blit(void* raw, const RenderStop& stop, const GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        if (self.threaded) return self.enqueue_render(stop, memory, PendingRenderKind::Blit);
        // Publication can release the final external owner. Keep the backend
        // alive until this callback has finished and released its mutex.
        const auto owner = self.shared_from_this();
        std::unique_lock lock(self.mutex);
        if (const auto ready = self.finish_pending(lock, true); !ready.ok()) return ready;
        if (self.stopped) return self.stopped->command();
        if (!self.literal_stop.ok()) return self.literal_stop;
        return self.blit_locked(stop, memory);
    }
    // Guest-visible synchronization point: write every GPU-dirty interval back
    // to guest memory before the front end publishes a semaphore, report or
    // syncpoint, or touches guest memory itself.
    static CommandResult coherence(void* raw, const GpuAddressSpace& memory,
                                   std::uint64_t gpu_address, std::uint64_t size) noexcept {
        return cohere(raw, memory, gpu_address, size, true);
    }
    static CommandResult read_coherence(void* raw, const GpuAddressSpace& memory,
                                        std::uint64_t gpu_address, std::uint64_t size) noexcept {
        return cohere(raw, memory, gpu_address, size, false);
    }
    static CommandResult cohere(void* raw, const GpuAddressSpace& memory,
                                std::uint64_t gpu_address, std::uint64_t size,
                                bool prepare_write) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        if (self.threaded) {
            return self.enqueue_coherence(memory, gpu_address, size, prepare_write);
        }
        const auto owner = self.shared_from_this();
        std::unique_lock lock(self.mutex);
        // A ranged read (command fetch) waits for queued guest effects only
        // when one of them stores into the bytes it reads; GPU results of
        // executed work are still resolved below. Parked renders keep their
        // place behind the effects.
        const bool read_clear_of_effects = !prepare_write && size != 0 &&
            std::ranges::none_of(self.pending, [&](const auto& item) {
                const auto* effect = std::get_if<PendingEffect>(&item);
                return effect && effect->effect.size &&
                       effect->effect.address < gpu_address + size &&
                       gpu_address < effect->effect.address + effect->effect.size;
            });
        if (!read_clear_of_effects) {
            if (const auto ready = self.finish_pending(lock, true); !ready.ok()) return ready;
        }
        if (self.stopped) return self.stopped->command();
        if (!self.literal_stop.ok()) return self.literal_stop;
        return self.cohere_locked(memory, gpu_address, size, prepare_write);
    }
    // Host-addressable guest memory: the bytes at a guest address are the host
    // bytes at that address, so the write-back copies there directly.
    static bool host_read(void*, std::uint64_t address, std::span<std::byte> output) noexcept {
        std::memcpy(output.data(), reinterpret_cast<const void*>(address), output.size());
        return true;
    }
    static bool host_write(void*, std::uint64_t address, std::span<const std::byte> input) noexcept {
        auto* host = reinterpret_cast<void*>(address);
        if (memory::WriteTracker::try_alias_store(host, input)) return true;
        memory::WriteTracker::prepare_host_write(host, input.size());
        std::memcpy(host, input.data(), input.size());
        return true;
    }
    static CommandResult resolve_guest(void* raw, std::uint64_t guest_address, std::uint64_t size) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        if (self.threaded) return self.enqueue_resolve(guest_address, size);
        const auto owner = self.shared_from_this();
        std::unique_lock lock(self.mutex);
        if (const auto ready = self.finish_pending(lock, true); !ready.ok()) return ready;
        if (self.stopped) return self.stopped->command();
        if (!self.literal_stop.ok()) return self.literal_stop;
        return self.resolve_locked(guest_address, size);
    }
    static CommandResult memory_changed(void* raw, std::span<const Mapping> removed,
        std::span<const std::shared_ptr<void>> released, MemoryCallbacks callbacks) noexcept {
        auto& self = *static_cast<EngineBackend*>(raw);
        const CallbackScope callback(self);
        if (self.threaded) return self.enqueue_memory_change(removed, released, callbacks);
        const auto owner = self.shared_from_this();
        std::unique_lock lock(self.mutex);
        if (const auto ready = self.finish_pending(lock, true); !ready.ok()) return ready;
        if (self.stopped) return self.stopped->command();
        if (!self.literal_stop.ok()) return self.literal_stop;
        return self.memory_changed_locked(removed, released, callbacks);
    }

    CommandResult execute(const RenderStop& stop, const GpuAddressSpace& memory, bool clear) noexcept {
        if (threaded) return enqueue_render(stop, memory,
            clear ? PendingRenderKind::Clear :
            stop.kind == RenderKind::Compute ? PendingRenderKind::Dispatch : PendingRenderKind::Draw);
        std::lock_guard lock(mutex);
        if (stopped) return stopped->command();
        if (!literal_stop.ok()) return literal_stop;
        return execute_locked(stop, memory, clear);
    }

    CommandResult execute_locked(const RenderStop& stop, const GpuAddressSpace& memory, bool clear) noexcept {
        if (stopped) return stopped->command();
        if (!literal_stop.ok()) return literal_stop;
        if (stop.kind == RenderKind::Compute && stop.compute &&
            std::ranges::find(stop.compute->grid, 0u) != stop.compute->grid.end()) {
            if (options.statistics) ++options.statistics->completed_dispatches;
            return {};
        }
        const SlowCall slow{"execute", unsigned(clear ? RenderKind::Clear : stop.kind)};
        @autoreleasepool {
            try {
                if (options.mode == MetalEngineMode::Legacy) {
                    MetalDrawContext context{options.lookup, options.lookup_context, &legacy_draw_statistics};
                    const auto result = clear ? execute_metal_color_clear(&legacy_clear_statistics, stop, memory)
                                              : execute_metal_draw(&context, stop, memory);
                    if (!result.ok()) throw RendererError(result);
                    if (options.statistics) {
                        if (clear) ++options.statistics->completed_clears;
                        else ++options.statistics->completed_draws;
                    }
                } else {
                    // The persistent renderer owns attachment, state and shader
                    // admission. Successful return still requires coherent completion
                    // unless deferred write-back batches the work below.
                    if (!resources) create_resources();
                    struct Scope {
                        ScopedMemory& memory;
                        ~Scope() { memory.active = {}; }
                    } scope{*resources->backend};
                    resources->backend->active = memory.backing_callbacks();
                    if (options.defer_gpu_writeback) check(surface_async_failure());
                    prepare_memory(memory);
                    if (sequence == std::numeric_limits<std::uint64_t>::max())
                        throw failure("Engine renderer submission sequence exhausted", CommandStatus::BudgetExceeded);
                    auto effects = std::make_shared<Effects>(resources->memory, resources->recovery);
                    // The renderer reads the stop only while preparing inside
                    // this call: lend it instead of copying 16 KiB per draw.
                    auto receipt = take(resources->renderer->enqueue(
                        {{domain, ++sequence},
                         std::shared_ptr<const RenderStop>(std::shared_ptr<const RenderStop>{}, &stop),
                         resources->memory, {}, effects, options.defer_gpu_writeback}));
                    const auto kind = clear ? RenderKind::Clear : stop.kind;
                    if (options.defer_gpu_writeback) {
                        // The draw only encodes into the open batch here. The GPU
                        // wait and guest writeback happen at the coherence,
                        // ownership, budget or shutdown boundary that follows.
                        outstanding.push_back({receipt.completion, kind});
                        if (++batched >= options.max_batch_draws)
                            check(finish_batch(false, FlushReason::Budget));
                        else
                            check(surface_async_failure());
                    } else {
                        check(drain_and_measure());
                        const auto completion = receipt.completion.snapshot();
                        if (completion.error) throw *completion.error;
                        if (!completion.guest_visible_complete())
                            throw failure("Engine renderer drained without coherent completion", CommandStatus::Stopped);
                        count_completed(kind);
                        record_statistics();
                    }
                }
                return {};
            } catch (RendererError& error) {
                if (commit_hold_blocked(RendererStatus(error)))
                    return {CommandStatus::Stopped, "Engine render awaits held commit"};
                stopped.emplace(std::move(error));
                return stopped->command();
            } catch (const std::bad_alloc&) {
                return stop_literal("Engine renderer allocation failed", CommandStatus::BudgetExceeded);
            } catch (...) {
                return stop_literal("Engine renderer provider or backend threw an exception", CommandStatus::Stopped);
            }
        }
    }

private:
    CommandResult wait_queue_slot(std::unique_lock<std::mutex>& lock,
                                  std::size_t bytes = 0) noexcept {
        wake.wait(lock, [&] {
            return (pending.size() + std::size_t(worker_active) < queue_capacity &&
                    bytes <= write_capacity - write_bytes) || closing ||
                   terminal.load(std::memory_order_acquire);
        });
        if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
        if (closing) return {CommandStatus::Stopped, "Engine renderer is shutting down"};
        return {};
    }
    CommandResult terminal_result(std::unique_lock<std::mutex>& queue_lock) noexcept {
        if (queue_lock.owns_lock()) queue_lock.unlock();
        CommandResult result;
        {
            std::lock_guard state_lock(mutex);
            result = stopped ? stopped->command() : literal_stop;
        }
        queue_lock.lock();
        return result;
    }
    CommandResult latch_producer_error(std::unique_lock<std::mutex>& queue_lock,
                                       RendererError error) noexcept {
        if (queue_lock.owns_lock()) queue_lock.unlock();
        CommandResult result;
        {
            std::lock_guard state_lock(mutex);
            if (!terminal.load(std::memory_order_relaxed)) stopped.emplace(std::move(error));
            result = stopped ? stopped->command() : literal_stop;
            terminal.store(true, std::memory_order_release);
        }
        queue_lock.lock();
        auto cancelled = cancel_pending(result);
        wake.notify_all();
        queue_lock.unlock();
        cancelled.clear();
        queue_lock.lock();
        pump_wake->notify();
        return result;
    }
    CommandResult latch_producer_literal(std::unique_lock<std::mutex>& queue_lock,
                                         const char* text, CommandStatus status) noexcept {
        if (queue_lock.owns_lock()) queue_lock.unlock();
        CommandResult result;
        {
            std::lock_guard state_lock(mutex);
            if (!terminal.load(std::memory_order_relaxed)) result = stop_literal(text, status);
            else result = stopped ? stopped->command() : literal_stop;
            terminal.store(true, std::memory_order_release);
        }
        queue_lock.lock();
        auto cancelled = cancel_pending(result);
        wake.notify_all();
        queue_lock.unlock();
        cancelled.clear();
        queue_lock.lock();
        pump_wake->notify();
        return result;
    }
    void complete(const std::shared_ptr<BarrierResult>& completion, CommandResult result) noexcept {
        completion->result = result;
        completion->done = true;
        completion->ready.notify_all();
    }
    std::list<Pending> cancel_pending(CommandResult result) noexcept {
        for (auto& request : pending) {
            if (const auto* write = std::get_if<PendingWrite>(&request)) write_bytes -= write->bytes.size();
            if (auto* barrier = std::get_if<PendingCoherence>(&request))
                complete(barrier->completion, result);
            else if (auto* barrier = std::get_if<PendingResolve>(&request))
                complete(barrier->completion, result);
            else if (auto* barrier = std::get_if<PendingMemoryChange>(&request))
                complete(barrier->completion, result);
        }
        return std::exchange(pending, {});
    }
    CommandResult wait_barrier(std::unique_lock<std::mutex>& lock,
                               const std::shared_ptr<BarrierResult>& completion) noexcept {
        // Queued cancellation completes the result in place; an active barrier
        // must keep its borrowed callbacks alive until the worker completes it.
        completion->ready.wait(lock, [&] { return completion->done; });
        return completion->result;
    }
    CommandResult enqueue_render(const RenderStop& stop, const GpuAddressSpace& memory,
                                 PendingRenderKind kind) noexcept {
        std::unique_lock producer_lock(producer_mutex);
        std::unique_lock lock(queue_mutex);
        if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
        try {
            if (const auto available = wait_queue_slot(lock); !available.ok()) return available;
            std::unique_ptr<RenderStop> snapshot;
            if (!render_pool.empty()) {
                snapshot = std::move(render_pool.back());
                render_pool.pop_back();
            }
            lock.unlock();
            if (!snapshot) snapshot = std::make_unique<RenderStop>();
            // Queue a complete source snapshot so the backend can consume it
            // directly without reconstructing metadata or register state.
            *snapshot = stop;
            auto request = PendingRender{std::move(snapshot), retain_memory(memory), kind};
            lock.lock();
            if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
            if (closing) return {CommandStatus::Stopped, "Engine renderer is shutting down"};
            push_pending(std::move(request));
            if (!worker_active) pump_wake->notify();
            return {};
        } catch (RendererError& error) {
            return latch_producer_error(lock, std::move(error));
        } catch (...) {
            return latch_producer_literal(lock, "Engine queued render allocation failed",
                                          CommandStatus::BudgetExceeded);
        }
    }
    static std::optional<std::uint64_t> host_address(const GpuAddressSpace& memory,
        std::uint64_t address, std::uint64_t size) noexcept {
        const auto mappings = memory.mappings();
        auto after = std::upper_bound(mappings.begin(), mappings.end(), address,
            [](auto value, const auto& mapping) { return value < mapping.gpu_base; });
        if (after == mappings.begin()) return std::nullopt;
        const auto& mapping = *std::prev(after);
        const auto offset = address - mapping.gpu_base;
        const auto backed = backed_bytes(mapping);
        if (offset >= mapping.mapped_size || size > mapping.mapped_size - offset ||
            offset > backed || size > backed - offset) return std::nullopt;
        return mapping.guest_address + mapping.buffer_offset + offset;
    }
    // Compare physical byte ranges, so aliases and different retained views
    // cannot hide a dependency. Invalid translations take the ordered path.
    static bool disjoint_store(const GpuAddressSpace& memory, std::uint64_t address,
                               std::uint64_t size, const GpuAddressSpace& written_memory,
                               std::uint64_t written_address, std::uint64_t written_size) noexcept {
        if (!written_size) return true;
        const auto first = host_address(memory, address, size);
        const auto second = host_address(written_memory, written_address, written_size);
        if (!first || !second) return false;
        return *first < *second ? size <= *second - *first : written_size <= *first - *second;
    }
    static bool read_can_pass(const Pending& request, const GpuAddressSpace& memory,
                              std::uint64_t address, std::uint64_t size) noexcept {
        if (const auto* write = std::get_if<PendingWrite>(&request))
            return disjoint_store(memory, address, size, *write->memory,
                                  write->address, write->bytes.size());
        if (const auto* effect = std::get_if<PendingEffect>(&request))
            return disjoint_store(memory, address, size, *effect->memory,
                                  effect->effect.address, effect->effect.size);
        if (const auto* read = std::get_if<PendingCoherence>(&request)) return read->bypass;
        // Render write intents are unknown until backend preparation. Mapping
        // changes and other barriers also retain their place in the stream.
        return false;
    }
    CommandResult enqueue_coherence(const GpuAddressSpace& memory, std::uint64_t gpu_address,
                                    std::uint64_t size, bool prepare_write) noexcept {
        std::unique_lock producer_lock(producer_mutex);
        std::unique_lock lock(queue_mutex);
        if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
        try {
            if (const auto available = wait_queue_slot(lock); !available.ok()) return available;
            lock.unlock();
            auto completion = std::make_shared<BarrierResult>();
            auto request = PendingCoherence{retain_memory(memory), gpu_address, size,
                                            prepare_write, completion};
            lock.lock();
            if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
            if (closing) return {CommandStatus::Stopped, "Engine renderer is shutting down"};
            // Only reads proven disjoint from every skipped operation may
            // overtake the queue. GPU results already registered by the backend
            // are still resolved by cohere_locked on the backend thread.
            request.bypass = !prepare_write && size &&
                (!active_request || read_can_pass(*active_request, memory, gpu_address, size)) &&
                std::ranges::all_of(pending, [&](const auto& item) {
                    return read_can_pass(item, memory, gpu_address, size);
                });
            if (request.bypass) {
                const auto position = std::find_if(pending.begin(), pending.end(), [](const auto& item) {
                    const auto* read = std::get_if<PendingCoherence>(&item);
                    return !read || !read->bypass;
                });
                push_pending(std::move(request), position);
            } else push_pending(std::move(request));
            if (!worker_active) pump_wake->notify();
            producer_lock.unlock();
            return wait_barrier(lock, completion);
        } catch (RendererError& error) {
            return latch_producer_error(lock, std::move(error));
        } catch (...) {
            return latch_producer_literal(lock, "Engine coherence barrier allocation failed",
                                          CommandStatus::BudgetExceeded);
        }
    }
    CommandResult enqueue_resolve(std::uint64_t guest_address, std::uint64_t size) noexcept {
        std::unique_lock producer_lock(producer_mutex);
        std::unique_lock lock(queue_mutex);
        if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
        try {
            if (const auto available = wait_queue_slot(lock); !available.ok()) return available;
            lock.unlock();
            auto completion = std::make_shared<BarrierResult>();
            lock.lock();
            if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
            if (closing) return {CommandStatus::Stopped, "Engine renderer is shutting down"};
            push_pending(PendingResolve{guest_address, size, completion});
            if (!worker_active) pump_wake->notify();
            producer_lock.unlock();
            return wait_barrier(lock, completion);
        } catch (...) {
            return latch_producer_literal(lock, "Engine resolve barrier allocation failed",
                                          CommandStatus::BudgetExceeded);
        }
    }
    CommandResult enqueue_memory_change(std::span<const Mapping> removed,
        std::span<const std::shared_ptr<void>> released, MemoryCallbacks callbacks) noexcept {
        std::unique_lock producer_lock(producer_mutex);
        std::unique_lock lock(queue_mutex);
        if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
        try {
            if (const auto available = wait_queue_slot(lock); !available.ok()) return available;
            lock.unlock();
            auto completion = std::make_shared<BarrierResult>();
            PendingMemoryChange request{{removed.begin(), removed.end()},
                                        {released.begin(), released.end()}, callbacks, completion};
            lock.lock();
            if (terminal.load(std::memory_order_acquire)) return terminal_result(lock);
            if (closing) return {CommandStatus::Stopped, "Engine renderer is shutting down"};
            push_pending(std::move(request));
            if (!worker_active) pump_wake->notify();
            producer_lock.unlock();
            return wait_barrier(lock, completion);
        } catch (...) {
            return latch_producer_literal(lock, "Engine mapping barrier allocation failed",
                                          CommandStatus::BudgetExceeded);
        }
    }
    std::shared_ptr<const GpuAddressSpace> retain_memory(const GpuAddressSpace& memory) {
        if (auto retained = retained_view.lock(); retained && retained->generation() == memory.generation())
            return retained;
        for (const auto& mapping : memory.mappings())
            if (!mapping.owner) throw failure("Engine ordered work requires pinned memory");
        auto made = make_address_space(memory.mappings(), {host_read, host_write, nullptr}, memory.generation());
        if (const auto* error = std::get_if<MemoryResult>(&made))
            throw failure(error->detail.data(), CommandStatus::MemoryFault);
        auto retained = std::make_shared<GpuAddressSpace>(std::get<GpuAddressSpace>(std::move(made)));
        retained_view = retained;
        return retained;
    }
    bool idle_poll_needed() const noexcept {
        return !terminal.load(std::memory_order_acquire) && batched == 0 && !outstanding.empty();
    }
    CommandResult poll_idle(std::unique_lock<std::mutex>& queue_lock) noexcept {
        if (closing || !idle_poll_needed()) return {};
        if (queue_lock.owns_lock()) queue_lock.unlock();
        CommandResult result;
        {
            std::lock_guard state_lock(mutex);
            try {
                check(surface_async_failure());
            } catch (RendererError& error) {
                stopped.emplace(std::move(error));
                result = stopped->command();
            } catch (...) {
                result = stop_literal("Engine asynchronous status polling failed",
                                      CommandStatus::Stopped);
            }
            if (!result.ok()) terminal.store(true, std::memory_order_release);
        }
        queue_lock.lock();
        if (!result.ok()) {
            auto cancelled = cancel_pending(result);
            wake.notify_all();
            queue_lock.unlock();
            cancelled.clear();
            queue_lock.lock();
        }
        return result;
    }
    // Effects retain their FIFO position while disjoint later renders prepare
    // and encode. Native commits stay held until prefix authority is coherent
    // and published; deferred-store guards reject dependent preparation.
    // Ordered writes, mappings, blits and full barriers remain fences.
    CommandResult finish_pending(std::unique_lock<std::mutex>& queue_lock, bool) noexcept {
        CommandResult last = poll_idle(queue_lock);
        if (!last.ok()) return last;
        bool look_ahead = false;
        while (!pending.empty() && !closing &&
               !terminal.load(std::memory_order_acquire)) {
            auto selected = pending.begin();
            if (look_ahead) {
                const auto* effect = std::get_if<PendingEffect>(&*selected);
                if (effect) {
                    if (!effect->boundary || !resources ||
                        !resources->renderer->can_accept_while_held()) return {};
                    if (++selected == pending.end()) return {};
                    const auto* render = std::get_if<PendingRender>(&*selected);
                    if (!render || render->kind == PendingRenderKind::Blit ||
                        render->memory->generation() != effect->memory->generation()) return {};
                }
            }
            std::optional<Pending> request(std::move(*selected));
            free_pending.splice(free_pending.end(), pending, selected);
            worker_active = true;
            active_request = &*request;
            wake.notify_all();
            queue_lock.unlock();

            bool retry_effect = false, retry_render = false;
            {
                std::lock_guard state_lock(mutex);
                if (terminal.load(std::memory_order_acquire)) {
                    last = stopped ? stopped->command() : literal_stop;
                } else try {
                    if (auto* render = std::get_if<PendingRender>(&*request)) {
                        last = render->kind == PendingRenderKind::Blit
                            ? blit_locked(*render->stop, *render->memory)
                            : execute_locked(*render->stop, *render->memory,
                                             render->kind == PendingRenderKind::Clear);
                        if (last.status == CommandStatus::Stopped &&
                            last.detail == "Engine render awaits held commit") {
                            retry_render = true;
                            last = {};
                        }
                    } else if (auto* write = std::get_if<PendingWrite>(&*request)) {
                        last = cohere_locked(*write->memory, write->address, write->bytes.size(), false);
                        if (last.ok()) {
                            // Reported to the renderer's caches instead of the
                            // tracker's unnoted clock.
                            const memory::WriteTracker::NotifiedStores reported;
                            const auto stored = write->memory->write(write->address, write->bytes);
                            if (!stored) throw RendererError({CommandStatus::MemoryFault, stored.detail});
                            if (resources && resources->renderer)
                                resources->renderer->note_engine_store(write->address, write->bytes.size());
                        }
                    } else if (auto* barrier = std::get_if<PendingCoherence>(&*request)) {
                        last = cohere_locked(*barrier->memory, barrier->gpu_address,
                                             barrier->size, barrier->prepare_write);
                        if (last.status == CommandStatus::Stopped &&
                            last.detail == "Engine render awaits held commit") {
                            retry_render = true;
                            last = {};
                        }
                    } else if (auto* barrier = std::get_if<PendingResolve>(&*request)) {
                        last = resolve_locked(barrier->guest_address, barrier->size);
                    } else if (auto* barrier = std::get_if<PendingMemoryChange>(&*request)) {
                        last = memory_changed_locked(barrier->removed, barrier->released,
                                                     barrier->callbacks);
                    } else {
                        auto& effect = std::get<PendingEffect>(*request);
                        if (!effect.committed) {
                            check(finish_batch(false, FlushReason::CoherenceFull));
                            if (!outstanding.empty()) effect.prefix = outstanding.back().first;
                            if (!resources) create_resources();
                            prepare_memory(*effect.memory);
                            effect.boundary = take(resources->memory->capture_guest_boundary(
                                options.gpu_resident_textures));
                            if (effect.effect.size) {
                                const auto range = take(resources->memory->retain(effect.effect.address,
                                    effect.effect.size, ResourceAccess::ReadWrite));
                                resources->deferred_store = take(resources->memory->defer_host_store(range.bytes));
                            }
                            check(resources->renderer->set_commit_hold(true));
                            resources->backend->held_prefix = SubmissionId{domain, sequence};
                            effect.committed = true;
                            effect.expires = deadline(options.completion_timeout_ns);
                        }
                        check(surface_async_failure());
                        bool ready = !effect.prefix.valid() ||
                            effect.prefix.snapshot().guest_visible_complete();
                        if (ready) {
                            if (!resources) create_resources();
                            auto& backend = *resources->backend;
                            struct Scope {
                                ScopedMemory& memory;
                                ~Scope() { memory.active = {}; }
                            } scope{backend};
                            backend.active = {host_read, host_write, nullptr};
                            if (!effect.coherent) {
                                prepare_memory(*effect.memory);
                                check(effect.boundary ? effect.boundary->publish() :
                                    options.gpu_resident_textures
                                        ? resources->memory->guest_boundary()
                                        : resources->memory->submission_boundary());
                                effect.coherent = true;
                            }
                            if (effect.effect.size) {
                                const memory::WriteTracker::NotifiedStores reported;
                                const auto written = effect.memory->write(effect.effect.address,
                                    std::span(effect.effect.payload).first(effect.effect.size));
                                if (!written)
                                    throw failure("Engine guest effect store failed",
                                                  CommandStatus::MemoryFault);
                                resources->renderer->note_engine_store(effect.effect.address, effect.effect.size);
                            }
                            if (const auto& acquire = effect.effect.acquire) {
                                std::array<std::byte, 4> word{};
                                if (!effect.memory->read(effect.effect.address, word))
                                    throw failure("Engine semaphore acquire read failed",
                                                  CommandStatus::MemoryFault);
                                std::uint32_t current = 0;
                                std::memcpy(&current, word.data(), sizeof current);
                                const bool met = acquire->operation == 1 ? current == acquire->value
                                    : acquire->operation == 4 ? current >= acquire->value
                                    : (current & acquire->value) != 0;
                                if (!met)
                                    throw failure("puller semaphore acquire dependency is unmet",
                                                  CommandStatus::Unsupported);
                            }
                            if (effect.effect.publish) {
                                const auto published = effect.effect.publish();
                                ready = published.has_value();
                                if (published && !published->ok()) throw RendererError(*published);
                            }
                        }
                        if (ready && effect.boundary) {
                            resources->deferred_store.reset();
                            resources->backend->held_prefix.reset();
                            check(resources->renderer->set_commit_hold(false));
                        }
                        if (!ready) {
                            if (deadline(0) >= effect.expires)
                                throw failure("Engine ordered effect completion timed out",
                                              CommandStatus::Stopped);
                            retry_effect = true;
                        }
                    }
                } catch (RendererError& error) {
                    stopped.emplace(std::move(error));
                    last = stopped->command();
                } catch (...) {
                    last = stop_literal("Engine ordered work failed", CommandStatus::Stopped);
                }
                if (!last.ok()) terminal.store(true, std::memory_order_release);
            }

            queue_lock.lock();
            worker_active = false;
            active_request = nullptr;
            if (const auto* write = std::get_if<PendingWrite>(&*request)) write_bytes -= write->bytes.size();
            if (auto* barrier = std::get_if<PendingCoherence>(&*request)) {
                if (!retry_render) complete(barrier->completion, last);
            }
            else if (auto* barrier = std::get_if<PendingResolve>(&*request))
                complete(barrier->completion, last);
            else if (auto* barrier = std::get_if<PendingMemoryChange>(&*request))
                complete(barrier->completion, last);

            if (closing) {
                auto cancelled = cancel_pending(CommandResult{
                    CommandStatus::Stopped, "Engine renderer stopped during shutdown"});
                wake.notify_all();
                queue_lock.unlock();
                request.reset();
                cancelled.clear();
                queue_lock.lock();
                return last;
            }
            if (!last.ok()) {
                auto cancelled = cancel_pending(last);
                wake.notify_all();
                queue_lock.unlock();
                request.reset();
                cancelled.clear();
                queue_lock.lock();
                return last;
            }
            if (retry_render) {
                // Producers inspect the active read's bypass flag under this
                // lock; change it only after active_request has been cleared.
                if (auto* barrier = std::get_if<PendingCoherence>(&*request)) barrier->bypass = false;
                auto position = pending.begin();
                while (position != pending.end()) {
                    const auto* read = std::get_if<PendingCoherence>(&*position);
                    if (!read || !read->bypass) break;
                    ++position;
                }
                // Concurrent disjoint reads may precede the parked effect.
                // The retry must follow the effect even when it is not first.
                if (position != pending.end() && std::holds_alternative<PendingEffect>(*position))
                    ++position;
                push_pending(std::move(*request), position);
                wake.notify_all();
                return {};
            }
            if (retry_effect) {
                // A disjoint read admitted while this effect was active can
                // make progress before the next publication retry.
                const auto position = std::find_if(pending.begin(), pending.end(), [](const auto& item) {
                    const auto* read = std::get_if<PendingCoherence>(&item);
                    return !read || !read->bypass;
                });
                push_pending(std::move(*request), position);
                wake.notify_all();
                queue_lock.unlock();
                request.reset();
                queue_lock.lock();
                look_ahead = true;
                continue;
            }
            look_ahead = false;
            if (auto* render = std::get_if<PendingRender>(&*request); render && render->stop)
                render_pool.push_back(std::move(render->stop));
            wake.notify_all();
            queue_lock.unlock();
            request.reset();
            queue_lock.lock();
        }
        return last;
    }
    CommandResult blit_locked(const RenderStop& stop, const GpuAddressSpace& memory) noexcept {
        if (options.mode != MetalEngineMode::Persistent || !device)
            return {CommandStatus::Unsupported,
                    "2D blit execution requires the persistent engine renderer"};
        @autoreleasepool {
            try {
                if (!resources) create_resources();
                struct Scope {
                    ScopedMemory& memory;
                    ~Scope() { memory.active = {}; }
                } scope{*resources->backend};
                resources->backend->active = memory.backing_callbacks();
                prepare_memory(memory);
                check(finish_batch(true, FlushReason::Blit));
                if (sequence == std::numeric_limits<std::uint64_t>::max())
                    throw failure("Engine renderer submission sequence exhausted",
                                  CommandStatus::BudgetExceeded);
                MetalTwodResources twod{device->device(), device->queue(), resources->memory,
                                        resources->textures.get(), {domain, ++sequence},
                                        options.completion_timeout_ns,
                                        options.defer_gpu_writeback};
                const auto result = execute_metal_twod_blit(twod, stop, memory);
                if (const auto* error = std::get_if<RendererError>(&result)) {
                    stopped.emplace(*error);
                    return stopped->command();
                }
                if (options.statistics) ++options.statistics->completed_blits;
                return {};
            } catch (RendererError& error) {
                stopped.emplace(std::move(error));
                return stopped->command();
            } catch (const std::bad_alloc&) {
                return stop_literal("Engine renderer allocation failed", CommandStatus::BudgetExceeded);
            } catch (...) {
                return stop_literal("Engine renderer provider or backend threw an exception",
                                    CommandStatus::Stopped);
            }
        }
    }
    CommandResult cohere_locked(const GpuAddressSpace& memory, std::uint64_t gpu_address,
                                std::uint64_t size, bool prepare_write) noexcept {
        if (!resources) return {};
        const SlowCall slow{"coherence", size != 0};
        @autoreleasepool {
            auto& backend = *resources->backend;
            struct Scope { ScopedMemory& memory; ~Scope() { memory.active = {}; } } scope{backend};
            backend.active = memory.backing_callbacks();
            try {
                prepare_memory(memory);
                // Clean reads and writes leave the open batch recording. Only
                // writes prepare a host store: a command fetch must preserve
                // the write-generation certificates of unchanged resources.
                if (size != 0 && !gpu_writes_overlap(gpu_address, size)) {
                    if (prepare_write) prepare_host_store(gpu_address, size);
                    return {};
                }
                check(finish_batch(true, size ? FlushReason::CoherenceRanged
                                              : FlushReason::CoherenceFull));
                if (size == 0) {
                    check(options.gpu_resident_textures ? resources->memory->guest_boundary()
                                                        : resources->memory->submission_boundary());
                    return {};
                }
                const std::uint64_t end = gpu_address + size;
                if (end < gpu_address)
                    throw failure("Engine coherence range overflows", CommandStatus::MemoryFault);
                for (auto it = first_entry(gpu_address);
                     it != backend.entries.end() && it->mapping.gpu_base < end; ++it) {
                    const auto& entry = *it;
                    const auto& mapping = entry.mapping;
                    const std::uint64_t first = std::max(gpu_address, mapping.gpu_base);
                    const std::uint64_t last = std::min({end, mapping.gpu_base + mapping.mapped_size,
                                                         mapping.gpu_base + backed_bytes(mapping)});
                    if (first >= last) continue;
                    check(resources->memory->resolve({entry.backing,
                        mapping.buffer_offset + (first - mapping.gpu_base), last - first},
                        ResolveReason::CpuEngineRead));
                }
                if (prepare_write) prepare_host_store(gpu_address, size);
                return {};
            } catch (RendererError& error) {
                if (commit_hold_blocked(RendererStatus(error)))
                    return {CommandStatus::Stopped, "Engine render awaits held commit"};
                stopped.emplace(std::move(error));
                return stopped->command();
            } catch (...) {
                return stop_literal("Engine renderer coherence failed", CommandStatus::Stopped);
            }
        }
    }
    CommandResult resolve_locked(std::uint64_t guest_address, std::uint64_t size) noexcept {
        if (!resources || !size) return {};
        if (guest_address > std::numeric_limits<std::uint64_t>::max() - size)
            return stop_literal("Engine guest resolve range overflows", CommandStatus::MemoryFault);
        @autoreleasepool {
            auto& backend = *resources->backend;
            struct Scope { ScopedMemory& memory; ~Scope() { memory.active = {}; } } scope{backend};
            backend.active = {host_read, host_write, nullptr};
            try {
                check(finish_batch(true, FlushReason::CoherenceRanged));
                const auto end = guest_address + size;
                for (const auto& allocation : backend.allocations) {
                    const auto base = allocation.mapping.guest_address;
                    const auto first = std::max(guest_address, base);
                    const auto last = std::min(end, base + allocation.mapping.backing_size);
                    if (first >= last) continue;
                    check(resources->memory->resolve({allocation.backing, first - base, last - first},
                                                     ResolveReason::CpuEngineRead));
                }
                return {};
            } catch (RendererError& error) {
                stopped.emplace(std::move(error));
                return stopped->command();
            } catch (...) {
                return stop_literal("Engine guest resolve failed", CommandStatus::Stopped);
            }
        }
    }
    CommandResult memory_changed_locked(std::span<const Mapping> removed,
        std::span<const std::shared_ptr<void>> released, MemoryCallbacks callbacks) noexcept {
        if (!resources) return {};
        @autoreleasepool {
            auto& backend = *resources->backend;
            struct Scope { ScopedMemory& memory; ~Scope() { memory.active = {}; } } scope{backend};
            backend.active = callbacks;
            try {
                check(finish_batch(true, FlushReason::MemoryChanged));
                std::vector<MappingId> mappings;
                std::vector<BackingId> backings;
                for (const auto& entry : backend.entries)
                    if (std::any_of(removed.begin(), removed.end(), [&](const auto& mapping) {
                        return same_mapping(mapping, entry.mapping);
                    })) mappings.push_back(entry.id);
                for (const auto& allocation : backend.allocations) {
                    if (!std::any_of(released.begin(), released.end(), [&](const auto& owner) {
                        return same_owner(owner, allocation.mapping.owner);
                    })) continue;
                    check(resources->memory->resolve({allocation.backing, 0,
                        allocation.mapping.backing_size}, ResolveReason::SubmissionBoundary));
                    backings.push_back(allocation.backing);
                }
                check(resources->memory->invalidate_mappings(mappings, backings));
                std::erase_if(backend.entries, [&](const auto& entry) {
                    return std::find(mappings.begin(), mappings.end(), entry.id) != mappings.end() ||
                        std::find(backings.begin(), backings.end(), entry.backing) != backings.end();
                });
                std::erase_if(backend.allocations, [&](const auto& allocation) {
                    return std::find(backings.begin(), backings.end(), allocation.backing) != backings.end();
                });
                backend.forget_views();
                return {};
            } catch (RendererError& error) {
                stopped.emplace(std::move(error));
                return stopped->command();
            } catch (...) {
                return stop_literal("Engine renderer mapping invalidation failed", CommandStatus::Stopped);
            }
        }
    }
    // Whether any accepted or written-back-pending GPU write overlaps the guest
    // GPU range: the registry marks write intents dirty at acceptance.
    // Entries are sorted by GPU base after prepare_memory() and never overlap:
    // the first entry that can contain or follow the address.
    std::vector<ScopedMemory::Entry>::const_iterator first_entry(std::uint64_t gpu_address) const {
        const auto& entries = resources->backend->entries;
        auto it = std::upper_bound(entries.begin(), entries.end(), gpu_address,
            [](std::uint64_t value, const auto& entry) { return value < entry.mapping.gpu_base; });
        if (it != entries.begin() && gpu_address - std::prev(it)->mapping.gpu_base <
                std::prev(it)->mapping.mapped_size) --it;
        return it;
    }
    // External callers may write through the original guest VA, so their
    // coherence callback retains the pre-write contract. Owned engine stores
    // prepare only at host_write(), which selects the alias when available.
    void prepare_host_store(std::uint64_t gpu_address, std::uint64_t size) {
        const std::uint64_t end = gpu_address + size;
        for (auto it = first_entry(gpu_address);
             it != resources->backend->entries.end() && it->mapping.gpu_base < end; ++it) {
            const auto& mapping = it->mapping;
            const std::uint64_t first = std::max(gpu_address, mapping.gpu_base);
            const std::uint64_t last = std::min(end, mapping.gpu_base + mapping.mapped_size);
            if (first >= last) continue;
            resources->memory->prepare_host_store({it->backing,
                mapping.buffer_offset + (first - mapping.gpu_base), last - first});
        }
    }
    static std::uint64_t backed_bytes(const Mapping& mapping) noexcept {
        return mapping.backing_size > mapping.buffer_offset ? mapping.backing_size - mapping.buffer_offset : 0;
    }
    bool gpu_writes_overlap(std::uint64_t gpu_address, std::uint64_t size) {
        const std::uint64_t end = gpu_address + size;
        if (end < gpu_address) throw failure("Engine coherence range overflows", CommandStatus::MemoryFault);
        for (auto it = first_entry(gpu_address);
             it != resources->backend->entries.end() && it->mapping.gpu_base < end; ++it) {
            const auto& mapping = it->mapping;
            const std::uint64_t first = std::max(gpu_address, mapping.gpu_base);
            // Mapping padding past the backing holds no GPU writes.
            const std::uint64_t last = std::min({end, mapping.gpu_base + mapping.mapped_size,
                                                 mapping.gpu_base + backed_bytes(mapping)});
            if (first >= last) continue;
            if (take(resources->memory->gpu_written({it->backing,
                    mapping.buffer_offset + (first - mapping.gpu_base), last - first})))
                return true;
        }
        return false;
    }
    CommandResult stop_literal(const char* text, CommandStatus status) noexcept {
        literal_stop = {status, text};
        return literal_stop;
    }
    // Converts batched receipts that reached coherent completion into the
    // public completed counters. Returns the first failed receipt error so
    // callers can latch it; failed receipts stay until an owner retires them.
    void count_completed(RenderKind kind) {
        if (!options.statistics) return;
        if (kind == RenderKind::Clear) ++options.statistics->completed_clears;
        else if (kind == RenderKind::Compute) ++options.statistics->completed_dispatches;
        else ++options.statistics->completed_draws;
    }
    // Receipts complete in submission order: retire the completed prefix and
    // stop at the first pending or failed one (a failure surfaces when it
    // reaches the front). A full sweep runs only when the list doubled.
    std::optional<RendererError> retire_completed() {
        std::optional<RendererError> failed;
        const auto retire = [&](const auto& entry) {
            const auto snapshot = entry.first.snapshot();
            if (snapshot.error && !failed) failed = snapshot.error;
            if (!snapshot.guest_visible_complete()) return false;
            count_completed(entry.second);
            return true;
        };
        std::size_t settled = 0;
        while (settled < outstanding.size() && retire(outstanding[settled])) ++settled;
        outstanding.erase(outstanding.begin(), outstanding.begin() + std::ptrdiff_t(settled));
        if (!failed && outstanding.size() > std::max<std::size_t>(64, 2 * swept_outstanding)) {
            std::erase_if(outstanding, retire);
            swept_outstanding = outstanding.size();
        }
        return failed;
    }
    std::size_t swept_outstanding = 0;
    void record_statistics(bool wait_for_encoding = true) {
        if (!options.statistics || !resources) return;
        const auto statistics = submission_statistics(*resources->submissions, wait_for_encoding);
        auto& stats = *options.statistics;
        stats.command_buffers += statistics.command_buffers - previous_buffers;
        stats.render_passes += statistics.render_passes - previous_passes;
        previous_buffers = statistics.command_buffers;
        previous_passes = statistics.render_passes;
        stats.draws_per_command_buffer = stats.command_buffers
            ? double(stats.completed_draws) / double(stats.command_buffers) : 0.0;
    }
    // Waits for coherent completion of everything the engine accepted. The
    // measured span is the CPU blocked on the GPU once per frame at its
    // readback boundary instead of once per draw.
    RendererStatus drain_and_measure() {
        const auto started = std::chrono::steady_clock::now();
        auto status = resources->renderer->drain(deadline(options.completion_timeout_ns));
        if (options.statistics)
            options.statistics->cpu_wait_ns += std::uint64_t(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count());
        return status;
    }
    // Commits the open batch. wait=false is budget pressure: the batch splits
    // into a committed command buffer without blocking the CPU. wait=true is
    // an observable boundary: the CPU then waits for coherent completion and
    // converts finished receipts. Callers hold the mutex.
    RendererStatus finish_batch(bool wait, FlushReason reason) {
        if (!resources || !resources->renderer) return renderer_success();
        struct Trace {
            FlushReason reason;
            std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
            ~Trace() { trace_flush(reason, std::chrono::steady_clock::now() - started); }
        } trace{reason};
        // Phase marks for ARMIGO_TRACE_ENGINE_FLUSH: commit, wait, retire.
        std::array<std::chrono::steady_clock::time_point, 3> marks{};
        struct Phases {
            const std::chrono::steady_clock::time_point& started;
            std::array<std::chrono::steady_clock::time_point, 3>& marks;
            std::size_t outstanding;
            ~Phases() {
                if (!trace_flush_enabled()) return;
                const auto now = std::chrono::steady_clock::now();
                if (now - started < std::chrono::seconds(2)) return;
                auto ms = [](auto from, auto to) {
                    return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(to - from).count();
                };
                const auto flushed = marks[0] == decltype(now){} ? now : marks[0];
                const auto drained = marks[1] == decltype(now){} ? now : marks[1];
                armigo::diag::opt_in( "engine_slow_flush outstanding=%zu flush_ms=%lld drain_ms=%lld retire_ms=%lld\n",
                             outstanding, ms(started, flushed), ms(flushed, drained), ms(drained, now));
            }
        } phases{trace.started, marks, outstanding.size()};
        if (const auto flushed = resources->renderer->flush(wait
                ? SubmissionBoundary::SubmissionEnd
                : SubmissionBoundary::ResourceBudget);
            std::holds_alternative<RendererError>(flushed)) return flushed;
        marks[0] = std::chrono::steady_clock::now();
        batched = 0;
        if (wait && !outstanding.empty()) {
            if (const auto drained = drain_and_measure();
                std::holds_alternative<RendererError>(drained)) return drained;
        } else if (const auto polled = resources->renderer->poll();
                   std::holds_alternative<RendererError>(polled)) return polled;
        marks[1] = std::chrono::steady_clock::now();
        if (const auto failed = retire_completed()) return *failed;
        // Optional telemetry must not turn the worker's asynchronous flush
        // into a join of the native encoder. Barriers still report settled counts.
        record_statistics(!threaded || wait);
        return renderer_success();
    }
    // Surfaces a GPU failure that arrived asynchronously for already accepted
    // work. The next draw, clear or coherence call returns it with the
    // original diagnostic and the engine latches it as sticky.
    RendererStatus surface_async_failure() {
        if (!resources || !resources->renderer) return renderer_success();
        if (const auto polled = resources->renderer->poll();
            std::holds_alternative<RendererError>(polled)) return polled;
        if (const auto failed = retire_completed()) return *failed;
        // Per-draw polling leaves the reporting counters to batch boundaries:
        // reading them waits for the encoding worker.
        return renderer_success();
    }
    void prepare_memory(const GpuAddressSpace& memory) {
        auto& backend = *resources->backend;
        if (backend.matches(memory.generation())) return;
        const auto mappings = memory.mappings();
        if (mappings.size() == backend.entries.size() &&
            std::equal(mappings.begin(), mappings.end(), backend.entries.begin(),
                [](const auto& mapping, const auto& entry) {
                    return same_mapping(mapping, entry.mapping);
                })) {
            backend.remember(memory.generation());
            return;
        }
        for (const auto& mapping : mappings)
            if (!mapping.owner) throw failure("Persistent engine renderer requires retained GPU mapping owners");
        // Both views are sorted by GPU base and never overlap there, so each
        // side finds its counterpart by binary search instead of a scan.
        const auto find_mapping = [&](const Mapping& wanted) {
            const auto it = std::lower_bound(mappings.begin(), mappings.end(), wanted.gpu_base,
                [](const Mapping& mapping, std::uint64_t base) { return mapping.gpu_base < base; });
            return it != mappings.end() && same_mapping(*it, wanted);
        };
        const auto find_entry = [&](const Mapping& wanted, std::size_t count) {
            const auto end = backend.entries.begin() + std::ptrdiff_t(count);
            const auto it = std::lower_bound(backend.entries.begin(), end, wanted.gpu_base,
                [](const auto& entry, std::uint64_t base) { return entry.mapping.gpu_base < base; });
            return it != end && same_mapping(it->mapping, wanted);
        };
        std::vector<MappingId> removed;
        for (const auto& entry : backend.entries)
            if (!find_mapping(entry.mapping)) removed.push_back(entry.id);
        // Direct callers can replace a CPU allocation without a driver callback.
        // A successful synchronous draw is already coherent; never write an old
        // dirty allocation through a new caller's CPU capability at the same VA.
        // A sweep over guest ranges pairs each allocation with the mappings
        // overlapping it.
        std::vector<bool> reused(backend.allocations.size());
        {
            struct Span { std::uint64_t first, last; const std::shared_ptr<void>* owner; std::size_t allocation; };
            constexpr auto mapping_span = std::numeric_limits<std::size_t>::max();
            std::vector<Span> spans;
            spans.reserve(mappings.size() + backend.allocations.size());
            for (const auto& mapping : mappings)
                spans.push_back({mapping.guest_address, mapping.guest_address + mapping.backing_size,
                                 &mapping.owner, mapping_span});
            for (std::size_t i = 0; i != backend.allocations.size(); ++i) {
                const auto& old = backend.allocations[i].mapping;
                spans.push_back({old.guest_address, old.guest_address + old.backing_size, &old.owner, i});
            }
            std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.first < b.first; });
            std::vector<const Span*> active;
            for (const auto& span : spans) {
                if (span.last <= span.first) continue;
                std::erase_if(active, [&](const Span* open) { return open->last <= span.first; });
                for (const auto* open : active) {
                    if ((open->allocation == mapping_span) == (span.allocation == mapping_span)) continue;
                    if (span.first >= open->last || same_owner(*open->owner, *span.owner)) continue;
                    reused[open->allocation == mapping_span ? span.allocation : open->allocation] = true;
                }
                active.push_back(&span);
            }
        }
        // Work already encoded against removed or replaced mappings must
        // retire before ownership mutates; added mappings leave it valid.
        if (!removed.empty() || std::find(reused.begin(), reused.end(), true) != reused.end())
            check(finish_batch(true, FlushReason::Mapping));
        std::vector<BackingId> replaced;
        for (std::size_t i = 0; i != backend.allocations.size(); ++i) {
            if (!reused[i]) continue;
            const auto& allocation = backend.allocations[i];
            for (const auto& dirty : take(resources->memory->inspect({allocation.backing, 0,
                                                                      allocation.mapping.backing_size})))
                if (dirty.gpu_dirty)
                    throw failure("Dirty backing replacement requires prior driver notification", CommandStatus::Stopped);
            replaced.push_back(allocation.backing);
        }
        if (!removed.empty() || !replaced.empty()) {
            check(resources->memory->invalidate_mappings(removed, replaced));
            const auto order = [](const MappingId& a, const MappingId& b) {
                return std::tie(a.address_space, a.mapping, a.generation) <
                       std::tie(b.address_space, b.mapping, b.generation);
            };
            std::sort(removed.begin(), removed.end(), order);
            const auto is_removed = [&](const MappingId& id) {
                return std::binary_search(removed.begin(), removed.end(), id, order);
            };
            const auto is_replaced = [&](const BackingId& backing) {
                return std::find(replaced.begin(), replaced.end(), backing) != replaced.end();
            };
            std::erase_if(backend.entries, [&](const auto& entry) {
                return is_removed(entry.id) || (!replaced.empty() && is_replaced(entry.backing));
            });
            std::erase_if(backend.allocations, [&](const auto& allocation) {
                return !replaced.empty() && is_replaced(allocation.backing);
            });
        }
        backend.entries.reserve(backend.entries.size() + mappings.size());
        backend.allocations.reserve(backend.allocations.size() + mappings.size());
        // Allocations by guest range, for aliasing new mappings to them.
        std::multimap<std::pair<std::uint64_t, std::uint64_t>, std::size_t> by_range;
        for (std::size_t i = 0; i != backend.allocations.size(); ++i)
            by_range.emplace(std::pair{backend.allocations[i].mapping.guest_address,
                                       backend.allocations[i].mapping.backing_size}, i);
        // New mappings register as one batch: the registry then rebuilds its
        // GPU view once instead of once per mapping.
        std::vector<graphics::MappingRegistration> registrations;
        const auto known = backend.entries.size();
        for (const auto& mapping : mappings) {
            if (find_entry(mapping, known)) continue;
            BackingId backing{};
            bool aliased = false;
            const auto [first, last] = by_range.equal_range({mapping.guest_address, mapping.backing_size});
            for (auto it = first; it != last; ++it) {
                if (!same_owner(mapping.owner, backend.allocations[it->second].mapping.owner)) continue;
                backing = backend.allocations[it->second].backing;
                aliased = true;
                break;
            }
            if (!aliased) {
                if (backend.next_backing == std::numeric_limits<std::uint64_t>::max())
                    throw failure("Engine renderer backing identity exhausted", CommandStatus::BudgetExceeded);
                backing = {domain, ++backend.next_backing, 1};
                check(resources->memory->register_backing({backing, mapping.backing_size,
                    std::make_shared<AllocationPin>(mapping.owner)}));
                backend.allocations.push_back({mapping, backing});
                by_range.emplace(std::pair{mapping.guest_address, mapping.backing_size},
                                 backend.allocations.size() - 1);
            }
            if (backend.next_mapping == std::numeric_limits<std::uint64_t>::max())
                throw failure("Engine renderer mapping identity exhausted", CommandStatus::BudgetExceeded);
            const MappingId id{domain, ++backend.next_mapping, 1};
            registrations.push_back({id, backing, mapping});
            backend.entries.push_back({mapping, backing, id});
        }
        if (!registrations.empty()) check(resources->memory->register_mappings(registrations));
        std::sort(backend.entries.begin(), backend.entries.end(), [](const auto& a, const auto& b) {
            return a.mapping.gpu_base < b.mapping.gpu_base;
        });
        backend.forget_views();
        backend.remember(memory.generation());
    }
    void create_resources() {
        if (domain == std::numeric_limits<std::uint64_t>::max())
            throw failure("Engine renderer memory domain exhausted", CommandStatus::BudgetExceeded);
        auto next = std::make_shared<Resources>();
        next->backend = std::make_shared<ScopedMemory>();
        next->backend->timeout = options.completion_timeout_ns;
        graphics::ResourceMemoryOptions memory_options;
        memory_options.address_space = ++domain;
        memory_options.backend = next->backend;
        memory_options.cpu_write_tracking = options.cpu_write_tracking;
        memory_options.host_addressable_backings = options.guest_memory_host_addressable;
        next->memory = take(make_resource_memory(std::move(memory_options)));
        const auto texture_budget = options.resident_bytes / 2;
        const auto buffer_budget = options.resident_bytes / 4;
        auto textures = take(create_texture_cache(device, next->memory, texture_budget));
        next->textures = textures;
        auto buffers = take(create_buffer_cache(device, next->memory, buffer_budget));
        auto bindings = take(create_binding_planner(pipelines, device));
        // Deferred write-back keeps one Metal command buffer open across the
        // configured batch; otherwise the queue commits and retires per draw.
        const std::uint32_t commands = options.defer_gpu_writeback ? options.max_batch_draws : 1;
        const std::uint32_t inflight = options.defer_gpu_writeback ? 8u : 1u;
        next->submissions = test_control
            ? take(create_submission_queue_for_testing(device, next->memory, commands, inflight, test_control))
            : take(create_submission_queue(device, next->memory, commands, inflight));
        // With deferred write-back, draw completions leave GPU writes
        // GPU-authoritative; the front end requests coherence() before every
        // guest-visible effect.
        next->renderer = take(create_renderer({options.resident_bytes, options.staging_bytes, inflight, commands,
                                               options.defer_gpu_writeback},
            {device, next->memory, stages, textures, buffers, pipelines, bindings, next->submissions}));
        // The threaded backend performs every front-end store itself and
        // reports it (see the PendingWrite and effect paths).
        if (threaded) next->renderer->set_engine_stores_reported(true);
        resources = std::move(next);
        sequence = previous_buffers = previous_passes = 0;
        if (options.statistics) ++options.statistics->memory_domains;
    }
};
struct EngineBackendOwner {
    std::shared_ptr<EngineBackend> backend;
    ~EngineBackendOwner() { backend->release_lifecycle(); }
};
} // namespace

RendererResult<RenderCallbacks> create_engine_backend(MetalEngineOptions options,
    EnginePipelineFactory pipelines, std::shared_ptr<SubmissionTestControl> test_control) {
    @autoreleasepool {
        try {
            if (options.mode == MetalEngineMode::Environment) {
                const auto* value = std::getenv("ARMIGO_METAL_RENDERER");
                const std::string_view selected = value ? value : "legacy";
                if (selected == "legacy") options.mode = MetalEngineMode::Legacy;
                else if (selected == "persistent") options.mode = MetalEngineMode::Persistent;
                else return failure("ARMIGO_METAL_RENDERER must be legacy or persistent");
            }
            if (options.mode != MetalEngineMode::Legacy && options.mode != MetalEngineMode::Persistent)
                return failure("Engine renderer mode is invalid");
            if (options.lookup_context && !options.lookup_owner)
                return failure("Engine renderer lookup context requires an owner");
            if ((options.resident_bytes != 0 && options.resident_bytes < 4) || !options.staging_bytes ||
                !options.completion_timeout_ns || options.completion_timeout_ns > 300'000'000'000ull ||
                !options.max_batch_draws)
                return failure("Engine renderer limits are invalid");
            auto context = std::make_shared<EngineBackend>();
            context->options = std::move(options);
            // Legacy execution has no device here; it keeps the floor.
            if (context->options.resident_bytes == 0 && context->options.mode != MetalEngineMode::Persistent)
                context->options.resident_bytes = 512ull << 20;
            context->pipeline_factory = std::move(pipelines);
            context->test_control = std::move(test_control);
            context->compute_context = {context->options.compute_lookup,
                                        context->options.lookup_context,
                                        &context->compute_statistics,
                                        context->options.completion_timeout_ns};
            if (context->options.mode == MetalEngineMode::Persistent) {
                if (!context->pipeline_factory) return failure("Engine renderer pipeline factory is missing");
                context->device = take(create_device_context());
                if (context->options.resident_bytes == 0)
                    context->options.resident_bytes = std::max<std::uint64_t>(512ull << 20,
                        context->device->device().recommendedMaxWorkingSetSize / 2);
                context->pipelines = take(context->pipeline_factory(context->device, context->options.resident_bytes / 4));
                context->stages = take(create_engine_stage_provider(context->options.lookup,
                    context->options.lookup_context, context->options.compute_lookup,
                    context->options.immutable_stages));
                if (context->options.statistics) ++context->options.statistics->device_creations;
            }
            // Legacy execution retains no backing state between callbacks.
            const bool persistent = context->options.mode == MetalEngineMode::Persistent;
            // Without native write tracking, the registry requires synchronous
            // guest access: never run background write-back in that fallback.
            const bool ordered = persistent && context->options.defer_gpu_writeback &&
                                 context->options.guest_memory_host_addressable &&
                                 context->options.cpu_write_tracking;
            std::shared_ptr<void> callback_owner = context;
            if (ordered) {
                auto lifecycle = std::make_shared<EngineBackendOwner>();
                lifecycle->backend = context;
                context->start_pump(context);
                callback_owner = std::move(lifecycle);
            }
            return RenderCallbacks{EngineBackend::clear, context.get(), EngineBackend::draw,
                std::move(callback_owner),
                persistent ? EngineBackend::memory_changed : nullptr,
                persistent && context->options.defer_gpu_writeback ? EngineBackend::coherence : nullptr,
                EngineBackend::dispatch,
                persistent ? EngineBackend::blit : nullptr,
                persistent && context->options.defer_gpu_writeback && context->options.guest_memory_host_addressable
                    ? EngineBackend::resolve_guest : nullptr,
                ordered ? EngineBackend::ordered_effect : nullptr,
                persistent && context->options.defer_gpu_writeback ? EngineBackend::read_coherence : nullptr,
                ordered ? EngineBackend::status : nullptr,
                persistent && context->options.guest_memory_host_addressable && context->options.cpu_write_tracking
                    ? EngineBackend::ordered_write : nullptr};
        } catch (const RendererError& error) { return error;
        } catch (const std::bad_alloc&) {
            return failure("Engine renderer construction allocation failed", CommandStatus::BudgetExceeded);
        } catch (...) { return failure("Engine renderer construction threw an exception"); }
    }
}
} // namespace armigo::platform::metal

namespace armigo::platform {
graphics::RendererResult<graphics::RenderCallbacks> create_metal_engine_backend(MetalEngineOptions options) {
    return metal::create_engine_backend(std::move(options), metal::create_pipeline_state_cache);
}
} // namespace armigo::platform
