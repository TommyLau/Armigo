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
#include "metal_renderer.h"
#include "device_context.h"
#include "draw_encoding.h"
#include "attachments.h"

#include <armigo/graphics/resource_memory_registry.h>
#include <armigo/memory/write_tracker.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <string_view>
#include <string>
#include <cstdlib>
#include <limits>
#include <map>
#include <mach/mach_time.h>
#include <new>
#include <tuple>

namespace armigo::platform::metal {
namespace {
using namespace graphics;

// ARMIGO_TRACE_ENGINE_FLUSH: tallies the preparation failures that make
// enqueue complete prior work and retry, by kind and diagnostic.
void trace_retry(const char* kind, std::string_view detail) {
    static const bool enabled = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
    if (!enabled) return;
    static std::map<std::string, std::uint64_t, std::less<>> counts;
    static std::uint64_t total = 0;
    // Group by the diagnostic's text before any parenthesized per-call detail.
    const auto stem = detail.substr(0, std::min<std::size_t>(detail.find(" ("), 120));
    ++counts[std::string(kind) + ": " + std::string(stem)];
    if (++total % 500) return;
    armigo::diag::opt_in( "engine_retry total=%llu\n", (unsigned long long)total);
    for (const auto& [text, count] : counts)
        armigo::diag::opt_in( "engine_retry_reason count=%llu %s\n", (unsigned long long)count, text.c_str());
}

RendererError error(CommandStatus status, const char* detail) {
    return RendererError({status, detail});
}

bool overlaps(BackingRange a, BackingRange b) {
    return a.backing == b.backing && a.offset < b.offset + b.size &&
           b.offset < a.offset + a.size;
}

bool covers(BackingRange outer, BackingRange inner) {
    return outer.backing == inner.backing && inner.offset >= outer.offset &&
           inner.offset - outer.offset <= outer.size &&
           inner.size <= outer.size - (inner.offset - outer.offset);
}

// YControl bit 0 negates guest Y.
// Stages that read S2R SR_YDIRECTION observe this state through their
// armigo_y_negate function constant, so it selects a pipeline variant.
constexpr std::size_t y_control_method = 0x4eb;
constexpr std::uint32_t y_control_negate_y = 1u << 0;
// y-direction-v1: the function-constant index the translated stage declares.
constexpr std::uint32_t y_direction_constant_index = 9;
// Point state (method = byte offset / 4): the fixed 3D
// point size (0x1518) and PointCoordReplace (0x1604), whose bit 2 selects an
// upper-left sprite origin.
constexpr std::size_t point_size_method = 0x546;
constexpr std::size_t point_coord_replace_method = 0x581;
constexpr std::uint32_t point_coord_upper_left = 4;
// Function-constant index of the lower-left point-coordinate origin (fragment).
constexpr std::uint32_t point_coord_origin_constant_index = 11;

// Split aliasing reads at their backing-relative endpoints, then retain one
// prepared authority per interval. L2 leases identify exact ranges, so clipped
// intervals acquire fresh pins through a matching live GPU mapping.
// The vertex fetch part of a pipeline key: the draw plan's layout, legacy
// attribute slots renumbered as the stage declares them, in location order.
RendererStatus vertex_layout_key(PipelineKey& key, const DrawPlan& draw, const CompiledGraphicsStage& vertex) {
    key.attributes.clear();
    key.buffers.clear();
    auto layout = populate_vertex_layout(key, draw);
    if (std::holds_alternative<RendererError>(layout)) return layout;
    for (auto& attribute : key.attributes) {
        const auto legacy = std::find_if(vertex.vertex_attributes.begin(), vertex.vertex_attributes.end(),
            [&](const auto& input) { return input.generic_slot == attribute.location; });
        if (legacy != vertex.vertex_attributes.end()) attribute.location = legacy->attribute_index;
    }
    std::sort(key.attributes.begin(), key.attributes.end(),
              [](const auto& a, const auto& b) { return a.location < b.location; });
    return renderer_success();
}

RendererStatus normalize_accesses(PreparedRender& prepared) {
    using Identity = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;
    std::map<Identity, std::vector<std::uint64_t>> endpoints;
    for (const auto& access : prepared.accesses) {
        const auto& range = access.range.bytes;
        auto& points = endpoints[{range.backing.owner, range.backing.allocation,
                                  range.backing.generation}];
        points.push_back(range.offset);
        points.push_back(range.offset + range.size);
    }
    std::vector<GpuAccessRequest> accesses;
    std::vector<RetainedResource> storage;
    for (auto& [identity, points] : endpoints) {
        std::sort(points.begin(), points.end());
        points.erase(std::unique(points.begin(), points.end()), points.end());
        const BackingId backing{std::get<0>(identity), std::get<1>(identity), std::get<2>(identity)};
        for (std::size_t point = 1; point != points.size(); ++point) {
            const BackingRange interval{backing, points[point - 1], points[point] - points[point - 1]};
            for (std::size_t index = 0; index != prepared.accesses.size(); ++index) {
                auto access = prepared.accesses[index];
                if (!covers(access.range.bytes, interval)) continue;
                if (access.range.bytes != interval) {
                    bool retained = false;
                    if (auto* registry = dynamic_cast<ResourceMemoryRegistry*>(prepared.request.memory.get())) {
                        if (auto range = registry->retain_backing(interval, access.access)) {
                            access.range = std::move(*range);
                            retained = true;
                        }
                    }
                    for (const auto& mapping : prepared.request.memory->address_space().mappings()) {
                        if (retained) break;
                        if (interval.offset < mapping.buffer_offset) continue;
                        const auto offset = interval.offset - mapping.buffer_offset;
                        if (offset >= mapping.mapped_size || interval.size > mapping.mapped_size - offset) continue;
                        auto candidate = prepared.request.memory->retain(mapping.gpu_base + offset,
                            interval.size, access.access);
                        const auto* range = std::get_if<RetainedRange>(&candidate);
                        if (!range || range->bytes != interval) continue;
                        access.range = *range;
                        retained = true;
                        break;
                    }
                    if (!retained)
                        return error(CommandStatus::MemoryFault, "Renderer normalized access has no retained mapping");
                }
                accesses.push_back(std::move(access));
                storage.push_back(prepared.access_storage[index]);
                break;
            }
        }
    }
    prepared.accesses = std::move(accesses);
    prepared.access_storage = std::move(storage);
    return renderer_success();
}

// Boundary-mode coherence belongs to the coordinator, before any guest effect.
// Retaining the registry here also owns the borrowed address-space callbacks.
class BoundaryActions final : public CompletionActions {
public:
    BoundaryActions(std::shared_ptr<ResourceMemoryRegistry> memory,
                    std::shared_ptr<CompletionActions> actions, bool defer_gpu_writeback)
        : memory_(std::move(memory)), actions_(std::move(actions)),
          defer_gpu_writeback_(defer_gpu_writeback) {}
    RendererStatus make_coherent() override {
        // Registry-wide boundary work belongs to the coordinator's observable
        // boundary, outside a single receipt's completion: one submission's
        // retirement must neither wait on nor fail for an independent batch.
        // Deferred write-back keeps GPU results GPU-authoritative until then;
        // the per-draw path retires synchronously, with no other batch out.
        if (!defer_gpu_writeback_) {
            auto result = memory_->submission_boundary();
            if (std::holds_alternative<RendererError>(result)) return result;
        }
        return actions_->make_coherent();
    }
    RendererStatus prepare_publication() override { return actions_->prepare_publication(); }
    void publish() noexcept override { actions_->publish(); }
    RendererStatus preserve_failed_authority() override {
        return actions_->preserve_failed_authority();
    }
private:
    std::shared_ptr<ResourceMemoryRegistry> memory_;
    std::shared_ptr<CompletionActions> actions_;
    bool defer_gpu_writeback_ = false;
};

class RendererImpl final : public MetalRenderer {
public:
    RendererImpl(RendererLimits limits, RendererServices services,
                 std::shared_ptr<ResourceMemoryRegistry> registry)
        : limits_(limits), services_(std::move(services)), registry_(std::move(registry)) {}

    // Telemetry clock: mach_absolute_time reads the counter directly, far
    // cheaper at per-draw rates than steady_clock (mach_continuous_time).
    static std::uint64_t telemetry_ns() noexcept {
        static const double scale = [] {
            mach_timebase_info_data_t info{};
            mach_timebase_info(&info);
            return info.denom ? double(info.numer) / double(info.denom) : 1.0;
        }();
        return std::uint64_t(double(mach_absolute_time()) * scale);
    }
    RendererResult<SubmissionReceipt> enqueue(RenderRequest request) override {
        static const bool timing = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        if (!timing) return enqueue_snapshot(std::move(request), {});
        // Renderer time per draw by path: repeated (draw parameters into the
        // open packet) or fully prepared, printed as average microseconds.
        const bool draw = request.snapshot && request.snapshot->kind == RenderKind::Draw;
        repeated_ = false;
        blocking_ = 0;
        phase_ns_ = {};
        const auto started = telemetry_ns();
        auto result = enqueue_snapshot(std::move(request), {});
        const auto ns = telemetry_ns() - started;
        if (draw) {
            static std::array<std::uint64_t, 4> totals{};   // repeat ns, repeat n, full ns, full n
            static std::array<std::uint64_t, 10> full_phases{};
            totals[repeated_ ? 0 : 2] += ns;
            ++totals[repeated_ ? 1 : 3];
            // Full-draw time by the dirty groups that blocked the repeat.
            static std::array<std::pair<std::uint64_t, std::uint64_t>, threed_dirty::bit(threed_dirty::Count)> by_mask{};
            if (!repeated_ && blocking_) {
                by_mask[blocking_].first += ns;
                ++by_mask[blocking_].second;
            }
            if ((totals[1] + totals[3]) % 500000 == 0) {
                std::vector<std::pair<std::uint64_t, std::size_t>> top;
                for (std::size_t mask = 0; mask < by_mask.size(); ++mask)
                    if (by_mask[mask].second) top.emplace_back(by_mask[mask].first, mask);
                std::sort(top.rbegin(), top.rend());
                armigo::diag::opt_in( "full_by_mask");
                for (std::size_t i = 0; i < std::min<std::size_t>(top.size(), 12); ++i)
                    armigo::diag::opt_in( " %#zx=%llux%.1fus", top[i].second,
                                 (unsigned long long)by_mask[top[i].second].second,
                                 double(top[i].first) / double(by_mask[top[i].second].second) / 1000.0);
                armigo::diag::newline();
            }
            if (!repeated_)
                for (std::size_t i = 0; i < full_phases.size(); ++i) full_phases[i] += phase_ns_[i];
            if ((totals[1] + totals[3]) % 50000 == 0) {
                const auto per_full = [&](std::uint64_t value) {
                    return totals[3] ? double(value) / double(totals[3]) / 1000.0 : 0.0;
                };
                armigo::diag::opt_in( "draw_time repeat_us=%.2f repeat_n=%llu full_us=%.2f full_n=%llu "
                             "full_boundary_us=%.2f full_try_us=%.2f full_prepare_us=%.2f full_append_us=%.2f\n",
                             totals[1] ? double(totals[0]) / double(totals[1]) / 1000.0 : 0.0,
                             (unsigned long long)totals[1], per_full(totals[2]), (unsigned long long)totals[3],
                             per_full(full_phases[0]), per_full(full_phases[1]), per_full(full_phases[2]),
                             per_full(full_phases[3]));
                armigo::diag::opt_in( "prepare_phases plan_us=%.2f stages_us=%.2f assemble_us=%.2f bindings_us=%.2f "
                             "pipeline_us=%.2f rest_us=%.2f epoch_overflows=%llu\n", per_full(full_phases[4]),
                             per_full(full_phases[5]), per_full(full_phases[6]), per_full(full_phases[7]),
                             per_full(full_phases[8]), per_full(full_phases[9]),
                             (unsigned long long)epoch_log_.overflows);
                const auto tracker = memory::WriteTracker::stats();
                armigo::diag::opt_in( "tracker_activity faults=%llu prepared=%llu protect_calls=%llu draws=%llu\n",
                             (unsigned long long)tracker.faults, (unsigned long long)tracker.prepared,
                             (unsigned long long)tracker.protect_calls,
                             (unsigned long long)(totals[1] + totals[3]));
            }
        }
        return result;
    }
    bool repeated_ = false;
    std::uint64_t blocking_ = 0;
    // Telemetry phases of the enqueue in progress (ns): boundary flushes and
    // throttling, the repeat attempt, preparation, append.
    // Then preparation's own phases: target/state planning, stages, draw
    // assembly, bindings, pipeline, the rest.
    std::array<std::uint64_t, 10> phase_ns_{};
    struct PhaseTimer {
        std::uint64_t* slot;
        std::uint64_t started = slot ? telemetry_ns() : 0;
        ~PhaseTimer() {
            if (slot) *slot += telemetry_ns() - started;
        }
    };
    std::uint64_t* phase(std::size_t index) {
        static const bool timing = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        return timing ? &phase_ns_[index] : nullptr;
    }

    RendererResult<SubmissionReceipt> enqueue_owned(
        RenderRequest request, std::unique_ptr<const RenderStop> snapshot) override {
        request.snapshot.reset();
        return enqueue_snapshot(std::move(request), std::move(snapshot));
    }

private:
    RendererResult<SubmissionReceipt> enqueue_snapshot(
        RenderRequest request, std::unique_ptr<const RenderStop> owned_snapshot) {
        if ((!request.snapshot && !owned_snapshot) || !request.completion || request.memory != services_.memory ||
            !request.id.queue || !request.id.sequence) {
            return error(CommandStatus::Unsupported, "Renderer request has no owned snapshot, actions or matching memory domain");
        }
        if (queue_ && (request.id.queue != queue_ || request.id.sequence <= sequence_)) {
            return error(CommandStatus::Unsupported, "Renderer submission identifiers are not strictly ordered");
        }
        for (const auto& dependency : request.dependencies) {
            if (!dependency.valid()) return error(CommandStatus::Unsupported, "Renderer dependency is invalid");
            const auto state = dependency.snapshot();
            if (state.error) return *state.error;
            if (state.id.queue == request.id.queue && state.id.sequence >= request.id.sequence)
                return error(CommandStatus::Unsupported, "Renderer dependency points to this or a later request");
        }
        try {
            prune();
            if (pending_.size() == pending_.capacity())
                pending_.reserve(std::max(pending_.size() + 1, 2 * pending_.capacity()));
            // A lent snapshot is read only during this call's preparation;
            // queued packets carry a placeholder instead (see below).
            if (owned_snapshot) request.snapshot = std::move(owned_snapshot);
            if (accepted_since_flush_ >= limits_.commands_per_submission) {
                PhaseTimer timer{phase(0)};
                auto boundary = flush(SubmissionBoundary::ResourceBudget);
                if (const auto* failure = std::get_if<RendererError>(&boundary)) return *failure;
            }
            if (flushed_.size() >= limits_.inflight_submissions) {
                // Wait for the oldest
                // submission only; later ones keep the GPU busy.
                PhaseTimer timer{phase(0)};
                auto retired = flush(SubmissionBoundary::ResourceBudget);
                if (!std::holds_alternative<RendererError>(retired)) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                    retired = services_.submissions->retire_below(limits_.inflight_submissions,
                        std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.time_since_epoch()).count());
                }
                if (const auto* failure = std::get_if<RendererError>(&retired)) return *failure;
                prune();
            }
            // Every callback entry observes guest CPU bytes as of that entry.
            // The deferred epoch advance is valid while earlier batched work
            // is pending: it neither waits on nor resolves GPU work, so
            // enqueued draws keep the immutable copies they captured while
            // this entry's caches revalidate CPU-backed inputs and create
            // replacement uploads. GPU-dirty intervals stay GPU-authoritative.
            // Native execution may have changed CPU bytes since the last empty
            // synchronous queue. The boundary invalidates copied versions
            // without inventing CPU-write generations.
            if (limits_.defer_gpu_writeback) {
                auto boundary = registry_->advance_cpu_epoch();
                if (const auto* failure = std::get_if<RendererError>(&boundary)) return *failure;
            } else if (pending_.empty()) {
                auto boundary = registry_->submission_boundary();
                if (const auto* failure = std::get_if<RendererError>(&boundary)) return *failure;
            }
            if (limits_.defer_gpu_writeback && request.snapshot->kind == RenderKind::Draw) {
                std::optional<SubmissionReceipt> repeated;
                {
                    PhaseTimer timer{phase(1)};
                    repeated = try_repeat(request);
                }
                if (repeated) return *repeated;
            }
            request.completion = std::make_shared<BoundaryActions>(registry_, request.completion,
                                                                   limits_.defer_gpu_writeback);
            bool completed = false, resolved = false;
            for (unsigned attempt = 0; attempt != 3; ++attempt) {
                auto prepared = [&] {
                    PhaseTimer timer{phase(2)};
                    return prepare(request);
                }();
                if (auto* failure = std::get_if<RendererError>(&prepared)) {
                    count_unknown(0);
                    bump_epoch(true);
                    last_draw_.reset();
                    reuse_.attachments.reset();
                    services_.bindings->drop_reuse();
                    if (registry_->has_deferred_host_store())
                        return std::get<RendererError>(commit_hold_wait());
                    const auto status = failure->command().status;
                    // Borrowed GpuAddressSpace callbacks can report only a bool;
                    // a refused pending-producer read arrives as MemoryFault.
                    // Retry after completing prior work, never this packet. A
                    // producer still in the open batch needs only completion:
                    // the retried preparation resolves the bytes it reads.
                    // Deferred writeback can also leave GPU-newer bytes that
                    // an incompatible alias reads; a second retry resolves
                    // every GPU result first.
                    const bool retryable = (!pending_.empty() || limits_.defer_gpu_writeback) &&
                        (status == CommandStatus::Stopped || status == CommandStatus::MemoryFault ||
                         status == CommandStatus::BudgetExceeded);
                    if (retryable && (!completed || (!resolved && limits_.defer_gpu_writeback))) {
                        const bool resolve = completed;
                        trace_retry(resolve ? "prepare-resolve" : "prepare", failure->command().detail);
                        auto split = complete_prior(resolve);
                        if (const auto* failed = std::get_if<RendererError>(&split)) return *failed;
                        (resolve ? resolved : completed) = true;
                        continue;
                    }
                    return *failure;
                }
                static const auto prepared_snapshot = std::make_shared<const RenderStop>();
                std::get<PreparedRender>(prepared).request.snapshot = prepared_snapshot;
                auto receipt = [&] {
                    PhaseTimer timer{phase(3)};
                    return services_.submissions->append(std::get<PreparedRender>(std::move(prepared)));
                }();
                if (auto* accepted = std::get_if<SubmissionReceipt>(&receipt)) {
                    queue_ = request.id.queue;
                    sequence_ = request.id.sequence;
                    pending_.push_back(accepted->completion);
                    ++accepted_since_flush_;
                    reuse_.last_packet = accepted->completion;
                    return receipt;
                }
                count_unknown(1);
                bump_epoch(true);
                if (registry_->has_deferred_host_store())
                    return std::get<RendererError>(commit_hold_wait());
                const auto& failure = std::get<RendererError>(receipt);
                if (attempt == 0 && !pending_.empty() &&
                    (failure.command().status == CommandStatus::Stopped ||
                     failure.command().status == CommandStatus::BudgetExceeded)) {
                    trace_retry("append", failure.command().detail);
                    auto split = complete_prior();
                    if (const auto* failed = std::get_if<RendererError>(&split)) return *failed;
                    continue;
                }
                return failure;
            }
        } catch (const std::bad_alloc&) {
            return error(CommandStatus::BudgetExceeded, "Renderer preparation allocation failed");
        }
        return error(CommandStatus::Stopped, "Renderer preparation could not retry");
    }

public:
    RendererStatus flush(SubmissionBoundary boundary) override {
        try {
            if (accepted_since_flush_) flushed_.reserve(flushed_.size() + 1);
        } catch (const std::bad_alloc&) {
            return error(CommandStatus::BudgetExceeded, "Renderer submission tracking allocation failed");
        }
        auto result = services_.submissions->flush(boundary);
        if (!std::holds_alternative<RendererError>(result) && accepted_since_flush_) {
            flushed_.push_back(pending_.back());
            accepted_since_flush_ = 0;
        }
        return result;
    }
    void note_engine_store(std::uint64_t gpu_address, std::uint64_t size) override {
        services_.bindings->note_engine_store(gpu_address, size);
    }
    void set_engine_stores_reported(bool reported) override {
        services_.bindings->set_engine_stores_reported(reported);
    }
    RendererStatus poll() override {
        auto result = services_.submissions->poll();
        prune();
        return result;
    }
    RendererStatus drain(std::uint64_t deadline) override {
        // Even a poisoned flush must hand already accepted work to the
        // submission queue for failed-authority retirement.
        static_cast<void>(flush(SubmissionBoundary::Shutdown));
        auto result = services_.submissions->drain(deadline);
        if (std::holds_alternative<std::monostate>(result)) accepted_since_flush_ = 0;
        prune();
        return result;
    }
    RendererStatus set_commit_hold(bool held) override {
        if (held && !limits_.defer_gpu_writeback)
            return error(CommandStatus::Unsupported,
                         "Renderer commit hold requires deferred GPU writeback");
        return services_.submissions->set_commit_hold(held);
    }
    RendererStatus abort_commit_hold(CommandResult error) override {
        return services_.submissions->abort_commit_hold(error);
    }
    bool commit_held() const noexcept override {
        return services_.submissions->commit_held();
    }
    bool can_accept_while_held() const noexcept override {
        return services_.submissions->can_accept_while_held();
    }

private:
    // Trims the completed prefix of each list (submissions complete in queue
    // order) and sweeps a whole list only when it has doubled since its last
    // sweep, so a large batch is not rescanned on every enqueue.
    void prune() {
        const auto complete = [](const CompletionToken& token) {
            return token.snapshot().guest_visible_complete();
        };
        const auto trim = [&](std::vector<CompletionToken>& tokens, std::size_t& swept) {
            std::size_t settled = 0;
            while (settled < tokens.size() && complete(tokens[settled])) ++settled;
            tokens.erase(tokens.begin(), tokens.begin() + std::ptrdiff_t(settled));
            if (tokens.size() <= std::max<std::size_t>(64, 2 * swept)) return;
            std::erase_if(tokens, complete);
            swept = tokens.size();
        };
        trim(pending_, swept_pending_);
        trim(flushed_, swept_flushed_);
    }
    std::size_t swept_pending_ = 0, swept_flushed_ = 0;
    RendererStatus complete_prior(bool resolve = false) {
        auto result = flush(SubmissionBoundary::ResourceBudget);
        if (std::holds_alternative<RendererError>(result)) return result;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        result = drain(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.time_since_epoch()).count());
        if (std::holds_alternative<RendererError>(result)) return result;
        // Drops completed L2 readers/authorities as well as submission leases,
        // which is required before L5 can grow an overlapping allocation.
        return limits_.defer_gpu_writeback && !resolve ? registry_->internal_boundary()
                                                       : registry_->submission_boundary();
    }

    RendererStatus attachment(const AttachmentPlan& plan,
                              std::optional<ResidentTexture>& resident,
                              std::optional<TextureViewKey>& identity,
                              PreparedRender& prepared) {
        auto retained = registry_->retain(plan.texture.address, plan.texture.backing_extent,
                                          ResourceAccess::ReadWrite);
        if (const auto* failed = std::get_if<RendererError>(&retained)) {
            const auto command = failed->command();
            char detail[256]{};
            std::snprintf(detail, sizeof detail,
                "Attachment lookup address=%#llx size=%#llx aspect=%u: %.*s",
                static_cast<unsigned long long>(plan.texture.address),
                static_cast<unsigned long long>(plan.texture.backing_extent), unsigned(plan.aspect),
                int(command.detail.size()), command.detail.data());
            return error(command.status, detail);
        }
        const auto range = std::get<RetainedRange>(std::move(retained));
        TextureViewKey key;
        key.storage = {range.bytes, plan.texture};
        key.aspect = plan.aspect;
        key.first_level = plan.level;
        key.first_layer = plan.layer;
        key.layer_count = plan.layer_count;
        auto acquired = services_.textures->acquire({range, key, ResourceAccess::ReadWrite});
        if (const auto* failed = std::get_if<RendererError>(&acquired)) return *failed;
        resident = std::get<ResidentTexture>(std::move(acquired));
        identity = resident->key;
        prepared.request.dependencies.insert(prepared.request.dependencies.end(),
            resident->dependencies.begin(), resident->dependencies.end());
        for (std::size_t i = 0; i != prepared.accesses.size(); ++i) {
            const auto& prior = prepared.accesses[i];
            if (!overlaps(prior.range.bytes, range.bytes)) continue;
            // Packed depth/stencil aspects share one allocation authority.
            // Other writable aliases need an explicit conversion/feedback path.
            if (prior.range.bytes == range.bytes &&
                plan.aspect == TextureAspect::Stencil &&
                prepared.attachments.depth &&
                prepared.attachments.depth->key.storage == key.storage) return renderer_success();
            return error(CommandStatus::Unsupported, "Renderer attachments overlap incompatible storage");
        }
        prepared.accesses.push_back({range, ResourceAccess::ReadWrite});
        prepared.access_storage.push_back(resident->storage);
        return renderer_success();
    }

    // Each binding access takes the resident storage of the binding that
    // covers it. Aliasing between bindings and attachments is not re-validated
    // per draw; the planner's resources already carry their hazards.
    RendererStatus binding_accesses(const Bindings& bindings, PreparedRender& prepared) {
        prepared.accesses.reserve(prepared.accesses.size() + bindings.accesses.size());
        prepared.access_storage.reserve(prepared.access_storage.size() + bindings.accesses.size());
        for (const auto& access : bindings.accesses) {
            const RetainedResource* storage = nullptr;
            const auto buffer = [&](const ResidentBuffer& candidate) {
                if (!storage && covers(candidate.key.bytes, access.range.bytes)) storage = &candidate.storage;
            };
            for (const auto& candidate : bindings.buffers) buffer(candidate.buffer);
            for (const auto& candidate : bindings.vertex_buffers) buffer(candidate.buffer);
            if (bindings.index_buffer) buffer(*bindings.index_buffer);
            for (const auto& candidate : bindings.textures)
                if (!storage && covers(candidate.texture.key.storage.bytes, access.range.bytes))
                    storage = &candidate.texture.storage;
            if (!storage || !*storage)
                return error(CommandStatus::Unsupported, "Renderer binding access has no matching resident storage");
            prepared.accesses.push_back(access);
            prepared.access_storage.push_back(*storage);
        }
        return renderer_success();
    }

    RendererResult<PreparedRender> prepare(const RenderRequest& request) {
        static const bool timing = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        std::uint64_t marked = timing ? telemetry_ns() : 0;
        const auto mark = [&](std::size_t phase) {
            if (!timing) return;
            const auto now = telemetry_ns();
            phase_ns_[4 + phase] += now - marked;
            marked = now;
        };
        struct Rest {
            const decltype(mark)& mark;
            ~Rest() { mark(5); }
        } rest{mark};
        PreparedRender result;
        result.request = request;
        result.prefix_safe_completion =
            limits_.defer_gpu_writeback && request.prefix_safe_completion;
        const auto& stop = *request.snapshot;
        if (stop.kind != RenderKind::Draw) {
            // A clear's targets are logged once planned; other operations
            // may write anywhere.
            if (stop.kind != RenderKind::Clear && stop.kind != RenderKind::Compute)
                count_unknown(4 + std::size_t(stop.kind));
            bump_epoch(stop.kind != RenderKind::Clear && stop.kind != RenderKind::Compute);
            last_draw_.reset();
        }
        if (const auto generation = registry_->address_space().generation(); generation != reuse_.view) {
            reuse_.view = generation;
            bump_epoch(false);
            // Only GPU ranges whose mapping changed may now name other memory:
            // they are logged as remapped in this epoch, and sets that read
            // none of them stay valid.
            const auto mappings = registry_->address_space().mappings();
            const auto same = [](const DrawReuse::ViewMapping& a, const Mapping& b) {
                return a.gpu_base == b.gpu_base && a.mapped_size == b.mapped_size &&
                       a.buffer_offset == b.buffer_offset && a.guest_address == b.guest_address &&
                       a.backing_size == b.backing_size && a.page_size == b.page_size &&
                       !a.owner.owner_before(b.owner) && !b.owner.owner_before(a.owner);
            };
            const auto log = [&](std::uint64_t base, std::uint64_t size) {
                epoch_log_.remap(base, size, reuse_epoch_);
            };
            const auto& previous = reuse_.mappings;
            std::size_t i = 0, j = 0;
            while (i < previous.size() || j < mappings.size()) {
                if (j == mappings.size() || (i < previous.size() && previous[i].gpu_base < mappings[j].gpu_base)) {
                    log(previous[i].gpu_base, previous[i].mapped_size);
                    ++i;
                } else if (i == previous.size() || mappings[j].gpu_base < previous[i].gpu_base) {
                    log(mappings[j].gpu_base, mappings[j].mapped_size);
                    ++j;
                } else {
                    if (!same(previous[i], mappings[j])) {
                        log(previous[i].gpu_base, previous[i].mapped_size);
                        log(mappings[j].gpu_base, mappings[j].mapped_size);
                    }
                    ++i;
                    ++j;
                }
            }
            reuse_.mappings.clear();
            for (const auto& mapping : mappings)
                reuse_.mappings.push_back({mapping.gpu_base, mapping.mapped_size, mapping.buffer_offset,
                                           mapping.guest_address, mapping.backing_size, mapping.page_size,
                                           mapping.owner});
            // Program addresses may now name other memory.
            stage_pairs_.clear();
        }
        std::uint64_t dirty = threed_dirty::all;
        if (stop.kind == RenderKind::Clear) {
            auto planned = plan_render_clear(stop);
            if (const auto* failed = std::get_if<RendererError>(&planned)) return *failed;
            const auto clear = std::get<ClearPlan>(std::move(planned));
            if (clear.no_op) {
                result.operation = PreparedClear{clear};
                return result;
            }
            result.targets.area = {0, 0, clear.target.texture.width, clear.target.texture.height};
            result.targets.samples = clear.target.texture.samples;
            if (clear.color_mask != 0) result.targets.colors[0] = clear.target;
            // A combined clear keeps its depth/stencil descriptor alongside
            // the colour attachment; single-aspect clears carry it in target.
            const graphics::AttachmentPlan& depth_stencil =
                clear.depth_stencil ? *clear.depth_stencil : clear.target;
            if (clear.clear_depth) {
                result.targets.depth = depth_stencil;
                result.targets.depth->aspect = TextureAspect::Depth;
            }
            if (clear.clear_stencil) {
                result.targets.stencil = depth_stencil;
                result.targets.stencil->aspect = TextureAspect::Stencil;
            }
            result.operation = PreparedClear{clear};
        } else if (stop.kind == RenderKind::Draw) {
            auto targets = plan_render_targets(stop);
            if (const auto* failed = std::get_if<RendererError>(&targets)) return *failed;
            result.targets = std::get<RenderTargetPlan>(std::move(targets));
            result.targets.area = render_pass_area(result.targets);
            auto state = plan_render_state(stop, result.targets);
            if (const auto* failed = std::get_if<RendererError>(&state)) return *failed;
            auto planned_targets = result.targets;
            auto planned_state = std::get<RenderState>(state);
            // A colour target whose write mask is zero receives no writes from
            // this draw, so it needs no attachment: it behaves like an unbound
            // logical target (export discarded, stored contents untouched).
            // Guests leave such targets bound across depth-only passes, and
            // they may alias other bound targets.
            for (std::size_t i = 0; i != result.targets.colors.size(); ++i) {
                if (result.targets.colors[i] &&
                    std::get<RenderState>(state).color[i].write_mask == 0)
                    result.targets.colors[i].reset();
            }
            dirty = reuse_dirty(stop);
            trace_dirty(stop, dirty);
            // Draws of one epoch share their attachments, so bindings reused
            // within it never sample bytes another epoch draw rendered.
            std::vector<std::pair<std::uint64_t, std::uint64_t>> attachment_ranges;
            for (const auto* plan : {&result.targets.colors[0], &result.targets.colors[1],
                                     &result.targets.colors[2], &result.targets.colors[3],
                                     &result.targets.colors[4], &result.targets.colors[5],
                                     &result.targets.colors[6], &result.targets.colors[7],
                                     &result.targets.depth, &result.targets.stencil})
                if (*plan) attachment_ranges.emplace_back((*plan)->texture.address, (*plan)->texture.backing_extent);
            if (attachment_ranges != reuse_.attachment_ranges) {
                reuse_.attachment_ranges = attachment_ranges;
                bump_epoch(false);
            }
            mark(0);
            auto admitted_stages = reuse_stages(stop, dirty);
            if (const auto* failed = std::get_if<RendererError>(&admitted_stages)) return *failed;
            const auto& admitted = std::get<AdmittedGraphicsStages>(admitted_stages);
            if (!admitted.stages.vertex || !admitted.stages.fragment)
                return error(CommandStatus::Unsupported, "Renderer stage provider returned missing stages");
            auto inputs = admitted.stages.vertex->vertex_class_inputs;
            for (const auto& legacy : admitted.stages.vertex->vertex_attributes) {
                inputs.push_back({legacy.generic_slot, VertexFormatClass::float_class});
            }
            if (stop.draw) {
                const auto topology = stop.registers[0x652] ? stop.registers[0x65c]
                                                           : (stop.draw->topology & 0xffffu);
                const bool normalized = stop.registers[0x652] ? topology == 0x1015
                    : topology == 2 || (topology >= 6 && topology <= 9);
                if ((stop.draw->indexed || normalized) &&
                    std::uint64_t(stop.draw->count) * (normalized ? 12 : 4) > limits_.staging_bytes)
                    return error(CommandStatus::BudgetExceeded, "Renderer index preparation exceeds the staging budget");
            }
            mark(1);
            auto assembled = assemble_draw(stop, inputs, registry_->address_space(),
                                           stop.inline_indices, true);
            if (const auto* failed = std::get_if<RendererError>(&assembled)) return *failed;
            PreparedDraw draw;
            draw.draw = std::get<DrawPlan>(std::move(assembled));
            auto native = select_native_draw(draw.draw, stop, *registry_);
            if (const auto* failed = std::get_if<RendererError>(&native)) return *failed;
            draw.state = std::get<RenderState>(std::move(state));
            // Native tests must not access absent depth/stencil storage, even
            // when the guest left their test/write enables set.
            if (!result.targets.depth) {
                draw.state.depth_stencil.depth_test = false;
                draw.state.depth_stencil.depth_write = false;
            }
            if (!result.targets.stencil) draw.state.depth_stencil.stencil_test = false;
            if (!result.targets.depth && !result.targets.stencil &&
                std::ranges::none_of(result.targets.colors,
                    [](const auto& color) { return color.has_value(); })) {
                // Surface/viewport state can exceed the native framebuffer.
                // Keep scissors inside the same bounded extent as the pass key.
                for (auto& scissor : draw.state.scissors) {
                    const auto right = std::min<std::uint64_t>(
                        scissor.x + scissor.width, result.targets.area.width);
                    const auto bottom = std::min<std::uint64_t>(
                        scissor.y + scissor.height, result.targets.area.height);
                    scissor.x = std::clamp<std::int64_t>(scissor.x, 0, result.targets.area.width);
                    scissor.y = std::clamp<std::int64_t>(scissor.y, 0, result.targets.area.height);
                    scissor.width = right > std::uint64_t(scissor.x) ? right - scissor.x : 0;
                    scissor.height = bottom > std::uint64_t(scissor.y) ? bottom - scissor.y : 0;
                }
            }
            mark(2);
            auto bindings = services_.bindings->prepare(stop, admitted, draw.draw,
                *registry_, *services_.buffers, *services_.textures,
                BindingReuse{dirty, reuse_epoch_, reuse_.attachment_ranges, &epoch_log_});
            if (const auto* failed = std::get_if<RendererError>(&bindings)) return *failed;
            draw.bindings = std::get<Bindings>(std::move(bindings));
            const bool shader_bias_capable = admitted.stages.fragment->emulates_depth_bias;
            const bool biased = result.targets.depth && draw.state.depth_bias.fill &&
                draw.draw.primitive_class == PrimitiveClass::Triangle &&
                draw.state.rasterizer.front_fill == FillMode::Fill;
            draw.shader_depth_bias = shader_bias_capable && biased &&
                                     draw.state.depth_bias.units != 0;
            if (biased && draw.state.depth_bias.units != 0 && !shader_bias_capable)
                return error(CommandStatus::Unsupported,
                             "constant depth bias requires the shader-v1 depth bias ABI");
            if (draw.shader_depth_bias) {
                const auto& bias = draw.state.depth_bias;
                // Unclamped slope uses rasterizer precision. Only a combined
                // signed cap requires reconstructing slope in the fragment.
                const std::array<float, 4> parameters{
                    bias.constant, bias.clamp != 0.0f ? bias.factor : 0.0f, bias.clamp, 1.0f};
                auto uploaded = services_.buffers->upload(std::as_bytes(std::span(parameters)), 16);
                if (const auto* failed = std::get_if<RendererError>(&uploaded)) return *failed;
                auto uniform = std::get<ResidentBuffer>(std::move(uploaded));
                draw.bindings.dependencies.insert(draw.bindings.dependencies.end(),
                    uniform.dependencies.begin(), uniform.dependencies.end());
                draw.bindings.buffers.push_back(
                    {shader_input::Stage::Fragment, 30, std::move(uniform)});
            }
            std::uint64_t staging = 0;
            const auto charge_upload = [&](const ResidentBuffer& buffer) {
                if (buffer.size > limits_.staging_bytes - staging) return false;
                staging += buffer.size;
                return true;
            };
            for (const auto& buffer : draw.bindings.buffers)
                if ((buffer.use != BufferUse::Storage || !buffer.buffer.key.bytes.backing.owner) &&
                    !charge_upload(buffer.buffer))
                    return error(CommandStatus::BudgetExceeded, "Renderer uniform snapshots exceed the staging budget");
            for (const auto& buffer : draw.bindings.vertex_buffers)
                if (!buffer.buffer.key.bytes.backing.owner && !charge_upload(buffer.buffer))
                    return error(CommandStatus::BudgetExceeded, "Renderer constant vertex data exceeds the staging budget");
            if (draw.bindings.index_buffer && draw.draw.indices && !draw.draw.indices->converted.empty() &&
                !charge_upload(*draw.bindings.index_buffer))
                return error(CommandStatus::BudgetExceeded, "Renderer index snapshot exceeds the staging budget");
            draw.feedback_bytes_limit = limits_.staging_bytes - staging;
            // Submission fills these immutable slices after feedback resolution.
            // Reserve their complete padded allocation before any native allocation.
            for (const auto& binding : draw.bindings.argument_buffers) {
                const auto bytes = std::uint64_t(binding.layout.byte_extent) + binding.layout.alignment - 1;
                if (bytes > limits_.staging_bytes - staging)
                    return error(CommandStatus::BudgetExceeded, "Renderer argument buffers exceed the staging budget");
                staging += bytes;
            }
            // The pipeline and depth-stencil objects follow from the stage pair,
            // vertex input, render target and output state registers plus the
            // draw's primitive class, emulation, depth-bias mode and instancing:
            // with those unchanged the previous draw's objects serve.
            constexpr auto pipeline_inputs = threed_dirty::bit(threed_dirty::Shaders) |
                threed_dirty::bit(threed_dirty::VertexStreams) | threed_dirty::bit(threed_dirty::VertexAttributes) |
                threed_dirty::bit(threed_dirty::RenderTargets) | threed_dirty::bit(threed_dirty::State);
            auto& cached_pipeline = reuse_.pipeline;
            mark(3);
            if (!(dirty & pipeline_inputs) && cached_pipeline &&
                cached_pipeline->vertex == admitted.stages.vertex &&
                cached_pipeline->fragment == admitted.stages.fragment &&
                cached_pipeline->primitive == draw.draw.primitive_class &&
                cached_pipeline->emulation == draw.draw.builtin_emulation &&
                cached_pipeline->depth_bias == draw.shader_depth_bias &&
                cached_pipeline->base_instance == draw.draw.source.base_instance &&
                cached_pipeline->instance_count == draw.draw.source.instance_count) {
                draw.pipeline = cached_pipeline->pipeline;
                draw.depth_stencil = cached_pipeline->depth_stencil;
            } else {
            cached_pipeline.reset();
            PipelineKey key;
            key.context = services_.device->identity();
            key.vertex = admitted.vertex_key;
            key.fragment = admitted.fragment_key;
            // y-direction-v1: a consuming stage's function key gains exactly
            // one bool constant carrying the guest Y sign, so both negate
            // states get their own pipeline while every other stage keeps the
            // admitted key unchanged.
            const bool y_negate = (stop.registers[y_control_method] & y_control_negate_y) != 0;
            const auto carry_y_direction = [&](ShaderFunctionKey& stage_key,
                                               const CompiledGraphicsStage& stage) {
                if (!stage.consumes_y_direction) return;
                stage_key.specialization.push_back(
                    {y_direction_constant_index, "bool", {std::byte(y_negate ? 1 : 0)}});
            };
            carry_y_direction(key.vertex, *admitted.stages.vertex);
            carry_y_direction(key.fragment, *admitted.stages.fragment);
            // PointCoordReplace bit 2 clear selects a
            // lower-left sprite origin. Only point draws specialize it, so other
            // draws keep one pipeline; Metal ignores [[point_coord]] elsewhere.
            const bool point_draw = draw.draw.primitive_class == PrimitiveClass::Point;
            if (point_draw && admitted.stages.vertex->fixed_point_size &&
                stop.registers[point_size_method] != std::bit_cast<std::uint32_t>(1.0f))
                return error(CommandStatus::Unsupported,
                    "point draw with a fixed point size other than 1.0 and no shader point size");
            const bool lower_left_origin =
                point_draw && (stop.registers[point_coord_replace_method] & point_coord_upper_left) == 0;
            if (admitted.stages.fragment->reads_point_coord)
                key.fragment.specialization.push_back(
                    {point_coord_origin_constant_index, "bool", {std::byte(lower_left_origin ? 1 : 0)}});
            key.coverage = draw.state.coverage;
            key.primitive = draw.draw.primitive_class;
            key.rasterization_enabled = draw.state.rasterizer.enabled;
            key.shader_depth_bias = draw.shader_depth_bias;
            key.shader_emulation = draw.draw.builtin_emulation;
            for (std::size_t i = 0; i != key.colors.size(); ++i) {
                if (result.targets.colors[i]) key.colors[i].format = result.targets.colors[i]->texture.format;
                key.colors[i].blend = draw.state.color[i];
            }
            if (result.targets.depth) key.depth_format = result.targets.depth->texture.format;
            if (result.targets.stencil) key.stencil_format = result.targets.stencil->texture.format;
            auto layout = vertex_layout_key(key, draw.draw, *admitted.stages.vertex);
            if (const auto* failed = std::get_if<RendererError>(&layout)) return *failed;
            auto pipeline = services_.pipelines->acquire(key, admitted);
            if (const auto* failed = std::get_if<RendererError>(&pipeline)) return *failed;
            draw.pipeline = std::get<ResidentPipeline>(std::move(pipeline));
            auto depth = services_.pipelines->depth_stencil(draw.state.depth_stencil);
            if (const auto* failed = std::get_if<RendererError>(&depth)) return *failed;
            draw.depth_stencil = std::get<RetainedResource>(std::move(depth));
            cached_pipeline = DrawReuse::Pipeline{admitted.stages.vertex, admitted.stages.fragment,
                draw.draw.primitive_class, draw.draw.builtin_emulation, draw.shader_depth_bias,
                draw.draw.source.base_instance, draw.draw.source.instance_count, draw.pipeline, draw.depth_stencil};
            }
            mark(4);
            last_draw_ = LastDraw{reuse_epoch_, inputs, draw.draw.topology, draw.draw.primitive_class,
                draw.draw.builtin_emulation, draw.draw.indices.has_value(),
                draw.draw.indices ? draw.draw.indices->encoded_format : IndexFormat::U16,
                draw.draw.indices ? draw.draw.indices->address : 0,
                draw.draw.indices ? draw.draw.indices->inclusive_limit : 0,
                draw.draw.indices && !draw.draw.indices->converted.empty(),
                draw.draw.source.base_instance, draw.draw.source.instance_count, draw.shader_depth_bias,
                stream_layout(stop)};
            last_draw_->prefix_safe_completion = result.prefix_safe_completion;
            for (const auto& input : inputs) last_draw_->attributes.push_back(stop.registers[0x458 + input.location]);
            last_draw_->targets = std::move(planned_targets);
            last_draw_->state = std::move(planned_state);
            last_draw_->applied = draw.state;
            last_draw_->vertex = draw.draw.vertex;
            last_draw_->attached = result.targets.depth || result.targets.stencil ||
                std::ranges::any_of(result.targets.colors, [](const auto& color) { return color.has_value(); });
            result.operation = std::move(draw);
            reuse_.stream = stop.state_stream;
            reuse_.serial = stop.state_serial;
        } else if (stop.kind == RenderKind::Compute) {
            // A launch is one more packet of the frame's command
            // stream, so it shares residency, caches and deferred writeback
            // with the draws around it instead of synchronizing the guest.
            if (!stop.compute)
                return error(CommandStatus::Unsupported, "Renderer compute request has no launch");
            auto stage = services_.stages->lookup_compute(stop, registry_->address_space());
            if (const auto* failed = std::get_if<RendererError>(&stage)) return *failed;
            auto admitted = std::get<AdmittedComputeStage>(std::move(stage));
            const auto& launch = *stop.compute;
            if (admitted.stage->compute_local_size != launch.block)
                return error(CommandStatus::Unsupported,
                             "Renderer compute stage local size differs from the launch block");
            PreparedDispatch dispatch;
            dispatch.grid = launch.grid;
            dispatch.block = launch.block;
            auto bindings = services_.bindings->prepare_compute(stop, *admitted.stage,
                *registry_, *services_.buffers, *services_.textures);
            if (const auto* failed = std::get_if<RendererError>(&bindings)) return *failed;
            dispatch.bindings = std::get<Bindings>(std::move(bindings));
            std::uint64_t staging = 0;
            for (const auto& buffer : dispatch.bindings.buffers) {
                if (buffer.use == BufferUse::Storage && buffer.buffer.key.bytes.backing.owner) continue;
                if (buffer.buffer.size > limits_.staging_bytes - staging)
                    return error(CommandStatus::BudgetExceeded, "Renderer compute uniform snapshots exceed the staging budget");
                staging += buffer.buffer.size;
            }
            dispatch.argument_bytes_limit = limits_.staging_bytes - staging;
            auto pipeline = services_.pipelines->acquire_compute(
                {services_.device->identity(), admitted.key}, admitted);
            if (const auto* failed = std::get_if<RendererError>(&pipeline)) return *failed;
            dispatch.pipeline = std::get<ResidentComputePipeline>(std::move(pipeline));
            result.operation = std::move(dispatch);
        } else {
            return error(CommandStatus::Unsupported, "Renderer accepts only draw, clear and compute requests");
        }
        // Empty attachment sets still carry their bounded framebuffer extent;
        // they are real passes and cannot share an encoder with attached draws.
        result.attachments.key.area = result.targets.area;
        result.attachments.key.samples = result.targets.samples;
        // Attachments of the epoch's previous draw serve while the planned
        // targets are unchanged and no CPU store reached them.
        auto& cached = reuse_.attachments;
        if (stop.kind == RenderKind::Draw && cached && cached->epoch == reuse_epoch_ &&
            cached->targets == result.targets &&
            std::ranges::none_of(cached->accesses, [&](const GpuAccessRequest& access) {
                return registry_->cpu_written_since(access.range.bytes, cached->stamp);
            })) {
            result.attachments = cached->attachments;
            result.request.dependencies.insert(result.request.dependencies.end(),
                cached->dependencies.begin(), cached->dependencies.end());
            if (reuse_.last_packet.valid()) result.request.dependencies.push_back(reuse_.last_packet);
            result.accesses = cached->accesses;
            result.access_storage = cached->access_storage;
        } else {
            cached.reset();
            const auto stamp = registry_->cpu_write_stamp();
            const auto dependencies = result.request.dependencies.size();
            for (std::size_t i = 0; i != result.targets.colors.size(); ++i) {
                if (!result.targets.colors[i]) continue;
                auto status = attachment(*result.targets.colors[i], result.attachments.colors[i],
                                         result.attachments.key.colors[i], result);
                if (const auto* failed = std::get_if<RendererError>(&status)) return *failed;
            }
            if (result.targets.depth) {
                auto status = attachment(*result.targets.depth, result.attachments.depth,
                                         result.attachments.key.depth, result);
                if (const auto* failed = std::get_if<RendererError>(&status)) return *failed;
            }
            if (result.targets.stencil) {
                auto status = attachment(*result.targets.stencil, result.attachments.stencil,
                                         result.attachments.key.stencil, result);
                if (const auto* failed = std::get_if<RendererError>(&status)) return *failed;
            }
            if (stop.kind == RenderKind::Draw && stamp)
                cached = DrawReuse::Attachments{reuse_epoch_, stamp, result.targets, result.attachments,
                    {result.request.dependencies.begin() + std::ptrdiff_t(dependencies),
                     result.request.dependencies.end()},
                    result.accesses, result.access_storage};
            // Record acquired storage once per attachment acquisition. Draws
            // reusing these attachments remain covered by this epoch's entry.
            // Only acquired leases prove that a sample needs no transfer or copy.
            const auto log_attachment = [&](const std::optional<ResidentTexture>& texture) {
                if (!texture) return;
                const auto& descriptor = texture->key.storage.descriptor;
                epoch_log_.write(descriptor.address, descriptor.backing_extent, reuse_epoch_,
                                 resident_texture_identity(*texture));
            };
            for (const auto& color : result.attachments.colors) log_attachment(color);
            log_attachment(result.attachments.depth);
            log_attachment(result.attachments.stencil);
        }
        if (const auto* draw = std::get_if<PreparedDraw>(&result.operation)) {
            auto status = binding_accesses(draw->bindings, result);
            if (const auto* failed = std::get_if<RendererError>(&status)) return *failed;
        } else if (const auto* dispatch = std::get_if<PreparedDispatch>(&result.operation)) {
            auto status = binding_accesses(dispatch->bindings, result);
            // The dispatch's writes are its write accesses.
            for (const auto& access : result.accesses)
                if (access.access != ResourceAccess::Read)
                    epoch_log_.write(access.range.bytes, reuse_epoch_);
            if (const auto* failed = std::get_if<RendererError>(&status)) return *failed;
        }
        auto normalized = normalize_accesses(result);
        if (const auto* failed = std::get_if<RendererError>(&normalized)) return *failed;
        return result;
    }

    // What a repeat of the last prepared draw must match.
    struct LastDraw {
        std::uint64_t epoch = 0;
        std::vector<VertexClassBinding> inputs;
        PrimitiveTopology topology{};
        PrimitiveClass primitive{};
        decltype(DrawPlan{}.builtin_emulation) emulation{};
        bool indexed = false;
        IndexFormat format{};
        std::uint64_t index_address = 0, index_limit = 0;
        bool converted = false;
        std::uint32_t base_instance = 0, instance_count = 0;
        bool depth_bias = false;
        // Stream layout registers (stride/enable and divisor words, instance
        // flags): a repeat may move streams but never change their layout.
        std::array<std::uint32_t, 48> stream_layout{};
        // Attribute registers of the locations the vertex stage reads (in
        // inputs order): others may change without affecting the draw.
        std::vector<std::uint32_t> attributes;
        // Planned targets and render state before pruning: a repeat may
        // follow target or state register changes that plan identically.
        RenderTargetPlan targets;
        RenderState state;
        // The state the source packet encodes, whether it has attachments,
        // and the viewport/scissor override repeats carry once they move.
        RenderState applied;
        bool attached = false;
        std::shared_ptr<const RenderState> override_state;
        // The index binding an earlier repeat switched to, if any.
        std::optional<ResidentBuffer> index_override;
        // The pipeline an earlier repeat switched to for a new vertex
        // attribute layout, if any.
        std::shared_ptr<const ResidentPipeline> pipeline_override;
        bool prefix_safe_completion = false;
        // The vertex plan repeats borrow while their streams stay put (native
        // index selection reads it).
        VertexInputPlan vertex;
    };
    std::optional<LastDraw> last_draw_;
    static std::array<std::uint32_t, 48> stream_layout(const RenderStop& stop) {
        std::array<std::uint32_t, 48> layout{};
        for (std::size_t i = 0; i != 16; ++i) {
            layout[i] = stop.registers[0x700 + 4 * i];
            layout[16 + i] = stop.registers[0x703 + 4 * i];
            layout[32 + i] = stop.registers[0x620 + i];
        }
        return layout;
    }

    // A draw that changed no register group since the last
    // prepared draw costs only its draw parameters. It assembles its draw plan
    // (no index read), refreshes uniform windows a CPU store reached, and the
    // queue encodes it into the last draw's open batch: no preparation,
    // planning, reservation or registry publication.
    std::optional<SubmissionReceipt> try_repeat(const RenderRequest& request) {
        static const bool disabled = std::getenv("ARMIGO_DISABLE_REPEAT") != nullptr;
        static const bool counting = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        // Fallback reasons: 0-10 blocking dirty groups, then the checks below.
        static std::array<std::uint64_t, 30> misses{};
        static std::uint64_t calls = 0, hits = 0;
        // Blocking dirty-group combinations of reason 19, by mask.
        static std::array<std::uint64_t, threed_dirty::bit(threed_dirty::Count)> blocking_masks{};
        const auto miss = [&](std::size_t reason) -> std::optional<SubmissionReceipt> {
            if (!counting) return std::nullopt;
            ++misses[reason];
            if (++calls % 20000 == 0) {
                armigo::diag::opt_in( "repeat_stats calls=%llu hits=%llu", (unsigned long long)calls,
                             (unsigned long long)hits);
                for (std::size_t i = 0; i < misses.size(); ++i)
                    if (misses[i]) armigo::diag::opt_in( " r%zu=%llu", i, (unsigned long long)misses[i]);
                armigo::diag::newline();
                if (calls % 200000 == 0) {
                    std::vector<std::pair<std::uint64_t, std::size_t>> top;
                    for (std::size_t mask = 0; mask < blocking_masks.size(); ++mask)
                        if (blocking_masks[mask]) top.emplace_back(blocking_masks[mask], mask);
                    std::sort(top.rbegin(), top.rend());
                    armigo::diag::opt_in( "repeat_blocking");
                    for (std::size_t i = 0; i < std::min<std::size_t>(top.size(), 12); ++i)
                        armigo::diag::opt_in( " %#zx=%llu", top[i].second, (unsigned long long)top[i].first);
                    armigo::diag::newline();
                }
            }
            return std::nullopt;
        };
        const auto& stop = *request.snapshot;
        // Index buffer, vertex stream and constant-buffer binds rebind their
        // buffers; every other group needs a full preparation.
        constexpr auto rebindable = threed_dirty::bit(threed_dirty::IndexBuffer) |
            threed_dirty::bit(threed_dirty::VertexStreams) |
            threed_dirty::bit(threed_dirty::ConstantBuffers0) | threed_dirty::bit(threed_dirty::ConstantBuffers0 + 4);
        if (disabled || registry_->has_deferred_host_store() || !last_draw_ || !stop.draw ||
            last_draw_->prefix_safe_completion !=
                (limits_.defer_gpu_writeback && request.prefix_safe_completion))
            return miss(12);
        if (last_draw_->epoch != reuse_epoch_) return miss(13);
        // A shader depth bias is a fragment uniform built from State registers
        // alone, so a repeat with State clean keeps it bound.
        if (last_draw_->converted) return miss(14);
        if (!stop.inline_indices.empty()) return miss(20);
        if (!reuse_.last_packet.valid() || registry_->address_space().generation() != reuse_.view)
            return miss(15);
        auto dirty = reuse_dirty(stop);
        // Attribute registers the vertex stage never reads leave the draw
        // unchanged.
        // Read attributes that change format or offset over the same streams
        // repeat with the pipeline of the new layout (same stages and outputs).
        bool attributes_moved = false;
        if (dirty & threed_dirty::bit(threed_dirty::VertexAttributes)) {
            bool same = last_draw_->attributes.size() == last_draw_->inputs.size();
            for (std::size_t i = 0; same && i < last_draw_->inputs.size(); ++i)
                same = stop.registers[0x458 + last_draw_->inputs[i].location] == last_draw_->attributes[i];
            if (same || (reuse_.pipeline && reuse_.stages &&
                         reuse_.stages->stages.vertex == reuse_.pipeline->vertex)) {
                attributes_moved = !same;
                dirty &= ~threed_dirty::bit(threed_dirty::VertexAttributes);
            }
        }
        // Target and state registers that plan the same targets and render
        // state (and still pass admission) leave the draw unchanged.
        constexpr auto output = threed_dirty::bit(threed_dirty::RenderTargets) | threed_dirty::bit(threed_dirty::State);
        // Point draws also take their size from a state register outside
        // the planned render state.
        if ((dirty & output) && !(dirty & ~(rebindable | output)) &&
            last_draw_->primitive != PrimitiveClass::Point) {
            auto planned = plan_render_targets(stop);
            auto* targets = std::get_if<RenderTargetPlan>(&planned);
            if (!targets) return miss(21);
            targets->area = render_pass_area(*targets);
            if (*targets == last_draw_->targets) {
                bool same_state = !(dirty & threed_dirty::bit(threed_dirty::State));
                if (!same_state && !std::holds_alternative<RendererError>(services_.stages->validate(stop))) {
                    auto state = plan_render_state(stop, *targets);
                    const auto* planned_state = std::get_if<RenderState>(&state);
                    same_state = planned_state && *planned_state == last_draw_->state;
                    if (planned_state && !same_state && last_draw_->attached) {
                        // Viewports, scissors, cull mode, winding, depth clip,
                        // blend constant and stencil references are encoder
                        // state: a change confined
                        // to them repeats with the source state carrying the
                        // new ones.
                        const auto encoder_state = [](RenderState& to, const RenderState& from) {
                            to.viewports = from.viewports;
                            to.scissors = from.scissors;
                            to.rasterizer.cull = from.rasterizer.cull;
                            to.rasterizer.front_face = from.rasterizer.front_face;
                            to.rasterizer.depth_clip = from.rasterizer.depth_clip;
                            to.blend_constant = from.blend_constant;
                            to.front_stencil_reference = from.front_stencil_reference;
                            to.back_stencil_reference = from.back_stencil_reference;
                        };
                        auto fixed = *planned_state;
                        encoder_state(fixed, last_draw_->state);
                        if (fixed == last_draw_->state) {
                            if (counting) ++misses[22];
                            encoder_state(last_draw_->applied, *planned_state);
                            last_draw_->override_state = std::make_shared<const RenderState>(last_draw_->applied);
                            last_draw_->state = *planned_state;
                            same_state = true;
                        }
                    }
                }
                if (same_state) dirty &= ~output;
            }
        }
        if (const auto blocking = dirty & ~rebindable) {
            if (counting) {
                for (std::uint32_t g = 0; g < threed_dirty::Count; ++g)
                    if (blocking & threed_dirty::bit(g)) ++misses[g];
                ++blocking_masks[blocking & threed_dirty::all];
                blocking_ = blocking & threed_dirty::all;
            }
            return miss(19);
        }
        // Only the layout of the streams the draw reads must hold; a read
        // stream whose stride changed (still enabled, same divisor and
        // instancing) repeats with the pipeline of the new layout.
        bool streams_relaid = false;
        if (dirty & threed_dirty::bit(threed_dirty::VertexStreams)) {
            const auto layout = stream_layout(stop);
            if (layout != last_draw_->stream_layout) {
                for (const auto& stream : last_draw_->vertex.streams) {
                    const auto i = stream.stream;
                    if (i >= 16 || layout[16 + i] != last_draw_->stream_layout[16 + i] ||
                        layout[32 + i] != last_draw_->stream_layout[32 + i])
                        return miss(16);
                    if (layout[i] == last_draw_->stream_layout[i]) continue;
                    // Bit 12 enables the stream; the low 12 bits hold its stride.
                    if (!(layout[i] & 0x1000u) || (layout[i] & ~0xfffu) != (last_draw_->stream_layout[i] & ~0xfffu) ||
                        !reuse_.pipeline || !reuse_.stages || reuse_.stages->stages.vertex != reuse_.pipeline->vertex)
                        return miss(16);
                    streams_relaid = true;
                }
            }
        }
        // A repeat keeps the last draw's vertex layout; the stream plan is
        // needed only to rebind moved streams.
        const bool streams_moved = (dirty & threed_dirty::bit(threed_dirty::VertexStreams)) != 0;
        const bool plan_vertex = streams_moved || attributes_moved || streams_relaid;
        auto assembled = assemble_draw(stop, last_draw_->inputs, registry_->address_space(), {}, true,
                                       plan_vertex);
        auto* plan = std::get_if<DrawPlan>(&assembled);
        if (!plan) return miss(23);
        // Unmoved streams borrow the last draw's vertex plan; moved ones
        // replace it once the repeat is accepted.
        struct Borrow {
            VertexInputPlan* from;
            VertexInputPlan* to;
            Borrow(VertexInputPlan* from, VertexInputPlan* to) : from(from), to(to) {
                if (from) std::swap(*from, *to);
            }
            Borrow(const Borrow&) = delete;
            Borrow& operator=(const Borrow&) = delete;
            ~Borrow() { if (from) std::swap(*from, *to); }
        } borrow(plan_vertex ? nullptr : &plan->vertex, &last_draw_->vertex);
        if (attributes_moved || streams_relaid) {
            // Only attribute formats and offsets, and stream strides, may
            // differ: the streams keep their slots and steps (and addresses,
            // unless the streams moved).
            const auto& before = last_draw_->vertex;
            const auto& after = plan->vertex;
            const bool same_streams = before.steps == after.steps &&
                std::ranges::equal(before.streams, after.streams, [&](const auto& a, const auto& b) {
                    return a.stream == b.stream && a.divisor == b.divisor &&
                           (streams_relaid || a.stride == b.stride) &&
                           (streams_moved || (a.address == b.address && a.inclusive_limit == b.inclusive_limit));
                });
            if (!same_streams) return miss(28);
        }
        if (auto native = select_native_draw(*plan, stop, *registry_); std::holds_alternative<RendererError>(native))
            return miss(24);
        if (plan->segments.empty() || plan->topology != last_draw_->topology ||
            plan->primitive_class != last_draw_->primitive ||
            plan->builtin_emulation != last_draw_->emulation ||
            plan->indices.has_value() != last_draw_->indexed ||
            plan->source.base_instance != last_draw_->base_instance ||
            plan->source.instance_count != last_draw_->instance_count)
            return miss(18);
        const bool same_indices = !plan->indices ||
            (plan->indices->address == last_draw_->index_address &&
             plan->indices->inclusive_limit == last_draw_->index_limit);
        if (plan->indices && (!plan->indices->converted.empty() ||
                              plan->indices->encoded_format != last_draw_->format ||
                              (!same_indices && !(dirty & threed_dirty::bit(threed_dirty::IndexBuffer)))))
            return miss(25);
        std::shared_ptr<const ResidentPipeline> swapped;
        if (attributes_moved || streams_relaid) {
            auto key = reuse_.pipeline->pipeline.key;
            if (std::holds_alternative<RendererError>(vertex_layout_key(key, *plan, *reuse_.stages->stages.vertex)))
                return miss(29);
            if (key.attributes != reuse_.pipeline->pipeline.key.attributes ||
                key.buffers != reuse_.pipeline->pipeline.key.buffers) {
                auto acquired = services_.pipelines->acquire(key, *reuse_.stages);
                if (std::holds_alternative<RendererError>(acquired)) return miss(29);
                swapped = std::make_shared<const ResidentPipeline>(std::get<ResidentPipeline>(std::move(acquired)));
            }
        }
        BindingPlanner::RepeatBindings bindings;
        if (!services_.bindings->refresh_repeat(stop, *plan, dirty, *registry_, *services_.buffers, bindings))
            return miss(17);
        if (plan->indices && !same_indices && !bindings.index_buffer) return miss(26);
        RepeatDraw repeat;
        repeat.changed = std::move(bindings.changed);
        repeat.vertex_buffers = std::move(bindings.vertex_buffers);
        if (!same_indices) last_draw_->index_override = std::move(bindings.index_buffer);
        repeat.index_buffer = last_draw_->index_override;
        repeat.segments = std::move(plan->segments);
        repeat.state = last_draw_->override_state;
        repeat.pipeline = swapped ? swapped : last_draw_->pipeline_override;
        if (!services_.submissions->append_repeat(std::move(repeat))) return miss(27);
        if (streams_relaid) last_draw_->stream_layout = stream_layout(stop);
        if (attributes_moved || streams_relaid) {
            for (std::size_t i = 0; i < last_draw_->inputs.size(); ++i)
                last_draw_->attributes[i] = stop.registers[0x458 + last_draw_->inputs[i].location];
            if (swapped) {
                last_draw_->pipeline_override = swapped;
                // The next full preparation reuses the pipeline of this layout.
                reuse_.pipeline->pipeline = *swapped;
            }
        }
        if (plan->indices) {
            last_draw_->index_address = plan->indices->address;
            last_draw_->index_limit = plan->indices->inclusive_limit;
        }
        if (plan_vertex) last_draw_->vertex = std::move(plan->vertex);
        if (counting) ++hits;
        repeated_ = true;
        reuse_.stream = stop.state_stream;
        reuse_.serial = stop.state_serial;
        queue_ = request.id.queue;
        sequence_ = request.id.sequence;
        return SubmissionReceipt{reuse_.last_packet};
    }

    // Derived state of the last prepared draw is reused per register group;
    // only dirty groups run their update callbacks.
    struct DrawReuse {
        std::uint64_t stream = 0, serial = 0;
        // The guest mappings of the last address-space view; owners are
        // compared without being kept alive.
        struct ViewMapping {
            std::uint64_t gpu_base = 0, mapped_size = 0, buffer_offset = 0, guest_address = 0,
                          backing_size = 0, page_size = 0;
            std::weak_ptr<void> owner;
        };
        std::vector<ViewMapping> mappings;
        // Admitted stages with the guest program ranges they were identified
        // from and the CPU write stamp taken before that identification.
        std::optional<AdmittedGraphicsStages> stages;
        std::vector<RetainedRange> code;
        std::uint64_t code_stamp = 0;
        // Address-space view and attachment GPU ranges of the current epoch.
        std::uint64_t view = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> attachment_ranges;
        // Attachment acquisition results of the epoch's last draw.
        struct Attachments {
            std::uint64_t epoch = 0, stamp = 0;
            // The planned targets (after write-mask pruning) they serve.
            RenderTargetPlan targets;
            ResidentAttachments attachments;
            std::vector<CompletionToken> dependencies;
            std::vector<GpuAccessRequest> accesses;
            std::vector<RetainedResource> access_storage;
        };
        std::optional<Attachments> attachments;
        // Pipeline objects of the last draw and the inputs they depend on.
        struct Pipeline {
            std::shared_ptr<const CompiledGraphicsStage> vertex, fragment;
            PrimitiveClass primitive{};
            decltype(DrawPlan{}.builtin_emulation) emulation{};
            bool depth_bias = false;
            std::uint32_t base_instance = 0, instance_count = 0;
            ResidentPipeline pipeline;
            RetainedResource depth_stencil;
        };
        std::optional<Pipeline> pipeline;
        // The last accepted packet: within an epoch it is the latest writer of
        // the shared attachments, which reused attachments depend on.
        CompletionToken last_packet;
    };
    DrawReuse reuse_;
    // Changes whenever anything other than consecutive draws with the same
    // attachments may have touched bound resources: other GPU operations,
    // attachment or address-space changes and failed preparation. Submission
    // and completion keep it; reused attachments depend on the last packet,
    // which orders them across command buffers.
    std::uint64_t reuse_epoch_ = 1;
    // What each epoch writes: its attachments, plus a clear's targets; `all`
    // for operations whose writes are unknown.
    EpochLog epoch_log_ = [] {
        EpochLog log;
        log.last_unknown = 1;
        return log;
    }();
    // Unknown-epoch causes for tracing: preparation failure, append failure,
    // a non-draw operation (by kind), an address-space view change.
    void count_unknown(std::size_t cause) {
        static const bool trace = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        static std::array<std::uint64_t, 12> causes{};
        if (!trace) return;
        ++causes[std::min<std::size_t>(cause, causes.size() - 1)];
        std::uint64_t total = 0;
        for (const auto value : causes) total += value;
        if (total % 100) return;
        armigo::diag::opt_in( "epoch_unknown");
        for (std::size_t i = 0; i < causes.size(); ++i)
            if (causes[i]) armigo::diag::opt_in( " c%zu=%llu", i, (unsigned long long)causes[i]);
        armigo::diag::opt_in( " overflows=%llu\n", (unsigned long long)epoch_log_.overflows);
    }
    void bump_epoch(bool all) {
        ++reuse_epoch_;
        if (all) epoch_log_.unknown(reuse_epoch_);
    }

    // Groups changed since the derived state in reuse_ was built: all of them
    // unless this stop directly follows (or repeats) the last prepared draw.
    std::uint64_t reuse_dirty(const RenderStop& stop) const {
        if (!stop.state_stream || stop.state_stream != reuse_.stream) return threed_dirty::all;
        if (stop.state_serial == reuse_.serial) return 0;
        if (stop.state_serial == reuse_.serial + 1) return stop.dirty_groups;
        return threed_dirty::all;
    }

    static void trace_dirty(const RenderStop& stop, std::uint64_t dirty) {
        static const bool enabled = std::getenv("ARMIGO_TRACE_ENGINE_FLUSH") != nullptr;
        if (!enabled) return;
        static std::array<std::uint64_t, threed_dirty::Count + 2> counts{};
        static std::uint64_t draws = 0;
        ++draws;
        if (dirty == threed_dirty::all && stop.dirty_groups != threed_dirty::all) ++counts[threed_dirty::Count];
        if (!dirty) ++counts[threed_dirty::Count + 1];
        for (std::uint32_t group = 0; group != threed_dirty::Count; ++group)
            if (dirty & threed_dirty::bit(group)) ++counts[group];
        if (draws % 20000) return;
        armigo::diag::opt_in( "draw_dirty draws=%llu rt=%llu shaders=%llu streams=%llu attributes=%llu "
            "index=%llu pools=%llu cb=%llu/%llu/%llu/%llu/%llu state=%llu broken=%llu clean=%llu\n",
            (unsigned long long)draws, (unsigned long long)counts[0], (unsigned long long)counts[1],
            (unsigned long long)counts[2], (unsigned long long)counts[3], (unsigned long long)counts[4],
            (unsigned long long)counts[5], (unsigned long long)counts[6], (unsigned long long)counts[7],
            (unsigned long long)counts[8], (unsigned long long)counts[9], (unsigned long long)counts[10],
            (unsigned long long)counts[11], (unsigned long long)counts[12], (unsigned long long)counts[13]);
    }

    // Stage identification depends on program, vertex input, render-target
    // and remaining state registers. Clean groups reuse the admitted stages
    // while the write tracker proves both programs' guest bytes unchanged.
    RendererResult<AdmittedGraphicsStages> reuse_stages(const RenderStop& stop, std::uint64_t dirty) {
        constexpr auto inputs = threed_dirty::bit(threed_dirty::Shaders) |
            threed_dirty::bit(threed_dirty::VertexAttributes) | threed_dirty::bit(threed_dirty::VertexStreams) |
            threed_dirty::bit(threed_dirty::RenderTargets) | threed_dirty::bit(threed_dirty::TexturePools) |
            threed_dirty::bit(threed_dirty::State);
        // The general state group alone changes no selection input the
        // provider's admission checks leave free (see validate()).
        const auto selection = inputs & ~threed_dirty::bit(threed_dirty::State);
        if (!(dirty & selection) && reuse_.stages && reuse_.code.size() == 2 &&
            std::ranges::none_of(reuse_.code, [&](const RetainedRange& range) {
                return registry_->cpu_written_since(range.bytes, reuse_.code_stamp);
            }) &&
            (!(dirty & threed_dirty::bit(threed_dirty::State)) ||
             !std::holds_alternative<RendererError>(services_.stages->validate(stop))))
            return *reuse_.stages;
        reuse_.stages.reset();
        reuse_.code.clear();
        // Programs seen before: their admitted pair serves while both programs'
        // code is unchanged, the registers stage selection reads are equal and
        // the stop passes admission.
        const auto key = std::pair{stop.programs[1].address, stop.programs[5].address};
        const auto inputs_now = selection_inputs(stop);
        if (const auto found = stage_pairs_.find(key); found != stage_pairs_.end()) {
            auto& pair = found->second;
            if (pair.inputs == inputs_now &&
                std::ranges::none_of(pair.code, [&](const RetainedRange& range) {
                    return registry_->cpu_written_since(range.bytes, pair.code_stamp);
                }) &&
                !std::holds_alternative<RendererError>(services_.stages->validate(stop))) {
                reuse_.stages = pair.admitted;
                reuse_.code = pair.code;
                reuse_.code_stamp = pair.code_stamp;
                return pair.admitted;
            }
            stage_pairs_.erase(found);
        }
        const auto stamp = registry_->cpu_write_stamp();
        auto stages = services_.stages->lookup(stop, registry_->address_space());
        if (const auto* failed = std::get_if<RendererError>(&stages)) return *failed;
        const auto& admitted = std::get<AdmittedGraphicsStages>(stages);
        if (stamp && admitted.stages.vertex && admitted.stages.fragment &&
            admitted.stages.vertex->guest_code_bytes && admitted.stages.fragment->guest_code_bytes) {
            std::vector<RetainedRange> code;
            for (const auto& [program, stage] : {std::pair{stop.programs[1], admitted.stages.vertex},
                                                 std::pair{stop.programs[5], admitted.stages.fragment}}) {
                auto retained = registry_->retain(program.address, stage->guest_code_bytes, ResourceAccess::Read);
                if (auto* range = std::get_if<RetainedRange>(&retained)) code.push_back(std::move(*range));
            }
            if (code.size() == 2) {
                if (stage_pairs_.size() >= 1024) stage_pairs_.clear();
                stage_pairs_.insert_or_assign(key, StagePair{admitted, code, stamp, inputs_now});
                reuse_.stages = admitted;
                reuse_.code = std::move(code);
                reuse_.code_stamp = stamp;
            }
        }
        return stages;
    }
    // Registers stage selection reads besides the programs: vertex attribute
    // formats, stream controls and instancing, render-target formats and
    // control, depth format/enable, the bindless selector and clip convention.
    static std::vector<std::uint32_t> selection_inputs(const RenderStop& stop) {
        std::vector<std::uint32_t> values;
        values.reserve(96);
        for (std::uint32_t i = 0; i < 0x20; ++i) values.push_back(stop.registers[0x458 + i]);
        for (std::uint32_t i = 0; i < 16; ++i) values.push_back(stop.registers[0x700 + 4 * i]);
        for (std::uint32_t i = 0; i < 16; ++i) values.push_back(stop.registers[0x620 + i]);
        for (std::uint32_t i = 0; i < 8; ++i) values.push_back(stop.registers[0x200 + 0x10 * i + 4]);
        for (const std::uint32_t method : {0x487u, 0x3fau, 0x54eu, 0x982u, 0x35fu})
            values.push_back(stop.registers[method]);
        return values;
    }
    struct StagePair {
        AdmittedGraphicsStages admitted;
        std::vector<RetainedRange> code;
        std::uint64_t code_stamp = 0;
        std::vector<std::uint32_t> inputs;
    };
    std::map<std::pair<std::uint64_t, std::uint64_t>, StagePair> stage_pairs_;

    RendererLimits limits_;
    RendererServices services_;
    std::shared_ptr<ResourceMemoryRegistry> registry_;
    std::vector<CompletionToken> pending_;
    std::vector<CompletionToken> flushed_;
    std::uint64_t queue_ = 0, sequence_ = 0;
    std::uint32_t accepted_since_flush_ = 0;
};
} // namespace

RendererResult<std::shared_ptr<MetalRenderer>> create_renderer(
    RendererLimits limits, RendererServices services) {
    if (!limits.resident_bytes || !limits.staging_bytes || !limits.inflight_submissions ||
        !limits.commands_per_submission || !services.device || !services.memory ||
        !services.stages || !services.textures || !services.buffers || !services.pipelines ||
        !services.bindings || !services.submissions)
        return error(CommandStatus::Unsupported, "Renderer requires all services and nonzero resource limits");
    const auto registry = std::dynamic_pointer_cast<ResourceMemoryRegistry>(services.memory);
    if (!registry || registry->coherence_mode() != CoherenceMode::SubmissionBoundary)
        return error(CommandStatus::Unsupported, "Renderer requires verified submission-boundary coherence");
    const auto& identity = services.device->identity();
    const ResidencyDomain domain{identity.device, services.memory.get()};
    if (!identity.device || !services.device->device() || !services.device->queue() ||
        services.textures->domain() != domain || services.buffers->domain() != domain ||
        services.submissions->domain() != domain || services.pipelines->context() != identity)
        return error(CommandStatus::Unsupported, "Renderer services belong to incompatible device or memory domains");
    try {
        return std::shared_ptr<MetalRenderer>(
            std::make_shared<RendererImpl>(limits, std::move(services), registry));
    } catch (const std::bad_alloc&) {
        return error(CommandStatus::BudgetExceeded, "Renderer allocation failed");
    }
}
} // namespace armigo::platform::metal
