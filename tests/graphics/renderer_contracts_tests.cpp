// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/renderer_types.h>
#include <armigo/graphics/render_submission.h>
#include <armigo/graphics/resource_memory.h>
#include <armigo/graphics/texture_cache.h>
#include <armigo/graphics/buffer_cache.h>
#include <armigo/graphics/render_target.h>
#include <armigo/graphics/render_state.h>
#include <armigo/graphics/draw_plan.h>
#include <armigo/graphics/renderer.h>

#include "../../src/platform/metal/pass_access.h"

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

namespace {
using namespace armigo::graphics;
unsigned failures = 0;
#define CHECK(...) do { if (!(__VA_ARGS__)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); } } while (false)

bool succeeded(const RendererStatus& result) {
    return std::holds_alternative<std::monostate>(result);
}
RendererError unsupported(const char* detail) {
    return RendererError({CommandStatus::Unsupported, detail});
}
struct Lease final : ResourceLease {};

enum class ActionFailure { None, Resolve, Prepare, ThrowResolve, ThrowPrepare };
struct Actions final : CompletionActions {
    Actions() { calls.reserve(3); }
    ActionFailure failure = ActionFailure::None;
    std::vector<int> calls;
    std::weak_ptr<const ResourceLease> resource;
    bool check_resource = false;
    CompletionToken observed;
    std::thread::id coordinator = std::this_thread::get_id();
    unsigned published = 0;
    std::function<void()> during_resolve;
    std::function<void()> handoff;
    bool refuse_handoff = false, throw_handoff = false;
    unsigned handoff_calls = 0;

    void check_context() {
        CHECK(std::this_thread::get_id() == coordinator);
        if (check_resource) CHECK(!resource.expired());
        if (observed.valid()) {
            const auto state = observed.snapshot();
            CHECK(state.phase == SubmissionPhase::Publishing);
            CHECK(!state.guest_visible_complete());
        }
    }
    RendererStatus make_coherent() override {
        check_context();
        calls.push_back(1);
        if (during_resolve) during_resolve();
        if (failure == ActionFailure::ThrowResolve) throw std::runtime_error("synthetic resolve failure");
        if (failure == ActionFailure::Resolve) return unsupported("synthetic resolve failure");
        return renderer_success();
    }
    RendererStatus prepare_publication() override {
        check_context();
        calls.push_back(2);
        if (failure == ActionFailure::ThrowPrepare) throw std::runtime_error("synthetic publication failure");
        if (failure == ActionFailure::Prepare) return unsupported("synthetic publication failure");
        return renderer_success();
    }
    void publish() noexcept override {
        check_context();
        calls.push_back(3);
        ++published;
    }
    RendererStatus preserve_failed_authority() override {
        CHECK(std::this_thread::get_id() == coordinator);
        if (check_resource) CHECK(!resource.expired());
        if (observed.valid()) CHECK(observed.snapshot().phase == SubmissionPhase::Failed);
        ++handoff_calls;
        if (throw_handoff) throw std::runtime_error("synthetic recovery failure");
        if (refuse_handoff) return unsupported("synthetic recovery refused");
        if (handoff) handoff();
        return renderer_success();
    }
};

// This oracle retains every access, independently of the summary's keys and
// insertion policy. Pass compatibility is the existence of a conflicting pair.
struct PassHistoryAccess {
    BackingRange bytes;
    ResourceAccess access;
    bool attachment;
};

bool history_conflicts(const std::vector<PassHistoryAccess>& history,
                       const PassHistoryAccess& incoming) {
    for (const auto& previous : history) {
        if (previous.bytes.backing != incoming.bytes.backing) continue;
        if (previous.bytes.offset >= incoming.bytes.offset + incoming.bytes.size ||
            incoming.bytes.offset >= previous.bytes.offset + previous.bytes.size)
            continue;
        if (previous.access == ResourceAccess::Read &&
            incoming.access == ResourceAccess::Read) continue;
        if (previous.attachment && incoming.attachment) continue;
        return true;
    }
    return false;
}

void test_pass_access_summary() {
    using armigo::platform::metal::PassAccessSummary;
    const BackingRange bytes{{1, 2, 3}, 16, 16};
    PassAccessSummary summary;
    for (unsigned draw = 0; draw != 4096; ++draw) {
        summary.record(bytes, ResourceAccess::Read, false);
        summary.record(bytes, ResourceAccess::Write, true);
    }
    CHECK(summary.size() == 2);
    CHECK(!summary.conflicts(bytes, ResourceAccess::Read, true));
    CHECK(summary.conflicts(bytes, ResourceAccess::Read, false));
    CHECK(summary.conflicts(bytes, ResourceAccess::Write, true));
    CHECK(summary.conflicts(bytes, ResourceAccess::Write, false));

    // A later read cannot forget an earlier write to the same exact range.
    summary.clear();
    summary.record(bytes, ResourceAccess::Write, false);
    summary.record(bytes, ResourceAccess::Read, false);
    CHECK(summary.size() == 1);
    CHECK(summary.conflicts(bytes, ResourceAccess::Read, false));
    CHECK(summary.conflicts(bytes, ResourceAccess::Read, true));
    summary.clear();
    CHECK(summary.size() == 0);
    CHECK(!summary.conflicts(bytes, ResourceAccess::Write, false));

    const std::array ranges{
        bytes,
        BackingRange{{1, 2, 3}, 16, 8},       // Contained interval.
        BackingRange{{1, 2, 3}, 8, 16},       // Partial overlap.
        BackingRange{{1, 2, 3}, 8, 32},       // Contains the original interval.
        BackingRange{{1, 2, 3}, 0, 16},       // Touches its beginning.
        BackingRange{{1, 2, 3}, 32, 16},      // Touches its end.
        BackingRange{{1, 2, 3}, 64, 16},      // Disjoint interval.
        BackingRange{{2, 2, 3}, 16, 16},      // Other owner.
        BackingRange{{1, 3, 3}, 16, 16},      // Other allocation.
        BackingRange{{1, 2, 4}, 16, 16},      // Reallocated backing.
    };
    const std::array access_modes{
        ResourceAccess::Read, ResourceAccess::Write, ResourceAccess::ReadWrite};
    std::vector<PassHistoryAccess> cases;
    for (const auto& range : ranges)
        for (const auto access : access_modes)
            for (const bool attachment : {false, true})
                cases.push_back({range, access, attachment});
    const auto record = [&](const PassHistoryAccess& access) {
        summary.record(access.bytes, access.access, access.attachment);
    };
    const auto compare = [&](const std::vector<PassHistoryAccess>& history) {
        for (const auto& incoming : cases)
            CHECK(summary.conflicts(incoming.bytes, incoming.access, incoming.attachment) ==
                  history_conflicts(history, incoming));
    };
    // Exhaust all ordered two-access histories and every candidate query.
    // This includes read-to-write upgrades and both attachment classes for
    // identical bytes, which must not be collapsed into one attachment record.
    for (const auto& first : cases) {
        for (const auto& second : cases) {
            summary.clear();
            record(first);
            record(second);
            compare({first, second});
        }
    }

    // Repeated and changing accesses stay equivalent across many draws and
    // across pass resets. The bounded range set also bounds summary storage.
    std::vector<PassHistoryAccess> history;
    std::uint32_t random = 0x12345678;
    for (unsigned draw = 0; draw != 1024; ++draw) {
        if (draw % 127 == 0) {
            summary.clear();
            history.clear();
        }
        random = random * 1664525u + 1013904223u;
        const auto access = cases[random % cases.size()];
        history.push_back(access);
        record(access);
        compare(history);
        CHECK(summary.size() <= ranges.size() * 2);
    }
}

void test_owned_errors() {
    RendererResult<SubmissionReceipt> result;
    CHECK(std::holds_alternative<RendererError>(result));
    CHECK(!std::get<RendererError>(result).command().ok());
    CHECK(!RendererError(CommandResult{}).command().ok());
    const CompletionToken invalid;
    CHECK(!invalid.valid());
    CHECK(invalid.snapshot().phase == SubmissionPhase::Failed);
    CHECK(invalid.snapshot().error.has_value());
    CHECK(!invalid.snapshot().guest_visible_complete());

    RendererError copied;
    {
        std::string temporary = "owned synthetic diagnostic with temporary backing";
        CommandResult command{CommandStatus::MemoryFault, temporary, 0x1234, 7, 2, 9, 11, 13};
        RendererError original(command);
        copied = original;
        temporary.assign(temporary.size(), 'x');
        original = unsupported("replacement");
    }
    auto moved = std::move(copied);
    const auto command = moved.command();
    CHECK(command.detail == "owned synthetic diagnostic with temporary backing");
    CHECK(command.status == CommandStatus::MemoryFault);
    CHECK(command.address == 0x1234 && command.entry_index == 7);
    CHECK(command.subchannel == 2 && command.method == 9);
    CHECK(command.argument == 11 && command.engine == 13);
}

void test_submission_phases() {
    auto resource = std::make_shared<Lease>();
    auto actions = std::make_shared<Actions>();
    actions->resource = resource;
    actions->check_resource = true;
    SubmissionCompletion completion({2, 7}, {resource}, actions);
    auto token = completion.token();
    actions->observed = token;
    resource.reset();
    CHECK(token.snapshot().id == SubmissionId{2, 7});
    CHECK(token.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(!token.snapshot().guest_visible_complete());
    CHECK(!succeeded(completion.finish()));
    CHECK(!succeeded(completion.release_failed()));
    CHECK(!succeeded(completion.reject({})));
    CHECK(!succeeded(completion.feedback({})));
    CHECK(token.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(succeeded(completion.submitted()));
    CHECK(token.snapshot().phase == SubmissionPhase::Submitted);
    CHECK(!succeeded(completion.release_failed()));
    CHECK(!succeeded(completion.reject({CommandStatus::Unsupported, "too late"})));
    CHECK(!succeeded(completion.submitted()));
    CHECK(!succeeded(completion.finish()));
    CHECK(actions->calls.empty());
    CHECK(!actions->resource.expired());

    RendererStatus feedback_result;
    std::thread worker([completion, &feedback_result]() mutable {
        feedback_result = completion.feedback({});
    });
    worker.join();
    CHECK(succeeded(feedback_result));
    CHECK(token.snapshot().phase == SubmissionPhase::GpuComplete);
    CHECK(!succeeded(completion.release_failed()));
    CHECK(!token.snapshot().guest_visible_complete());
    CHECK(actions->calls.empty());
    CHECK(!actions->resource.expired());
    CHECK(!succeeded(completion.feedback({CommandStatus::MemoryFault, "late failure"})));
    CHECK(!token.snapshot().error);
    actions->during_resolve = [&completion] { CHECK(!succeeded(completion.finish())); };
    CHECK(succeeded(completion.finish()));
    CHECK(actions->calls == std::vector<int>({1, 2, 3}));
    CHECK(actions->published == 1);
    CHECK(token.snapshot().guest_visible_complete());
    CHECK(actions->resource.expired());
    CHECK(succeeded(completion.finish()));
    CHECK(!succeeded(completion.submitted()));
    CHECK(!succeeded(completion.feedback({})));
    CHECK(actions->calls == std::vector<int>({1, 2, 3}));
    CHECK(token.snapshot().phase == SubmissionPhase::Complete);
    CHECK(!succeeded(completion.release_failed()));
}

void test_callback_retention() {
    std::weak_ptr<const ResourceLease> resource;
    std::weak_ptr<Actions> actions;
    std::function<SubmissionCompletion()> gpu_callback;
    {
        auto owner = std::make_shared<Lease>();
        resource = owner;
        auto coordinator_actions = std::make_shared<Actions>();
        actions = coordinator_actions;
        coordinator_actions->resource = owner;
        coordinator_actions->check_resource = true;
        SubmissionCompletion completion({3, 1}, {owner}, coordinator_actions);
        auto caller_token = completion.token();
        CHECK(succeeded(completion.submitted()));
        gpu_callback = [completion]() mutable {
            CHECK(succeeded(completion.feedback({})));
            return completion;
        };
        CHECK(caller_token.valid());
    }
    // The captured handler is now the sole state owner, after caller destruction.
    CHECK(!resource.expired());
    CHECK(!actions.expired());
    auto ready = gpu_callback();
    gpu_callback = {};
    CHECK(!resource.expired());
    CHECK(actions.lock()->calls.empty());
    CHECK(ready.token().snapshot().phase == SubmissionPhase::GpuComplete);
    CHECK(succeeded(ready.finish()));
    CHECK(resource.expired());
    CHECK(actions.expired());
    CHECK(ready.token().snapshot().guest_visible_complete());
}

void test_failure_paths() {
    for (const auto failure : {ActionFailure::Resolve, ActionFailure::Prepare,
                               ActionFailure::ThrowResolve, ActionFailure::ThrowPrepare}) {
        auto resource = std::make_shared<Lease>();
        auto actions = std::make_shared<Actions>();
        actions->resource = resource;
        actions->check_resource = true;
        actions->failure = failure;
        SubmissionCompletion completion({4, 1}, {resource}, actions);
        auto token = completion.token();
        actions->observed = token;
        resource.reset();
        CHECK(succeeded(completion.submitted()));
        CHECK(succeeded(completion.feedback({})));
        CHECK(!succeeded(completion.finish()));
        CHECK(token.snapshot().phase == SubmissionPhase::Failed);
        CHECK(token.snapshot().error.has_value());
        CHECK(!token.snapshot().guest_visible_complete());
        CHECK(actions->published == 0);
        const auto expected = failure == ActionFailure::Resolve || failure == ActionFailure::ThrowResolve
            ? std::vector<int>{1} : std::vector<int>{1, 2};
        CHECK(actions->calls == expected);
        CHECK(!actions->resource.expired());
        const auto first_error = std::string(token.snapshot().error->command().detail);
        CHECK(!succeeded(completion.finish()));
        CHECK(actions->calls == expected);
        CHECK(token.snapshot().error->command().detail == first_error);
        // Recovery owns the failed allocation before explicit retirement.
        auto recovery_owner = actions->resource.lock();
        CHECK(succeeded(completion.release_failed()));
        CHECK(!actions->resource.expired());
        recovery_owner.reset();
        CHECK(actions->resource.expired());
        CHECK(!succeeded(completion.finish()));
        CHECK(token.snapshot().error->command().detail == first_error);
        CHECK(!token.snapshot().guest_visible_complete());
    }
    auto actions = std::make_shared<Actions>();
    auto resource = std::make_shared<Lease>();
    std::weak_ptr<const ResourceLease> retained = resource;
    SubmissionCompletion failed({5, 1}, {resource}, actions);
    resource.reset();
    CHECK(succeeded(failed.submitted()));
    {
        std::string temporary = "asynchronous synthetic GPU failure";
        CHECK(succeeded(failed.feedback({CommandStatus::MemoryFault, temporary, 0x5678})));
        temporary.assign(temporary.size(), 'z');
    }
    CHECK(!retained.expired());
    CHECK(!succeeded(failed.release_failed()));
    CHECK(actions->calls.empty());
    CHECK(!succeeded(failed.feedback({})));
    auto snapshot = failed.token().snapshot();
    CHECK(snapshot.phase == SubmissionPhase::Failed);
    CHECK(!snapshot.guest_visible_complete());
    CHECK(snapshot.error->command().detail == "asynchronous synthetic GPU failure");
    CHECK(snapshot.error->command().address == 0x5678);
    CHECK(!succeeded(failed.finish()));
    CHECK(!succeeded(failed.finish()));
    CHECK(!retained.expired());
    CHECK(actions->calls.empty());
    CHECK(actions->published == 0);
    CHECK(snapshot.error->command().detail == "asynchronous synthetic GPU failure");
    auto recovery_owner = retained.lock();
    CHECK(succeeded(failed.release_failed()));
    CHECK(!retained.expired());
    recovery_owner.reset();
    CHECK(retained.expired());
    CHECK(!succeeded(failed.finish()));
    CHECK(!failed.token().snapshot().guest_visible_complete());

    for (const SubmissionId id : {SubmissionId{0, 1}, SubmissionId{1, 0}}) {
        SubmissionCompletion invalid(id, {}, actions);
        CHECK(invalid.token().snapshot().phase == SubmissionPhase::Failed);
        CHECK(!succeeded(invalid.submitted()));
        CHECK(!succeeded(invalid.finish()));
        CHECK(succeeded(invalid.release_failed()));
    }
    SubmissionCompletion missing_actions({1, 1}, {}, nullptr);
    CHECK(!succeeded(missing_actions.submitted()));
    CHECK(!succeeded(missing_actions.finish()));
    CHECK(succeeded(missing_actions.release_failed()));
    auto orphaned_resource = std::make_shared<Lease>();
    SubmissionCompletion missing_recovery({1, 2}, {orphaned_resource}, nullptr);
    CHECK(!succeeded(missing_recovery.finish()));
    CHECK(!succeeded(missing_recovery.release_failed()));
}

void test_precommit_rejection() {
    auto actions = std::make_shared<Actions>();
    auto resource = std::make_shared<Lease>();
    std::weak_ptr<const ResourceLease> retained = resource;
    SubmissionCompletion completion({6, 1}, {resource}, actions);
    resource.reset();
    {
        std::string temporary = "synthetic command encoding failure";
        CHECK(succeeded(completion.reject({CommandStatus::Unsupported, temporary})));
    }
    CHECK(completion.token().snapshot().phase == SubmissionPhase::Failed);
    CHECK(!succeeded(completion.release_failed()));
    CHECK(!succeeded(completion.submitted()));
    CHECK(!succeeded(completion.feedback({})));
    CHECK(!succeeded(completion.reject({CommandStatus::Stopped, "duplicate rejection"})));
    CHECK(!succeeded(completion.finish()));
    CHECK(actions->calls.empty());
    CHECK(!retained.expired());
    CHECK(completion.token().snapshot().error->command().detail == "synthetic command encoding failure");
    auto recovery_owner = retained.lock();
    CHECK(succeeded(completion.release_failed()));
    recovery_owner.reset();
    CHECK(retained.expired());
    CHECK(!completion.token().snapshot().guest_visible_complete());
}

void test_failed_authority_handoff() {
    auto actions = std::make_shared<Actions>();
    auto resource = std::make_shared<Lease>();
    actions->resource = resource;
    actions->check_resource = true;
    SubmissionCompletion completion({7, 1}, {resource}, actions);
    actions->observed = completion.token();
    resource.reset();
    CHECK(succeeded(completion.submitted()));
    CHECK(succeeded(completion.feedback({CommandStatus::MemoryFault, "original GPU error"})));
    CHECK(!succeeded(completion.release_failed()));
    CHECK(actions->handoff_calls == 0);
    CHECK(!succeeded(completion.finish()));
    actions->refuse_handoff = true;
    CHECK(!succeeded(completion.release_failed()));
    CHECK(!actions->resource.expired());
    CHECK(actions->observed.snapshot().error->command().detail == "original GPU error");
    actions->refuse_handoff = false;
    actions->throw_handoff = true;
    CHECK(!succeeded(completion.release_failed()));
    CHECK(!actions->resource.expired());
    CHECK(actions->observed.snapshot().error->command().detail == "original GPU error");
    actions->throw_handoff = false;
    RetainedResource recovered;
    actions->handoff = [&] {
        CHECK(!succeeded(completion.release_failed()));
        recovered = actions->resource.lock();
        CHECK(bool(recovered));
    };
    CHECK(succeeded(completion.release_failed()));
    CHECK(actions->handoff_calls == 3);
    CHECK(!actions->resource.expired());
    recovered.reset();
    CHECK(actions->resource.expired());
    CHECK(succeeded(completion.release_failed()));
    CHECK(actions->handoff_calls == 3);
    CHECK(!succeeded(completion.finish()));
    CHECK(actions->observed.snapshot().error->command().detail == "original GPU error");
    CHECK(!actions->observed.snapshot().guest_visible_complete());
    CHECK(actions->published == 0 && actions->calls.empty());
}

// This backend models one synthetic event, with manual GPU feedback. It cannot
// execute real guest draws, memory accesses, dependencies, or future waits.
class SyntheticRenderer final : public Renderer {
    struct SnapshotLease final : ResourceLease {
        explicit SnapshotLease(const RenderStop& stop) : snapshot(stop) {}
        const RenderStop snapshot;
    };
    std::optional<SubmissionCompletion> pending_;
    std::shared_ptr<const SnapshotLease> snapshot_;
public:
    RendererResult<SubmissionReceipt> enqueue(RenderRequest request) override {
        if (pending_ || request.id != SubmissionId{9, 1} || !request.snapshot ||
            !request.completion || request.memory || !request.dependencies.empty() ||
            request.snapshot->kind != RenderKind::Draw || !request.snapshot->draw ||
            request.snapshot->draw->count != 3 || request.snapshot->registers[17] != 42) {
            return unsupported("mock renderer only admits its single synthetic event");
        }
        snapshot_ = std::make_shared<const SnapshotLease>(*request.snapshot);
        pending_.emplace(request.id, std::vector<RetainedResource>{snapshot_}, request.completion);
        return SubmissionReceipt{pending_->token()};
    }
    RendererStatus flush(SubmissionBoundary boundary) override {
        if (boundary != SubmissionBoundary::SubmissionEnd || !pending_)
            return unsupported("mock renderer boundary is not implemented");
        return pending_->submitted();
    }
    RendererStatus poll() override {
        if (!pending_) return renderer_success();
        const auto phase = pending_->token().snapshot().phase;
        if (phase == SubmissionPhase::Accepted || phase == SubmissionPhase::Submitted)
            return renderer_success();
        return pending_->finish();
    }
    RendererStatus drain(std::uint64_t deadline) override {
        if (deadline != 0) return unsupported("mock renderer has no timed wait implementation");
        auto result = poll();
        if (!succeeded(result)) return result;
        if (pending_ && !pending_->token().snapshot().guest_visible_complete())
            return RendererError({CommandStatus::Stopped, "synthetic drain deadline expired"});
        return renderer_success();
    }
    RendererStatus gpu_feedback() {
        if (!pending_) return unsupported("mock renderer has no accepted event");
        return pending_->feedback({});
    }
    const RenderStop& accepted_snapshot() const { return snapshot_->snapshot; }
};

void test_owned_snapshot_fallback() {
    SyntheticRenderer renderer;
    Renderer& interface = renderer;
    auto effects = std::make_shared<Actions>();
    CHECK(std::holds_alternative<RendererError>(interface.enqueue_owned(
        {{9, 1}, {}, {}, {}, effects}, {})));
    auto source = std::make_unique<RenderStop>();
    source->kind = RenderKind::Draw;
    source->draw = DrawDescription{};
    source->draw->count = 3;
    source->registers[17] = 42;
    source->constant_buffers[2][3] = {0x2400, 64, true};
    source->inline_indices = {2, 1, 0};
    const auto accepted = interface.enqueue_owned(
        {{9, 1}, {}, {}, {}, effects}, std::move(source));
    CHECK(!source);
    CHECK(std::holds_alternative<SubmissionReceipt>(accepted));
    if (!std::holds_alternative<SubmissionReceipt>(accepted)) return;
    CHECK(renderer.accepted_snapshot().registers[17] == 42);
    CHECK(renderer.accepted_snapshot().constant_buffers[2][3].address == 0x2400);
    CHECK((renderer.accepted_snapshot().inline_indices == std::vector<std::uint32_t>{2, 1, 0}));
    const auto token = std::get<SubmissionReceipt>(accepted).completion;
    CHECK(token.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(succeeded(interface.flush(SubmissionBoundary::SubmissionEnd)));
    CHECK(succeeded(renderer.gpu_feedback()));
    CHECK(succeeded(interface.drain(0)));
    CHECK(token.snapshot().guest_visible_complete());
    CHECK(effects->published == 1);
}

void test_renderer_admission_and_timeout() {
    SyntheticRenderer renderer;
    Renderer& interface = renderer;
    CHECK(std::holds_alternative<RendererError>(interface.enqueue({})));
    CHECK(!succeeded(interface.flush(SubmissionBoundary::Capture)));
    auto source = std::make_shared<RenderStop>();
    source->kind = RenderKind::Draw;
    source->draw = DrawDescription{};
    source->draw->count = 3;
    source->registers[17] = 42;
    source->constant_buffers[2][3] = {0x2400, 64, true};
    auto actions = std::make_shared<Actions>();
    RenderRequest request{{9, 1}, source, {}, {}, actions};
    const auto accepted = interface.enqueue(request);
    CHECK(std::holds_alternative<SubmissionReceipt>(accepted));
    if (!std::holds_alternative<SubmissionReceipt>(accepted)) return;
    const auto token = std::get<SubmissionReceipt>(accepted).completion;
    source->registers[17] = 99;
    source->draw->count = 8;
    source->constant_buffers[2][3].address = 0x9800;
    source.reset();
    request.snapshot.reset();
    CHECK(renderer.accepted_snapshot().registers[17] == 42);
    CHECK(renderer.accepted_snapshot().draw->count == 3);
    CHECK(renderer.accepted_snapshot().constant_buffers[2][3].address == 0x2400);
    CHECK(token.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(succeeded(interface.poll()));
    CHECK(!succeeded(interface.drain(0)));
    CHECK(token.snapshot().phase == SubmissionPhase::Accepted);
    CHECK(succeeded(interface.flush(SubmissionBoundary::SubmissionEnd)));
    const auto timeout = interface.drain(0);
    CHECK(std::holds_alternative<RendererError>(timeout));
    CHECK(std::get<RendererError>(timeout).command().status == CommandStatus::Stopped);
    CHECK(token.snapshot().phase == SubmissionPhase::Submitted);
    CHECK(actions->calls.empty());
    CHECK(!token.snapshot().guest_visible_complete());
    CHECK(succeeded(renderer.gpu_feedback()));
    CHECK(token.snapshot().phase == SubmissionPhase::GpuComplete);
    CHECK(actions->calls.empty());
    CHECK(succeeded(interface.poll()));
    CHECK(succeeded(interface.drain(0)));
    CHECK(token.snapshot().guest_visible_complete());
    CHECK(actions->published == 1);
    CHECK(!succeeded(interface.drain(1)));
    CHECK(std::holds_alternative<RendererError>(interface.enqueue(request)));
}

// Compiles the L2/L3 seam without claiming an implemented backing registry.
// Every operation that would require storage or coherence fails closed.
class UnsupportedResourceMemory final : public ResourceMemory {
    GpuAddressSpace memory_ = std::get<GpuAddressSpace>(make_address_space({}, {}));
public:
    RendererStatus register_backing(BackingRegistration) override { return {}; }
    RendererStatus register_mapping(MappingRegistration) override { return {}; }
    RendererStatus unmap(MappingId) override { return {}; }
    RendererStatus retire_backing(BackingId) override { return {}; }
    RendererResult<RetainedRange> retain(std::uint64_t, std::uint64_t,
                                        ResourceAccess) override { return {}; }
    const GpuAddressSpace& address_space() const noexcept override { return memory_; }
    CoherenceMode coherence_mode() const noexcept override {
        return CoherenceMode::SubmissionBoundary;
    }
    RendererResult<std::vector<DirtyRange>> inspect(BackingRange) override { return {}; }
    RendererStatus resolve(BackingRange, ResolveReason) override { return {}; }
    RendererResult<std::unique_ptr<CpuReadTransaction>> prepare_cpu_read(
        BackingRange, ResolveReason) override { return {}; }
    RendererResult<std::unique_ptr<CpuWriteTransaction>> prepare_cpu_write(
        BackingRange) override { return {}; }
    RendererResult<std::shared_ptr<GpuAccessReservation>> reserve_accesses(
        std::span<const GpuAccessRequest>, CompletionToken) override { return {}; }
};

void test_portable_keys_and_values() {
    UnsupportedResourceMemory memory;
    CHECK(memory.coherence_mode() == CoherenceMode::SubmissionBoundary);
    CHECK(std::holds_alternative<RendererError>(memory.retain(0x1000, 16, ResourceAccess::Read)));
    CHECK(std::holds_alternative<RendererError>(memory.prepare_cpu_read({}, ResolveReason::CpuRead)));
    CHECK(std::holds_alternative<RendererError>(memory.prepare_cpu_write({})));
    CHECK(std::holds_alternative<RendererError>(memory.reserve_accesses({}, {})));
    static_assert(std::is_abstract_v<Renderer> && std::is_abstract_v<ResourceMemory>);
    static_assert(std::is_abstract_v<TextureCache> && std::is_abstract_v<BufferCache>);
    TextureStorageKey storage;
    storage.bytes = {{2, 3, 1}, 32, 512};
    storage.descriptor.address = 0x4000;
    storage.descriptor.backing_extent = 512;
    storage.descriptor.width = 8;
    storage.descriptor.height = 8;
    storage.descriptor.depth = 1;
    storage.descriptor.levels = 2;
    storage.descriptor.layers = 1;
    storage.descriptor.samples = 1;
    auto alias = storage;
    alias.descriptor.address = 0x9000;
    CHECK(alias == storage);
    alias.descriptor.raw_words[0] = 17;
    alias.descriptor.normalized_coordinates = false;
    alias.descriptor.swizzle[0] = TextureSwizzle::Zero;
    CHECK(alias == storage);
    auto changed = storage;
    ++changed.bytes.backing.generation;
    CHECK(changed != storage);
    changed = storage;
    ++changed.bytes.offset;
    CHECK(changed != storage);
    changed = storage;
    ++changed.descriptor.width;
    CHECK(changed != storage);
    changed = storage;
    ++changed.descriptor.layer_stride;
    CHECK(changed != storage);
    changed = storage;
    changed.descriptor.layout = TextureLayout::Pitch;
    CHECK(changed != storage);
    TextureViewKey view{storage};
    auto other_view = view;
    other_view.storage = alias;
    CHECK(view == other_view);
    other_view.first_level = 1;
    CHECK(view != other_view);
    other_view = view;
    other_view.first_layer = 1;
    CHECK(view != other_view);
    other_view = view;
    other_view.aspect = TextureAspect::Depth;
    CHECK(view != other_view);
    other_view = view;
    other_view.swizzle[0] = TextureSwizzle::Zero;
    CHECK(view != other_view);
    const BufferKey buffer{storage.bytes};
    auto other_buffer = buffer;
    ++other_buffer.bytes.backing.allocation;
    CHECK(buffer != other_buffer);
    other_buffer = buffer;
    ++other_buffer.bytes.backing.generation;
    CHECK(buffer != other_buffer);
    other_buffer = buffer;
    ++other_buffer.bytes.size;
    CHECK(buffer != other_buffer);
    CHECK(BufferKey{alias.bytes} == buffer);

    RenderPassKey pass;
    pass.colors[3] = view;
    pass.area = {-2, 1, 8, 4};
    auto other_pass = pass;
    CHECK(pass == other_pass);
    other_pass.colors[3].reset();
    other_pass.colors[2] = view;
    CHECK(pass != other_pass);
    other_pass = pass;
    ++other_pass.area.x;
    CHECK(pass != other_pass);
    RenderTargetPlan targets;
    targets.colors[3] = AttachmentPlan{};
    targets.colors[3]->physical_target = 6;
    CHECK(!targets.colors[0] && targets.colors[3]->physical_target == 6);
    ClearPlan clear;
    clear.color_mask = 5;
    CHECK(!clear.fold_into_load && clear.color_mask == 5);
    RenderState state;
    state.scissors.push_back({-4, 2, 0, 8});
    CHECK(state.scissors.front().width == 0);
    DrawPlan draw;
    draw.indices = IndexPlan{};
    draw.indices->converted = {std::byte{1}, std::byte{0}};
    draw.segments.push_back({2, 3, -1, 4, 2});
    auto draw_copy = draw;
    draw.indices->converted[0] = std::byte{9};
    CHECK(draw_copy.indices->converted.front() == std::byte{1});
    CHECK(draw_copy.segments.front().base_vertex == -1);
    CHECK(ResourceVersion{1, 2} != ResourceVersion{1, 3});
    CHECK(MappingId{1, 2, 3} != MappingId{1, 2, 4});
    TextureRequest texture_request{{{}, storage.bytes, {}}, view, ResourceAccess::Read};
    BufferRequest buffer_request{{{}, storage.bytes, {}}, BufferUse::Uniform, ResourceAccess::Read, 16};
    CHECK(texture_request.range.bytes == buffer_request.range.bytes);
    ResidentTexture resident_texture{view, {1, 2}, {}, {}};
    ResidentBuffer resident_buffer{buffer, {1, 2}, {}, 0, 512, {}};
    CHECK(resident_texture.uploaded_version == resident_buffer.uploaded_version);
}
} // namespace

int main() {
    test_pass_access_summary();
    test_owned_errors();
    test_submission_phases();
    test_callback_retention();
    test_failure_paths();
    test_precommit_rejection();
    test_failed_authority_handoff();
    test_owned_snapshot_fallback();
    test_renderer_admission_and_timeout();
    test_portable_keys_and_values();
    std::printf("Renderer contract failures: %u\n", failures);
    return failures == 0 ? 0 : 1;
}
