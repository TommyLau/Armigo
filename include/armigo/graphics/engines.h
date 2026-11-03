// ============================================================================
// Armigo Graphics - GPU Engine Front End
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Begin instance state, compact draw forms, and owned inline index snapshots
// are captured by Armigo's bounded graphics front end.

#pragma once

#include <armigo/graphics/channel.h>
#include <armigo/graphics/upload.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace armigo::graphics {

struct RenderStop;

// Owned publication only: captures must outlive submission and must not retain
// borrowed driver or memory callbacks. A busy serialization lock returns nullopt;
// nullopt and errors publish nothing. Success is an infallible commit of the
// validated effect, exactly once.
using GuestPublication = std::function<std::optional<CommandResult>()>;
struct GuestEffect {
    std::uint64_t address = 0;
    std::array<std::byte, 16> payload{};
    std::uint32_t size = 0;
    GuestPublication publish{};
    // A semaphore acquire checked in stream order instead of waited for: once
    // the preceding work is guest-visible, the word at `address` must equal
    // `value` (operation 1), be at least it (4) or share a bit with it (8).
    struct Acquire {
        std::uint32_t value = 0;
        std::uint32_t operation = 1;
    };
    std::optional<Acquire> acquire;
};

// Feature test for clients that construct RenderCallbacks positionally.
#define ARMIGO_RENDER_CALLBACKS_HAVE_COHERENCE 1
#define ARMIGO_RENDER_CALLBACKS_HAVE_BLIT 1
#define ARMIGO_RENDER_CALLBACKS_HAVE_RESOLVE_GUEST 1
#define ARMIGO_RENDER_CALLBACKS_HAVE_ORDERED_EFFECT 1
#define ARMIGO_RENDER_CALLBACKS_HAVE_READ_COHERENCE 1
#define ARMIGO_RENDER_CALLBACKS_HAVE_STATUS 1
#define ARMIGO_RENDER_CALLBACKS_HAVE_ORDERED_WRITE 1

struct RenderCallbacks {
    // Success means GPU execution has completed, or an owned backend queue
    // has accepted it when ordered_effect is provided. Guest-memory coherence has
    // also completed unless the backend provides coherence (deferred
    // write-back), in which case coherence() publishes the results.
    // Inputs and memory callbacks are borrowed only for this synchronous call.
    // Returned diagnostic text must remain valid for the channel lifetime.
    // The context outlives registration across all client clones; the callback
    // must not re-enter its driver client while the submission lock is held.
    CommandResult (*clear)(void*, const RenderStop&, const GpuAddressSpace&) noexcept = nullptr;
    void* context = nullptr;
    // Optional event-latched draw path, with the same synchronous completion
    // and borrowing contract as clear. Kept after the original aggregate fields.
    CommandResult (*draw)(void*, const RenderStop&, const GpuAddressSpace&) noexcept = nullptr;
    // Optional ownership for context, retained by driver clones and sticky
    // channel errors. This does not extend the borrowed memory callback scope.
    std::shared_ptr<void> owner{};
    // Called before driver mapping removal/replacement or final backing release.
    // Removed mappings and released allocation owners are exact old identities.
    // Released allocations are still pinned and CPU-accessible during this call,
    // including exports; preserve their bytes before releasing renderer state.
    // Failure prevents publication. Callbacks are borrowed and must not reenter
    // the driver. Empty removals may notify a final unmapped allocation release.
    CommandResult (*memory_changed)(void*, std::span<const Mapping>,
        std::span<const std::shared_ptr<void>>, MemoryCallbacks) noexcept = nullptr;
    // Optional. A backend that defers writing GPU results back to guest memory
    // provides this; the front end calls it after rendering and before any
    // guest-visible effect (semaphore, report or syncpoint publication) or its
    // own guest-memory access. Same borrowing contract as clear.
    // A nonzero size limits the write-back to that GPU address range (the
    // front end is about to write it); size zero makes all prior GPU results
    // guest-visible.
    CommandResult (*coherence)(void*, const GpuAddressSpace&, std::uint64_t gpu_address,
                               std::uint64_t size) noexcept = nullptr;
    // Optional compute launch path with the same synchronous completion and
    // borrowing contract as draw. Registered last so positional aggregate
    // initializers written before compute dispatch keep their meaning.
    CommandResult (*dispatch)(void*, const RenderStop&, const GpuAddressSpace&) noexcept = nullptr;
    // Optional 2D blit (engine 0x902d trigger 0x237), with the same
    // synchronous completion and borrowing contract as clear. Kept after
    // dispatch so positional initializers keep their meaning.
    CommandResult (*blit)(void*, const RenderStop&, const GpuAddressSpace&) noexcept = nullptr;
    // Optional, for backends that keep GPU results resident when guest
    // memory is host-addressable: completes accepted work and writes back
    // GPU-newer bytes of the guest-address range, so a consumer outside the
    // command stream (a display reading a presented surface) observes them.
    CommandResult (*resolve_guest)(void*, std::uint64_t guest_address, std::uint64_t size) noexcept = nullptr;
    // Owns the effect and pins its memory view before returning. Later render
    // preparation and encoding may pass an effect while native commits wait
    // for its prefix writeback/publication. Dependencies on effect stores and
    // barriers remain ordered; coherence finishes pending effects before any
    // front-end memory access that depends on them.
    CommandResult (*ordered_effect)(void*, const GpuAddressSpace&, GuestEffect) noexcept = nullptr;
    // Optional read-only coherence for command fetches. Finishes prior ordered
    // writes/effects and resolves GPU results overlapping the range, without preparing
    // a host store or invalidating clean CPU write-generation certificates.
    // Without this callback, command fetches use a full coherence boundary.
    CommandResult (*read_coherence)(void*, const GpuAddressSpace&, std::uint64_t gpu_address,
                                   std::uint64_t size) noexcept = nullptr;
    // Optional nonblocking status query. Threaded backends expose failures that
    // surfaced after an asynchronous callback returned; success means no error
    // has been observed yet and does not imply that accepted work completed.
    CommandResult (*status)(void*) noexcept = nullptr;
    // Optional ordered front-end write path. The backend copies and owns the
    // payload and pins its memory view before returning; borrowed memory
    // callbacks are not retained. Accepted writes execute in FIFO order with
    // rendering and ordered effects. Front-end reads synchronize them through
    // read_coherence when present, or through a full coherence boundary.
    CommandResult (*ordered_write)(void*, const GpuAddressSpace&, std::uint64_t gpu_address,
                                   std::span<const std::byte>) noexcept = nullptr;
};

// Feature test for clients that construct RenderCallbacks positionally with a
// trailing dispatch entry.
#define ARMIGO_RENDER_CALLBACKS_HAVE_DISPATCH 1

// One constant buffer published by a compute queue-meta-data descriptor.
struct ComputeConstantBuffer {
    std::uint64_t address = 0;
    std::uint32_t size = 0;
    bool enabled = false;
};

// The launch fields Armigo consumes from a compute queue-meta-data descriptor.
// Grid dimensions select dispatched threadgroups; block dimensions select the
// threads of one threadgroup. Populated by decode_compute_qmd.
struct ComputeLaunch {
    std::array<std::uint32_t, 3> grid{};
    std::array<std::uint32_t, 3> block{};
    std::uint32_t shared_memory_size = 0;
    std::uint32_t local_memory_size = 0;
    std::uint32_t local_crs_size = 0;
    std::uint32_t barrier_count = 0;
    std::uint32_t register_count = 0;
    std::uint32_t program_offset = 0;
    bool sampler_index = false;
    std::array<ComputeConstantBuffer, 8> constant_buffers{};
};

struct CompletionCallbacks {
    bool (*increment)(void*, std::uint32_t) noexcept = nullptr;
    bool (*expired)(void*, std::uint32_t, std::uint32_t) noexcept = nullptr;
    std::uint64_t (*ticks)(void*) noexcept = nullptr;
    void* context = nullptr;
    RenderCallbacks rendering{};
    // Optional factory for a publication that is safe beyond this call. The
    // backend may execute it on its pump or synchronously on the caller.
    GuestPublication (*retain_increment)(void*, std::uint32_t, std::uint32_t) = nullptr;
};

enum class RenderKind : std::uint8_t { Clear, Draw, Compute, Blit };

struct TargetDescription {
    std::uint64_t address = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0;
    std::uint32_t tile_mode = 0;
    std::uint32_t depth = 0;
    std::uint32_t array_pitch = 0;
    std::uint32_t base_layer = 0;
};

struct DepthDescription {
    std::uint64_t address = 0;
    std::uint32_t format = 0;
    std::uint32_t tile_mode = 0;
    std::uint32_t array_pitch = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool enabled = false;
};

struct ProgramDescription {
    std::uint64_t address = 0;
    std::uint32_t offset = 0;
    std::uint32_t control = 0;
    bool enabled = false;
};

struct ConstantBufferDescription {
    std::uint64_t address = 0;
    std::uint32_t size = 0;
    bool enabled = false;
};

// Vertex (both programs), tessellation control/evaluation, geometry, fragment.
using ConstantBufferBindings = std::array<std::array<ConstantBufferDescription, 18>, 5>;

struct DrawDescription {
    bool indexed = false;
    std::uint32_t topology = 4; // Triangle list in the supported Begin/End form.
    // Index-buffer element offset for indexed draws, vertex number for arrays.
    std::uint32_t first = 0;
    std::uint32_t count = 0;
    // Applied to decoded indices only; array first already selects its vertices.
    std::int32_t base_vertex = 0;
    // Original guest base B and continuation index i remain distinct.
    std::uint32_t base_instance = 0;
    std::uint32_t instance_count = 1;
    std::uint32_t instance_index = 0;
};

// 3D register groups a backend may key derived draw state by. A register write
// marks its group only when it
// changes the stored value; constant-buffer binds mark their stage's group.
// Registers outside every group (draw parameters, constant-buffer selection
// and data, macro scratch) never invalidate derived state.
namespace threed_dirty {
enum Group : std::uint32_t {
    RenderTargets,
    Shaders,
    VertexStreams,
    VertexAttributes,
    IndexBuffer,
    TexturePools,
    ConstantBuffers0,
    ConstantBuffersLast = ConstantBuffers0 + 4,
    // Every other rendering register: pipeline, raster and output state.
    State,
    Count
};
constexpr std::uint64_t bit(std::uint32_t group) noexcept { return std::uint64_t{1} << group; }
inline constexpr std::uint64_t all = bit(Count) - 1;
} // namespace threed_dirty

struct RenderStop {
    MethodCall original{};
    std::uint32_t engine = 0;
    RenderKind kind = RenderKind::Draw;
    std::array<std::uint32_t, 0x1000> registers{};
    // Changed words since the immediately preceding stop of this stream,
    // including clear, compute and blit stops. Apply only to an exact serial
    // successor; zero identities and gaps require the full register snapshot.
    // First stops, engine switches and non-3D stops mark every word.
    std::array<std::uint64_t, 0x1000 / 64> changed_registers{};
    std::uint64_t register_stream = 0, register_serial = 0;
    // Effective condition starts enabled independently of the raw method word.
    std::uint32_t effective_render_enable_mode = 1;
    std::array<TargetDescription, 8> color_targets{};
    DepthDescription depth{};
    std::array<ProgramDescription, 6> programs{};
    ConstantBufferBindings constant_buffers{};
    // Groups changed since this front end's previous draw stop. Derived state
    // from that draw is reusable only by a backend that executed it, which
    // state_serial - 1 of the same state_stream identifies.
    std::uint64_t dirty_groups = threed_dirty::all;
    std::uint64_t state_stream = 0, state_serial = 0;
    // Present for validated normal Begin/End and compact draw triggers. A bare
    // End retains raw state without authorizing dispatch.
    std::optional<DrawDescription> draw;
    // Raw zero-extended inline indices owned by this stopped operation. A
    // nonempty stream replaces the guest index-buffer source for this draw.
    std::vector<std::uint32_t> inline_indices;
    // Compute dispatches own the complete QMD snapshot. No guest-memory or
    // callback lifetime is retained after call() returns.
    std::uint64_t compute_descriptor_address = 0;
    std::array<std::byte, 0x100> compute_descriptor{};
    bool has_compute_descriptor = false;
    // The decoded queue-meta-data launch, valid exactly when
    // has_compute_descriptor holds.
    std::optional<ComputeLaunch> compute;
    // Copy every field except registers; reuse inline-index allocation capacity.
    void copy_metadata(const RenderStop& source);
};

// A pending rendering stop that keeps its storage between draws, so stopping
// does not clear and reconstruct the whole register snapshot each time.
class RenderStopSlot {
  public:
    [[nodiscard]] bool has_value() const noexcept { return engaged_; }
    explicit operator bool() const noexcept { return engaged_; }
    [[nodiscard]] RenderStop& operator*() noexcept { return value_; }
    [[nodiscard]] const RenderStop& operator*() const noexcept { return value_; }
    [[nodiscard]] RenderStop* operator->() noexcept { return &value_; }
    [[nodiscard]] const RenderStop* operator->() const noexcept { return &value_; }
    void reset() noexcept { engaged_ = false; }
    // Engages the slot with the previous stop's contents; the caller
    // overwrites or resets every field.
    RenderStop& recycle() noexcept
    {
        engaged_ = true;
        return value_;
    }

  private:
    RenderStop value_{};
    bool engaged_ = false;
};

class EngineFrontEnd {
  public:
    EngineFrontEnd() noexcept;

    static constexpr std::size_t macro_code_words = 0x10000;
    static constexpr std::size_t macro_entries = 0x80;

    [[nodiscard]] CommandResult call(const MethodCall& call,
                                     std::uint32_t engine,
                                     const GpuAddressSpace& memory,
                                     CompletionCallbacks completion) noexcept;

    [[nodiscard]] const RenderStopSlot& rendering_stop() const noexcept
    {
        return rendering_stop_;
    }

    // Writes deferred GPU results back to guest memory before the caller
    // publishes completion (for example an automatic submission fence).
    [[nodiscard]] CommandResult make_coherent(const GpuAddressSpace& memory,
                                              const CompletionCallbacks& completion) noexcept
    {
        return ensure_coherent(memory, completion);
    }

    [[nodiscard]] CommandResult increment_syncpoint(const GpuAddressSpace& memory,
        const CompletionCallbacks& completion, std::uint32_t id, std::uint32_t count = 1) noexcept;

    [[nodiscard]] bool has_pending_memory_work() const noexcept
    {
        return threed_upload_.blocked() || compute_upload_.blocked() || inline_upload_.blocked();
    }

  private:
    [[nodiscard]] CommandResult call_puller(const MethodCall& call,
                                            const GpuAddressSpace& memory,
                                            CompletionCallbacks completion) noexcept;
    [[nodiscard]] CommandResult call_threed(const MethodCall& call,
                                            const GpuAddressSpace& memory,
                                            CompletionCallbacks completion,
                                            bool from_macro) noexcept;
    [[nodiscard]] CommandResult execute_pending_macro(const MethodCall& call,
                                                      const GpuAddressSpace& memory,
                                                      CompletionCallbacks completion) noexcept;
    [[nodiscard]] CommandResult
    stop_rendering(const MethodCall& call, std::uint32_t engine, RenderKind kind) noexcept;
    [[nodiscard]] CommandResult draw_end(const MethodCall& call,
                                         const GpuAddressSpace& memory,
                                         CompletionCallbacks completion) noexcept;
    [[nodiscard]] CommandResult draw_begin(const MethodCall& call,
                                           std::uint32_t argument) noexcept;
    [[nodiscard]] CommandResult compact_draw(const MethodCall& call,
                                             const GpuAddressSpace& memory,
                                             CompletionCallbacks completion,
                                             std::uint32_t argument,
                                             bool indexed,
                                             bool subsequent) noexcept;
    [[nodiscard]] CommandResult dispatch_draw(const MethodCall& call,
                                              const GpuAddressSpace& memory,
                                              CompletionCallbacks completion,
                                              DrawDescription draw,
                                              bool continuation) noexcept;
    [[nodiscard]] CommandResult append_inline_indices(const MethodCall& call,
                                                      std::span<const std::uint32_t> indices) noexcept;
    [[nodiscard]] CommandResult call_upload(InlineUpload& upload,
                                            const MethodCall& call,
                                            const GpuAddressSpace& memory,
                                            const CompletionCallbacks& completion,
                                            std::uint32_t engine) noexcept;
    // DMA copy class launch (method 0xc0): the copy, then its semaphore.
    [[nodiscard]] CommandResult call_dma(const MethodCall& call,
                                         const GpuAddressSpace& memory,
                                         const CompletionCallbacks& completion,
                                         std::uint32_t engine) noexcept;

    std::array<std::uint32_t, 0x40> puller_registers_{};
    std::array<std::uint32_t, 0x1000> threed_registers_{};
    std::array<std::uint32_t, 0x1000> threed_shadow_{};
    ConstantBufferBindings constant_buffers_{};
    std::array<std::uint32_t, 0x1000> compute_registers_{};
    std::array<std::uint32_t, 0x1000> twod_registers_{};
    std::array<std::uint32_t, 0x1000> inline_registers_{};
    std::array<std::uint32_t, 0x1000> dma_registers_{};
    InlineUpload threed_upload_;
    InlineUpload compute_upload_;
    InlineUpload inline_upload_;
    // A rejected ordered upload remains blocked and repeats the backend's
    // exact diagnostic if another data word is presented.
    std::optional<CommandResult> threed_upload_failure_;
    std::optional<CommandResult> compute_upload_failure_;
    std::optional<CommandResult> inline_upload_failure_;
    std::array<std::uint32_t, macro_code_words> macro_code_{};
    std::array<bool, macro_code_words> macro_valid_{};
    // End of the valid code run from each entry, cached until code changes
    // (bumped on every uploaded word); avoids rescanning per invocation.
    std::uint64_t macro_code_version_ = 1;
    std::array<std::pair<std::uint64_t, std::uint32_t>, macro_entries> macro_end_{};
    std::array<std::uint32_t, macro_entries> macro_entry_{};
    std::array<bool, macro_entries> macro_entry_valid_{};
    std::uint32_t macro_code_pointer_ = 0;
    std::uint32_t macro_entry_pointer_ = 0;
    std::uint32_t effective_render_enable_mode_ = 1;
    std::uint32_t shadow_mode_ = 0;
    std::uint32_t pending_macro_method_ = 0;
    // Index count writes select the next normal draw, independently of the
    // stored count. Begin and vertex count writes do not reset this latch.
    bool draw_indexed_ = false;
    std::optional<std::uint32_t> draw_begin_;
    std::uint32_t draw_instance_ = 0;
    bool draw_instance_continuation_ = false;
    bool inline_draw_consumed_ = false;
    std::vector<std::uint32_t> inline_indices_;
    std::vector<std::uint32_t> pending_macro_arguments_;
    RenderStopSlot rendering_stop_;
    // 3D registers written since the last 3D stop copied them, and the engine
    // whose registers the recycled stop holds (none before the first stop).
    std::array<std::uint64_t, 0x1000 / 64> changed_registers_{};
    std::uint32_t copied_engine_ = ~0u;
    void mark_register(std::uint32_t method) noexcept
    {
        changed_registers_[method / 64] |= std::uint64_t{1} << (method % 64);
    }
    // Register groups changed since the last draw stop (threed_dirty), the
    // front end's unique stream and its draw stop count.
    std::uint64_t dirty_groups_ = threed_dirty::all;
    std::uint64_t state_stream_ = 0, state_serial_ = 0;
    // Every stop advances this sequence, independently of draw-state reuse.
    // Exhaustion permanently disables register deltas for this front end.
    std::uint64_t register_serial_ = 0;
    // A halted decoder may still borrow the failed backend's diagnostic.
    std::shared_ptr<void> rendering_owner_;
    // Set after rendering or an ordered write is accepted; cleared by a full
    // ensure_coherent() boundary.
    bool rendering_since_coherence_ = false;
    [[nodiscard]] CommandResult acquire_semaphore(const GpuAddressSpace& memory,
        const CompletionCallbacks& completion, std::uint64_t semaphore, std::uint32_t value,
        std::uint32_t operation) noexcept;
    [[nodiscard]] CommandResult release_semaphore(const GpuAddressSpace& memory,
        const CompletionCallbacks& completion, std::uint64_t address,
        std::span<const std::byte> payload) noexcept;
    [[nodiscard]] CommandResult ensure_coherent(const GpuAddressSpace& memory,
                                                const CompletionCallbacks& completion,
                                                std::uint64_t gpu_address = 0,
                                                std::uint64_t size = 0) noexcept;
    // Makes GPU results and queued guest effects overlapping [gpu_address,
    // gpu_address + size) visible to a front-end read. Uses the backend's
    // read-only coherence when present, else a full coherence boundary.
    [[nodiscard]] CommandResult ensure_readable(const GpuAddressSpace& memory,
                                               const CompletionCallbacks& completion,
                                               std::uint64_t gpu_address,
                                               std::uint64_t size) noexcept;
};

}  // namespace armigo::graphics
