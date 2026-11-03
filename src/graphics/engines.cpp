// ============================================================================
// Armigo Graphics - GPU Engine Front End
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include <armigo/common/diagnostics.h>
// The 3D engine records Begin instance state, compact draw forms, and inline
// indices through bounded synchronous callbacks. Only value-changing register
// writes mark their derived-state group dirty. The 2D engine accumulates its
// register file and dispatches at PixelsFromMemorySrcY0Int (method 0x237).
// Compute dispatch consumes the queue metadata descriptor named by SendPcasA
// through Armigo's decoded ComputeLaunch contract.

#include <armigo/graphics/engines.h>
#include <armigo/graphics/dma_copy.h>
#include <armigo/graphics/macro.h>
#include "compute_qmd.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <new>
#include <limits>
#include <utility>
#include <span>

namespace armigo::graphics {
namespace {

constexpr std::size_t maximum_inline_indices = 1u << 20;

constexpr CommandResult result(CommandStatus status,
                               std::string_view detail,
                               const MethodCall& call,
                               std::uint32_t engine) noexcept
{
    return {status, detail, call.address, 0, call.subchannel, call.method, call.argument, engine};
}

constexpr std::uint64_t address(std::uint32_t high, std::uint32_t low) noexcept
{
    return (static_cast<std::uint64_t>(high) << 32) | low;
}

std::array<std::byte, 4> word_bytes(std::uint32_t value) noexcept
{
    return {static_cast<std::byte>(value),
            static_cast<std::byte>(value >> 8),
            static_cast<std::byte>(value >> 16),
            static_cast<std::byte>(value >> 24)};
}

void store_qword(std::span<std::byte, 8> output, std::uint64_t value) noexcept
{
    for (unsigned index = 0; index < output.size(); ++index) {
        output[index] = static_cast<std::byte>(value >> (index * 8));
    }
}

std::uint32_t read_word(std::span<const std::byte, 4> bytes) noexcept
{
    return std::uint32_t{std::to_integer<unsigned char>(bytes[0])} |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[1])} << 8) |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[2])} << 16) |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[3])} << 24);
}

CommandResult memory_failure(const MemoryResult& memory_result,
                             const MethodCall& call,
                             std::uint32_t engine) noexcept
{
    return result(CommandStatus::MemoryFault,
                  memory_result.detail.empty() ? "GPU engine memory access failed"
                                               : memory_result.detail,
                  call,
                  engine);
}

// Register-to-group table of 3D state. Entries hold group + 1; zero
// marks registers that never invalidate derived draw state. Indices are 3D
// methods (word offsets); names follow the Maxwell 3D class layout.
constexpr std::array<std::uint8_t, 0x1000> make_dirty_groups() noexcept
{
    std::array<std::uint8_t, 0x1000> table{};
    const auto fill = [&](std::size_t first, std::size_t count, std::uint32_t group) {
        for (std::size_t index = first; index != first + count; ++index)
            table[index] = static_cast<std::uint8_t>(group + 1);
    };
    // Every class register from the first shadowed method through the shader
    // scratch area is rendering state unless a narrower group claims it.
    fill(0x80, 0xd00 - 0x80, threed_dirty::State);
    fill(0x200, 0x80, threed_dirty::RenderTargets);          // rt[8]
    fill(0x3f8, 5, threed_dirty::RenderTargets);             // zeta
    fill(0x487, 1, threed_dirty::RenderTargets);             // rt_control
    fill(0x48a, 2, threed_dirty::RenderTargets);             // zeta_size
    fill(0x54e, 1, threed_dirty::RenderTargets);             // zeta_enable
    fill(0x458, 0x20, threed_dirty::VertexAttributes);       // vertex_attrib_format
    fill(0x620, 0x10, threed_dirty::VertexStreams);          // vertex_stream_instances
    fill(0x700, 0x40, threed_dirty::VertexStreams);          // vertex_streams
    fill(0x7c0, 0x20, threed_dirty::VertexStreams);          // vertex_stream_limits
    fill(0x5f2, 5, threed_dirty::IndexBuffer);               // index_buffer address, limit, format
    fill(0x557, 3, threed_dirty::TexturePools);              // tex_sampler
    fill(0x55d, 3, threed_dirty::TexturePools);              // tex_header
    fill(0x982, 1, threed_dirty::TexturePools);              // bindless_texture_const_buffer_slot
    fill(0x582, 2, threed_dirty::Shaders);                   // program_region
    fill(0x800, 0x60, threed_dirty::Shaders);                // pipelines[6]
    // Draw parameters reach the backend through RenderStop::draw on every
    // draw: vertex_buffer first/count, vertex_id_base, instance and base
    // indices, draw begin/end, index first/count and compact draw triggers.
    for (const std::size_t method : {0x35dU, 0x35eU, 0x42bU, 0x446U, 0x485U, 0x486U, 0x50dU,
                                     0x50eU, 0x585U, 0x586U})
        table[method] = 0;
    fill(0x5f7, 8, threed_dirty::Count);
    // Inline uploads, inline indices, semaphores and constant-buffer
    // selection/data are consumed when written; binds mark their stage.
    fill(0x60, 0xe, threed_dirty::Count);
    for (const std::size_t method : {0x4c1U, 0x57aU, 0x57cU}) table[method] = 0;
    fill(0x6c0, 4, threed_dirty::Count);
    fill(0x8e0, 0x14, threed_dirty::Count);
    fill(0x904, 0x28, threed_dirty::Count);
    for (auto& entry : table)
        if (entry == threed_dirty::Count + 1) entry = 0;
    return table;
}
constexpr auto dirty_groups = make_dirty_groups();

std::uint64_t next_state_stream() noexcept
{
    static std::atomic<std::uint64_t> next{0};
    auto value = next.load(std::memory_order_relaxed);
    while (value != std::numeric_limits<std::uint64_t>::max()) {
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
            return value + 1;
    }
    // Never reuse an identity after the process-wide counter is exhausted.
    return 0;
}

}  // namespace

void RenderStop::copy_metadata(const RenderStop& source)
{
    if (this == &source) return;
    original = source.original;
    engine = source.engine;
    kind = source.kind;
    changed_registers = source.changed_registers;
    register_stream = source.register_stream;
    register_serial = source.register_serial;
    effective_render_enable_mode = source.effective_render_enable_mode;
    color_targets = source.color_targets;
    depth = source.depth;
    programs = source.programs;
    constant_buffers = source.constant_buffers;
    dirty_groups = source.dirty_groups;
    state_stream = source.state_stream;
    state_serial = source.state_serial;
    draw = source.draw;
    inline_indices.assign(source.inline_indices.begin(), source.inline_indices.end());
    compute_descriptor_address = source.compute_descriptor_address;
    compute_descriptor = source.compute_descriptor;
    has_compute_descriptor = source.has_compute_descriptor;
    compute = source.compute;
}

EngineFrontEnd::EngineFrontEnd() noexcept
    : state_stream_(next_state_stream())
{
    // Identity is the reset state, not the encoding of a zero word.
    for (std::size_t viewport = 0; viewport != 16; ++viewport) {
        threed_registers_[0x286 + viewport * 8] = 0x6420;
        threed_shadow_[0x286 + viewport * 8] = 0x6420;
    }
    // Initialize rasterization in both active and replay state. Explicit guest writes still replace this default.
    threed_registers_[0xdf] = 1;
    threed_shadow_[0xdf] = 1;
}

// A backend may leave GPU results GPU-authoritative after rendering. Before
// the front end publishes a semaphore, report or syncpoint, or touches guest
// memory itself, it asks the backend to write them back.
CommandResult EngineFrontEnd::ensure_coherent(const GpuAddressSpace& memory,
                                              const CompletionCallbacks& completion,
                                              std::uint64_t gpu_address,
                                              std::uint64_t size) noexcept
{
    if (completion.rendering.coherence == nullptr ||
        (!rendering_since_coherence_ && !completion.rendering.ordered_effect)) return {};
    const auto result = completion.rendering.coherence(completion.rendering.context, memory,
                                                       gpu_address, size);
    if (!result.ok()) return result;
    // A ranged write-back leaves other GPU results pending.
    if (size == 0) rendering_since_coherence_ = false;
    return {};
}

CommandResult EngineFrontEnd::ensure_readable(const GpuAddressSpace& memory,
                                              const CompletionCallbacks& completion,
                                              std::uint64_t gpu_address,
                                              std::uint64_t size) noexcept
{
    if (completion.rendering.read_coherence)
        return completion.rendering.read_coherence(completion.rendering.context, memory, gpu_address, size);
    return ensure_coherent(memory, completion);
}

CommandResult EngineFrontEnd::release_semaphore(const GpuAddressSpace& memory,
    const CompletionCallbacks& completion, std::uint64_t destination,
    std::span<const std::byte> payload) noexcept
{
    if (completion.rendering.ordered_effect) {
        GuestEffect effect;
        effect.address = destination;
        effect.size = static_cast<std::uint32_t>(payload.size());
        std::copy(payload.begin(), payload.end(), effect.payload.begin());
        rendering_since_coherence_ = true;
        return completion.rendering.ordered_effect(completion.rendering.context, memory, std::move(effect));
    }
    if (const auto coherent = ensure_coherent(memory, completion); !coherent.ok()) return coherent;
    const auto write = memory.write(destination, payload);
    return write ? CommandResult{} : CommandResult{CommandStatus::MemoryFault, write.detail};
}

// An acquire only orders the GPU stream behind earlier work, as hardware
// stalls its front end; with ordered effects it is checked when that work is
// guest-visible instead of blocking command processing on a backend round trip.
CommandResult EngineFrontEnd::acquire_semaphore(const GpuAddressSpace& memory,
    const CompletionCallbacks& completion, std::uint64_t semaphore, std::uint32_t value,
    std::uint32_t operation) noexcept
{
    try {
        GuestEffect effect;
        effect.address = semaphore;
        effect.acquire = GuestEffect::Acquire{value, operation};
        rendering_since_coherence_ = true;
        return completion.rendering.ordered_effect(completion.rendering.context, memory, std::move(effect));
    } catch (...) {
        return {CommandStatus::BudgetExceeded, "Semaphore acquire allocation failed"};
    }
}

CommandResult EngineFrontEnd::increment_syncpoint(const GpuAddressSpace& memory,
    const CompletionCallbacks& completion, std::uint32_t id, std::uint32_t count) noexcept
{
    try {
        if (completion.rendering.ordered_effect && completion.retain_increment) {
            GuestEffect effect;
            effect.publish = completion.retain_increment(completion.context, id, count);
            if (!effect.publish) return {CommandStatus::Unsupported, "Syncpoint publication owner is unavailable"};
            rendering_since_coherence_ = true;
            return completion.rendering.ordered_effect(completion.rendering.context, memory, std::move(effect));
        }
        if (const auto coherent = ensure_coherent(memory, completion); !coherent.ok()) return coherent;
        for (std::uint32_t i = 0; i != count; ++i)
            if (!completion.increment || !completion.increment(completion.context, id))
                return {CommandStatus::Unsupported, "GPU syncpoint increment failed"};
        return {};
    } catch (...) {
        return {CommandStatus::BudgetExceeded, "GPU syncpoint publication allocation failed"};
    }
}

CommandResult EngineFrontEnd::call(const MethodCall& method_call,
                                   std::uint32_t engine,
                                   const GpuAddressSpace& memory,
                                   CompletionCallbacks completion) noexcept
{
    if (rendering_stop_) {
        return result(CommandStatus::Stopped,
                      "GPU engine already stopped at a rendering operation",
                      method_call,
                      engine);
    }
    const bool upload_configuration =
        method_call.method >= 0x60 && method_call.method <= 0x6b;
    const bool upload_data = method_call.method == 0x6d;
    const bool threed_macro_argument =
        engine == engine_class::threed && method_call.method >= 0xe00;
    if (has_pending_memory_work() && !upload_configuration && !upload_data &&
        !threed_macro_argument) {
        return result(CommandStatus::Unsupported,
                      "pending inline upload blocks later GPU side effects",
                      method_call,
                      engine);
    }
    if (engine == 0) {
        return call_puller(method_call, memory, completion);
    }
    if (engine == engine_class::threed) {
        return call_threed(method_call, memory, completion, false);
    }

    if (method_call.method >= 0x1000) {
        return result(CommandStatus::Unsupported,
                      "GPU engine method exceeds the register file",
                      method_call,
                      engine);
    }
    if (engine == engine_class::compute) {
        compute_registers_[method_call.method] = method_call.argument;
        if (method_call.method >= 0x60 && method_call.method <= 0x6d) {
            return call_upload(compute_upload_, method_call, memory, completion, engine);
        }
        if (method_call.method == 0xaf) {
            const std::uint64_t descriptor_address =
                static_cast<std::uint64_t>(compute_registers_[0xad]) << 8;
            std::array<std::byte, 0x100> descriptor{};
            if (const auto coherent = ensure_readable(memory, completion, descriptor_address, descriptor.size());
                !coherent.ok()) return coherent;
            const MemoryResult read = memory.read(descriptor_address, descriptor);
            if (!read) {
                return memory_failure(read, method_call, engine);
            }
            CommandResult stopped = stop_rendering(method_call, engine, RenderKind::Compute);
            if (stopped.status != CommandStatus::Stopped) return stopped;
            RenderStop& stop = *rendering_stop_;
            stop.compute_descriptor_address = descriptor_address;
            stop.compute_descriptor = descriptor;
            stop.has_compute_descriptor = true;
            stop.compute = decode_compute_qmd(descriptor);
            const std::uint32_t program_start =
                read_word(std::span<const std::byte, 4>(descriptor.data() + 0x20, 4));
            stop.programs[0].address =
                address(compute_registers_[0x582], compute_registers_[0x583]) + program_start;
            stop.programs[0].offset = program_start;
            stop.programs[0].enabled = true;
            if (!completion.rendering.dispatch) return stopped;
            // Same synchronous completion contract as a draw: a successful
            // dispatch consumed the stop, and its guest-visible effects are
            // complete or owned by the backend's coherence callback.
            rendering_owner_ = completion.rendering.owner;
            const auto executed = completion.rendering.dispatch(completion.rendering.context,
                                                                *rendering_stop_, memory);
            if (executed.ok()) {
                rendering_since_coherence_ = true;
                rendering_stop_.reset();
                rendering_owner_.reset();
                return {};
            }
            return result(executed.status, executed.detail, method_call, engine);
        }
        return {};
    }
    if (engine == engine_class::twod) {
        twod_registers_[method_call.method] = method_call.argument;
        if (method_call.method == 0x237) {
            const auto stopped = stop_rendering(method_call, engine, RenderKind::Blit);
            if (stopped.status != CommandStatus::Stopped || !completion.rendering.blit) {
                return stopped;
            }
            // The same synchronous completion contract as draws: a successful
            // callback consumed the stop, and its results stay GPU-authoritative
            // until a coherence boundary or guest-visible effect.
            rendering_owner_ = completion.rendering.owner;
            const auto executed = completion.rendering.blit(completion.rendering.context,
                                                            *rendering_stop_, memory);
            if (executed.ok()) {
                rendering_since_coherence_ = true;
                rendering_stop_.reset();
                rendering_owner_.reset();
                return {};
            }
            return result(executed.status, executed.detail, method_call, engine);
        }
        return {};
    }
    if (engine == engine_class::dma_copy) {
        dma_registers_[method_call.method] = method_call.argument;
        if (method_call.method == dma_method::launch)
            return call_dma(method_call, memory, completion, engine);
        return {};
    }
    if (engine == engine_class::inline_to_memory) {
        inline_registers_[method_call.method] = method_call.argument;
        if (method_call.method >= 0x60 && method_call.method <= 0x6d) {
            return call_upload(inline_upload_, method_call, memory, completion, engine);
        }
        return {};
    }
    return result(
        CommandStatus::Unsupported, "GPU engine class is unsupported", method_call, engine);
}

CommandResult EngineFrontEnd::call_upload(InlineUpload& upload,
                                          const MethodCall& method_call,
                                          const GpuAddressSpace& memory,
                                          const CompletionCallbacks& completion,
                                          std::uint32_t engine) noexcept
{
    auto& ordered_failure = &upload == &threed_upload_ ? threed_upload_failure_
        : &upload == &compute_upload_ ? compute_upload_failure_
                                      : inline_upload_failure_;
    if (ordered_failure) return *ordered_failure;

    UploadResult upload_result;
    if (method_call.method <= 0x6b) {
        upload_result = upload.set_register(method_call.method - 0x60, method_call.argument);
    } else if (method_call.method == 0x6c) {
        upload_result = upload.launch(method_call.argument);
    } else if (method_call.method == 0x6d) {
        if (completion.rendering.ordered_write) {
            upload_result = upload.push_word(method_call.argument);
            if (upload_result.accepted()) {
                if (const auto write = upload.staged_write()) {
                    const auto written = completion.rendering.ordered_write(
                        completion.rendering.context, memory, write->destination, write->payload);
                    if (!written.ok()) {
                        ordered_failure = written;
                        return written;
                    }
                    rendering_since_coherence_ = true;
                    upload_result = upload.commit_write();
                }
            }
        } else {
            if (const auto range = upload.write_on_next_word()) {
                if (const auto coherent =
                        ensure_coherent(memory, completion, range->first, range->second);
                    !coherent.ok()) return coherent;
            }
            upload_result = upload.push_word(method_call.argument, memory);
        }
    } else {
        return result(CommandStatus::Unsupported,
                      "GPU method is outside the inline upload register block",
                      method_call,
                      engine);
    }
    if (upload_result.accepted()) {
        return {};
    }
    const CommandStatus status =
        upload_result.status == UploadStatus::MemoryFault
            ? CommandStatus::MemoryFault
        : upload_result.status == UploadStatus::BudgetExceeded
            ? CommandStatus::BudgetExceeded
            : CommandStatus::Unsupported;
    return result(status, upload_result.detail, method_call, engine);
}

// Copies through guest memory: the source is made readable (GPU results
// written back), and the destination is written in order with rendering.
CommandResult EngineFrontEnd::call_dma(const MethodCall& method_call,
                                       const GpuAddressSpace& memory,
                                       const CompletionCallbacks& completion,
                                       std::uint32_t engine) noexcept
{
    const std::uint32_t launch = method_call.argument;
    const auto planned = plan_dma_copy(dma_registers_, launch);
    if (const auto* error = std::get_if<DmaCopyError>(&planned))
        return result(CommandStatus::Unsupported, error->detail, method_call, engine);
    const auto& plan = std::get<DmaCopyPlan>(planned);
    if (plan.destination_size != 0) {
        constexpr std::uint64_t copy_limit = std::uint64_t{256} << 20;
        if (plan.source_size > copy_limit || plan.destination_size > copy_limit)
            return result(CommandStatus::BudgetExceeded, "DMA copy exceeds the copy budget", method_call, engine);
        std::vector<std::byte> source, destination;
        try {
            source.resize(static_cast<std::size_t>(plan.source_size));
            destination.resize(static_cast<std::size_t>(plan.destination_size));
        } catch (const std::bad_alloc&) {
            return result(CommandStatus::BudgetExceeded, "DMA copy storage could not be allocated", method_call, engine);
        }
        const auto read = [&](std::uint64_t address, std::span<std::byte> bytes) -> CommandResult {
            if (const auto coherent = ensure_readable(memory, completion, address, bytes.size()); !coherent.ok())
                return coherent;
            const MemoryResult loaded = memory.read(address, bytes);
            return loaded ? CommandResult{} : memory_failure(loaded, method_call, engine);
        };
        if (!source.empty())
            if (const auto loaded = read(plan.source, source); !loaded.ok()) return loaded;
        if (!plan.destination_covered)
            if (const auto loaded = read(plan.destination, destination); !loaded.ok()) return loaded;
        execute_dma_copy(dma_registers_, launch, plan, source, destination);
        if (completion.rendering.ordered_write) {
            const auto written = completion.rendering.ordered_write(completion.rendering.context, memory,
                                                                    plan.destination, destination);
            if (!written.ok()) return written;
            rendering_since_coherence_ = true;
        } else {
            if (const auto coherent = ensure_coherent(memory, completion, plan.destination, destination.size());
                !coherent.ok()) return coherent;
            const MemoryResult stored = memory.write(plan.destination, destination);
            if (!stored) return memory_failure(stored, method_call, engine);
        }
    }
    if (plan.semaphore == 0) return {};
    const std::uint64_t semaphore = (std::uint64_t{dma_registers_[dma_method::semaphore_a] & 0xffu} << 32) |
                                    dma_registers_[dma_method::semaphore_b];
    const std::uint32_t payload = dma_registers_[dma_method::semaphore_payload];
    if (plan.semaphore == 1) {
        std::array<std::byte, 4> bytes{};
        std::memcpy(bytes.data(), &payload, sizeof payload);
        return release_semaphore(memory, completion, semaphore, bytes);
    }
    if (completion.ticks == nullptr)
        return result(CommandStatus::Unsupported, "DMA four-word semaphore has no tick source", method_call, engine);
    std::array<std::byte, 16> bytes{};
    store_qword(std::span<std::byte, 8>(bytes.data(), 8), payload);
    store_qword(std::span<std::byte, 8>(bytes.data() + 8, 8), completion.ticks(completion.context));
    return release_semaphore(memory, completion, semaphore, bytes);
}

CommandResult EngineFrontEnd::call_puller(const MethodCall& method_call,
                                          const GpuAddressSpace& memory,
                                          CompletionCallbacks completion) noexcept
{
    if (method_call.method >= puller_registers_.size()) {
        return result(CommandStatus::Unsupported,
                      "puller method exceeds the supported register file",
                      method_call,
                      0);
    }
    puller_registers_[method_call.method] = method_call.argument;
    switch (method_call.method) {
    case 0x9:   // coherent write-cache barrier
    case 0x1e:  // synchronous wait-for-idle
        // Later command fetches and front-end reads may consume GPU results.
        // With read-only coherence every such read resolves its own bytes and
        // guest effects are ordered, so the barrier needs no drain.
        if (completion.rendering.read_coherence && completion.rendering.ordered_effect) return {};
        return ensure_coherent(memory, completion);
    case 0x2:  // NOP
    case 0xa:
    case 0xb:
    case 0xc:
    case 0xd:
    case 0x14:  // reference point
        return {};
    case 0x4:
    case 0x5:
    case 0x6:
    case 0x1c:
        return {};
    case 0x1a: {
        std::array<std::byte, 4> bytes{};
        const auto semaphore = address(puller_registers_[4], puller_registers_[5]);
        if (completion.rendering.ordered_effect)
            return acquire_semaphore(memory, completion, semaphore, method_call.argument, 1);
        if (const auto coherent = ensure_readable(memory, completion, semaphore, bytes.size()); !coherent.ok())
            return coherent;
        const MemoryResult read = memory.read(semaphore, bytes);
        if (!read) {
            return memory_failure(read, method_call, 0);
        }
        if (read_word(bytes) != method_call.argument) {
            return result(CommandStatus::Unsupported,
                          "puller semaphore acquire dependency is unmet",
                          method_call,
                          0);
        }
        return {};
    }
    case 0x1b: {
        const auto bytes = word_bytes(method_call.argument);
        return release_semaphore(memory, completion,
                                 address(puller_registers_[4], puller_registers_[5]), bytes);
    }
    case 0x1d: {
        const std::uint32_t syncpoint = (method_call.argument >> 8) & 0xfff;
        if ((method_call.argument & 1) != 0) {
            return increment_syncpoint(memory, completion, syncpoint);
        }
        if (const auto coherent = ensure_coherent(memory, completion); !coherent.ok()) return coherent;
        if (completion.expired == nullptr ||
            !completion.expired(completion.context, syncpoint, puller_registers_[0x1c])) {
            return result(CommandStatus::Unsupported,
                          "puller syncpoint acquire dependency is unmet",
                          method_call,
                          0);
        }
        return {};
    }
    case 0x7: {
        const std::uint32_t operation = method_call.argument & 0xf;
        const std::uint64_t semaphore_address = address(puller_registers_[4], puller_registers_[5]);
        if (operation == 2) {
            if (completion.ticks == nullptr) {
                return result(CommandStatus::Unsupported,
                              "puller long semaphore has no tick source",
                              method_call,
                              0);
            }
            std::array<std::byte, 16> bytes{};
            const std::uint64_t payload = puller_registers_[6];
            const std::uint64_t ticks = completion.ticks(completion.context);
            store_qword(std::span<std::byte, 8>(bytes.data(), 8), payload);
            store_qword(std::span<std::byte, 8>(bytes.data() + 8, 8), ticks);
            return release_semaphore(memory, completion, semaphore_address, bytes);
        }
        if (completion.rendering.ordered_effect && (operation == 1 || operation == 4 || operation == 8))
            return acquire_semaphore(memory, completion, semaphore_address, puller_registers_[6], operation);
        std::array<std::byte, 4> bytes{};
        if (const auto coherent = ensure_readable(memory, completion, semaphore_address, bytes.size());
            !coherent.ok()) return coherent;
        const MemoryResult read = memory.read(semaphore_address, bytes);
        if (!read) {
            return memory_failure(read, method_call, 0);
        }
        const std::uint32_t current = read_word(bytes);
        const std::uint32_t wanted = puller_registers_[6];
        const bool ready = operation == 1   ? current == wanted
                           : operation == 4 ? current >= wanted
                           : operation == 8 ? (current & wanted) != 0
                                            : false;
        return ready ? CommandResult{}
                     : result(CommandStatus::Unsupported,
                              "puller semaphore dependency is unmet or invalid",
                              method_call,
                              0);
    }
    default:
        return result(CommandStatus::Unsupported,
                      "puller method has unsupported side effects",
                      method_call,
                      0);
    }
}

CommandResult EngineFrontEnd::call_threed(const MethodCall& method_call,
                                          const GpuAddressSpace& memory,
                                          CompletionCallbacks completion,
                                          bool from_macro) noexcept
{
    const bool upload_configuration =
        method_call.method >= 0x60 && method_call.method <= 0x6b;
    const bool upload_data = method_call.method == 0x6d;
    if (from_macro && has_pending_memory_work() &&
        !upload_configuration && !upload_data) {
        return result(CommandStatus::Unsupported,
                      "pending inline upload blocks expanded macro side effects",
                      method_call,
                      engine_class::threed);
    }
    if (method_call.method >= threed_registers_.size()) {
        return result(CommandStatus::Unsupported,
                      "3D method exceeds the register file",
                      method_call,
                      engine_class::threed);
    }
    if (from_macro && method_call.method >= 0xe00) {
        return result(CommandStatus::Unsupported,
                      "recursive macro method emission is unsupported",
                      method_call,
                      engine_class::threed);
    }

    std::uint32_t argument = method_call.argument;
    if (method_call.method >= 0x80) {
        if (shadow_mode_ <= 1) {
            threed_shadow_[method_call.method] = argument;
        } else if (shadow_mode_ == 3) {
            argument = threed_shadow_[method_call.method];
        }
    }
    if ((method_call.method == 0x49 || method_call.method == 0x3f8 ||
         method_call.method == 0x3f9 || method_call.method == 0x54e ||
         method_call.method == 0x674 ||
         (method_call.method >= 0x557 && method_call.method <= 0x55f) ||
         (method_call.method >= 0x200 && method_call.method < 0x280) ||
         method_call.method >= 0xe00) &&
        [] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }()) {
        armigo::diag::opt_in(
            "gpu_binding engine=%p method=%#x input=%#x effective=%#x previous=%#x "
            "shadow=%u macro=%u depth=%#llx clear=%#x\n",
            static_cast<const void*>(this), method_call.method, method_call.argument,
            argument, threed_registers_[method_call.method], shadow_mode_, unsigned(from_macro),
            static_cast<unsigned long long>(address(threed_registers_[0x3f8], threed_registers_[0x3f9])),
            threed_registers_[0x674]);
    }
    if (threed_registers_[method_call.method] != argument) {
        if (const auto group = dirty_groups[method_call.method])
            dirty_groups_ |= threed_dirty::bit(group - 1u);
        mark_register(method_call.method);
    }
    threed_registers_[method_call.method] = argument;

    if (method_call.method >= 0x60 && method_call.method <= 0x6d) {
        return call_upload(threed_upload_, method_call, memory, completion, engine_class::threed);
    }

    switch (method_call.method) {
    case 0x44:
        return {};
    case 0x45:
        if (argument >= macro_code_words) {
            return result(CommandStatus::Unsupported,
                          "macro code pointer is outside macro RAM",
                          method_call,
                          engine_class::threed);
        }
        macro_code_pointer_ = argument;
        return {};
    case 0x46:
        if (macro_code_pointer_ >= macro_code_words) {
            return result(CommandStatus::Unsupported,
                          "macro code upload exceeds macro RAM",
                          method_call,
                          engine_class::threed);
        }
        macro_code_[macro_code_pointer_] = argument;
        macro_valid_[macro_code_pointer_++] = true;
        ++macro_code_version_;
        return {};
    case 0x47:
        if (argument >= macro_entries) {
            return result(CommandStatus::Unsupported,
                          "macro entry pointer is outside entry RAM",
                          method_call,
                          engine_class::threed);
        }
        macro_entry_pointer_ = argument;
        return {};
    case 0x48:
        if (macro_entry_pointer_ >= macro_entries || argument >= macro_code_words) {
            return result(CommandStatus::Unsupported,
                          "macro entry binding is outside macro RAM",
                          method_call,
                          engine_class::threed);
        }
        macro_entry_[macro_entry_pointer_] = argument;
        ++macro_code_version_;
        macro_entry_valid_[macro_entry_pointer_++] = true;
        return {};
    case 0x49:
        shadow_mode_ = method_call.argument & 3;
        threed_registers_[0x49] = method_call.argument;
        mark_register(0x49);
        return {};
    case 0xb2: {
        const std::uint32_t syncpoint = argument & 0xffff;
        return increment_syncpoint(memory, completion, syncpoint);
    }
    case 0x378:
    case 0x3dd:
    case 0x3df:
        return {};
    case 0x556:
        effective_render_enable_mode_ = argument;
        return {};
    case 0x586:
        return draw_begin(method_call, argument);
    case 0x5f8:
        draw_indexed_ = true;
        return {};
    case 0x54c:
        return result(CommandStatus::Unsupported,
                      "3D counter reset is not implemented",
                      method_call,
                      engine_class::threed);
    case 0x6c3: {
        const std::uint32_t operation = argument & 3;
        const std::uint32_t report = (argument >> 23) & 0x1f;
        if ((operation != 0 && operation != 2) || report != 0) {
            return result(CommandStatus::Unsupported,
                          "3D report query form is unsupported",
                          method_call,
                          engine_class::threed);
        }
        const std::uint64_t destination =
            address(threed_registers_[0x6c0], threed_registers_[0x6c1]);
        const std::uint64_t payload = threed_registers_[0x6c2];
        if (((argument >> 28) & 1) != 0) {
            const auto bytes = word_bytes(static_cast<std::uint32_t>(payload));
            return release_semaphore(memory, completion, destination, bytes);
        }
        if (completion.ticks == nullptr) {
            return result(CommandStatus::Unsupported,
                          "3D long report has no tick source",
                          method_call,
                          engine_class::threed);
        }
        std::array<std::byte, 16> bytes{};
        const std::uint64_t ticks = completion.ticks(completion.context);
        store_qword(std::span<std::byte, 8>(bytes.data(), 8), payload);
        store_qword(std::span<std::byte, 8>(bytes.data() + 8, 8), ticks);
        return release_semaphore(memory, completion, destination, bytes);
    }
    case 0x8c4:
        // Acknowledge this firmware call in active scratch only. The physical
        // configuration has no rendering effect in this compatibility model.
        threed_registers_[0xd00] = 1;
        mark_register(0xd00);
        return {};
    default:
        break;
    }

    // Other Falcon slots retain normal register/shadow writes without an
    // execution callback; no backend obligation is queued.
    if (method_call.method >= 0x8c0 && method_call.method <= 0x8df) {
        return {};
    }

    if (method_call.method >= 0xe00) {
        if (from_macro) {
            return result(CommandStatus::Unsupported,
                          "recursive macro execution is unsupported",
                          method_call,
                          engine_class::threed);
        }
        if (pending_macro_method_ == 0) {
            if ((method_call.method & 1) != 0) {
                return result(CommandStatus::Unsupported,
                              "macro invocation starts at an argument method",
                              method_call,
                              engine_class::threed);
            }
            pending_macro_method_ = method_call.method;
        } else if (method_call.method != pending_macro_method_ + 1) {
            return result(CommandStatus::Unsupported,
                          "macro invocation arguments are not contiguous",
                          method_call,
                          engine_class::threed);
        }
        try {
            pending_macro_arguments_.push_back(argument);
        } catch (const std::bad_alloc&) {
            return result(CommandStatus::BudgetExceeded,
                          "macro argument allocation failed",
                          method_call,
                          engine_class::threed);
        }
        return method_call.last ? execute_pending_macro(method_call, memory, completion)
                                : CommandResult{};
    }

    if (method_call.method >= 0x904 && method_call.method <= 0x924 &&
        (method_call.method - 0x904) % 8 == 0) {
        const std::size_t stage = (method_call.method - 0x904) / 8;
        const std::size_t slot = (argument >> 4) & 0x1f;
        if (slot >= constant_buffers_[stage].size()) {
            return result(CommandStatus::Unsupported,
                          "3D constant-buffer binding slot is outside the supported range",
                          method_call, engine_class::threed);
        }
        // The bind consumes the selected window now, independently of later
        // buffer selections. Payload bytes remain live until execution reads them.
        const auto bound = (argument & 1) != 0
            ? ConstantBufferDescription{address(threed_registers_[0x8e1], threed_registers_[0x8e2]),
                                        threed_registers_[0x8e0], true}
            : ConstantBufferDescription{};
        auto& binding = constant_buffers_[stage][slot];
        if (binding.address != bound.address || binding.size != bound.size ||
            binding.enabled != bound.enabled)
            dirty_groups_ |= threed_dirty::bit(threed_dirty::ConstantBuffers0 + std::uint32_t(stage));
        binding = bound;
        return {};
    }

    if (method_call.method >= 0x8e4 && method_call.method <= 0x8f3) {
        const std::uint32_t size = threed_registers_[0x8e0];
        const std::uint32_t offset = threed_registers_[0x8e3];
        if (offset > size || size - offset < 4) {
            return result(CommandStatus::Unsupported,
                          "constant-buffer upload exceeds its selected range",
                          method_call,
                          engine_class::threed);
        }
        const auto bytes = word_bytes(argument);
        const std::uint64_t upload_address =
            address(threed_registers_[0x8e1], threed_registers_[0x8e2]) + offset;
        if (completion.rendering.ordered_write) {
            const auto written = completion.rendering.ordered_write(
                completion.rendering.context, memory, upload_address, bytes);
            if (!written.ok()) return written;
            rendering_since_coherence_ = true;
        } else {
            if (const auto coherent =
                    ensure_coherent(memory, completion, upload_address, bytes.size());
                !coherent.ok()) return coherent;
            const MemoryResult write = memory.write(upload_address, bytes);
            if (!write) {
                return memory_failure(write, method_call, engine_class::threed);
            }
        }
        threed_registers_[0x8e3] = offset + 4;
        mark_register(0x8e3);
        return {};
    }

    if (method_call.method == 0x674) {
        auto stopped = stop_rendering(method_call, engine_class::threed, RenderKind::Clear);
        if (stopped.status != CommandStatus::Stopped || !completion.rendering.clear) {
            return stopped;
        }
        rendering_owner_ = completion.rendering.owner;
        const auto executed = completion.rendering.clear(
            completion.rendering.context, *rendering_stop_, memory);
        if (executed.ok()) {
            rendering_since_coherence_ = true;
            rendering_stop_.reset();
            rendering_owner_.reset();
            return {};
        }
        return result(executed.status, executed.detail, method_call, engine_class::threed);
    }
    switch (method_call.method) {
    case 0x4c1: {
        const std::array indices{
            argument & 0xffu, (argument >> 8) & 0xffu,
            (argument >> 16) & 0xffu, argument >> 24};
        return append_inline_indices(method_call, indices);
    }
    case 0x57a: {
        const std::array indices{argument};
        return append_inline_indices(method_call, indices);
    }
    case 0x57c: {
        const std::array indices{argument & 0xffffu, argument >> 16};
        return append_inline_indices(method_call, indices);
    }
    case 0x42b:
        return stop_rendering(method_call, engine_class::threed, RenderKind::Draw);
    case 0x485:
        return compact_draw(method_call, memory, completion, argument, false, false);
    case 0x486:
        return compact_draw(method_call, memory, completion, argument, false, true);
    case 0x5f9:
    case 0x5fa:
    case 0x5fb:
        return compact_draw(method_call, memory, completion, argument, true, false);
    case 0x5fc:
    case 0x5fd:
    case 0x5fe:
        return compact_draw(method_call, memory, completion, argument, true, true);
    case 0x585:
        return draw_end(method_call, memory, completion);
    default:
        return {};
    }
}

CommandResult EngineFrontEnd::append_inline_indices(
    const MethodCall& method_call, std::span<const std::uint32_t> indices) noexcept
{
    if (indices.size() > maximum_inline_indices -
                             std::min(inline_indices_.size(), maximum_inline_indices)) {
        static_cast<void>(stop_rendering(
            method_call, engine_class::threed, RenderKind::Draw));
        return result(CommandStatus::BudgetExceeded,
                      "3D inline index stream exceeds the bounded word limit",
                      method_call,
                      engine_class::threed);
    }
    try {
        inline_indices_.insert(inline_indices_.end(), indices.begin(), indices.end());
        return {};
    } catch (const std::bad_alloc&) {
        static_cast<void>(stop_rendering(
            method_call, engine_class::threed, RenderKind::Draw));
        return result(CommandStatus::BudgetExceeded,
                      "3D inline index stream allocation failed",
                      method_call,
                      engine_class::threed);
    }
}

CommandResult EngineFrontEnd::draw_begin(const MethodCall& method_call,
                                         std::uint32_t argument) noexcept
{
    constexpr std::uint32_t instance_increment = 1u << 26;
    constexpr std::uint32_t instance_preserve = 1u << 27;
    constexpr std::uint32_t allowed = 0xffffu | instance_increment | instance_preserve;
    if ((argument & ~allowed) != 0) {
        static_cast<void>(stop_rendering(method_call, engine_class::threed, RenderKind::Draw));
        return result(CommandStatus::Unsupported,
                      "Draw Begin contains unsupported upper controls",
                      method_call,
                      engine_class::threed);
    }

    const bool increment = (argument & instance_increment) != 0;
    const bool preserve = (argument & instance_preserve) != 0;
    if ((inline_draw_consumed_ || !inline_indices_.empty()) && (increment || preserve)) {
        static_cast<void>(stop_rendering(method_call, engine_class::threed, RenderKind::Draw));
        return result(CommandStatus::Unsupported,
                      "Inline index draws do not support instance continuation",
                      method_call,
                      engine_class::threed);
    }
    if (increment) {
        if (draw_instance_ == std::numeric_limits<std::uint32_t>::max()) {
            static_cast<void>(stop_rendering(
                method_call, engine_class::threed, RenderKind::Draw));
            return result(CommandStatus::BudgetExceeded,
                          "Draw instance counter overflowed",
                          method_call,
                          engine_class::threed);
        }
        ++draw_instance_;
    } else if (!preserve) {
        draw_instance_ = 0;
        inline_draw_consumed_ = false;
    }
    draw_instance_continuation_ = increment || preserve;
    draw_begin_ = argument & 0xffffu;
    return {};
}

CommandResult EngineFrontEnd::compact_draw(const MethodCall& method_call,
                                           const GpuAddressSpace& memory,
                                           CompletionCallbacks completion,
                                           std::uint32_t argument,
                                           bool indexed,
                                           bool subsequent) noexcept
{
    const auto prior_indexed = draw_indexed_;
    if ((inline_draw_consumed_ || !inline_indices_.empty()) && subsequent) {
        static_cast<void>(stop_rendering(method_call, engine_class::threed, RenderKind::Draw));
        return result(CommandStatus::Unsupported,
                      "Inline index draws do not support compact instance continuation",
                      method_call,
                      engine_class::threed);
    }
    if (subsequent) {
        if (draw_instance_ == std::numeric_limits<std::uint32_t>::max()) {
            static_cast<void>(stop_rendering(
                method_call, engine_class::threed, RenderKind::Draw));
            return result(CommandStatus::BudgetExceeded,
                          "Compact draw instance counter overflowed",
                          method_call,
                          engine_class::threed);
        }
        ++draw_instance_;
    } else {
        draw_instance_ = 0;
        inline_draw_consumed_ = false;
    }

    const std::uint32_t topology = argument >> 28;
    if (draw_begin_) {
        draw_begin_ = topology;
        draw_instance_continuation_ = subsequent;
    }
    DrawDescription draw{
        indexed,
        topology,
        argument & 0xffffu,
        (argument >> 16) & 0xfffu,
        indexed ? std::bit_cast<std::int32_t>(threed_registers_[0x50d]) : 0,
        0,
        1};
    const auto dispatched = dispatch_draw(
        method_call, memory, completion, draw, subsequent);
    draw_indexed_ = prior_indexed;
    return dispatched;
}

CommandResult EngineFrontEnd::draw_end(const MethodCall& method_call,
                                       const GpuAddressSpace& memory,
                                       CompletionCallbacks completion) noexcept
{
    const auto begin = draw_begin_;
    const bool indexed = draw_indexed_;
    const bool continuation = draw_instance_continuation_;
    draw_begin_.reset();
    draw_indexed_ = false;
    draw_instance_continuation_ = false;
    // Preserve the diagnostic stop for an unproven trigger, even when a
    // backend is installed. No stored count alone authorizes a draw callback.
    if (!begin) {
        const auto stopped = stop_rendering(
            method_call, engine_class::threed, RenderKind::Draw);
        rendering_stop_->inline_indices = std::move(inline_indices_);
        inline_indices_.clear();
        inline_draw_consumed_ = !rendering_stop_->inline_indices.empty();
        return stopped;
    }

    if (threed_registers_[0x585] != 0) {
        const auto stopped = stop_rendering(
            method_call, engine_class::threed, RenderKind::Draw);
        rendering_stop_->inline_indices = std::move(inline_indices_);
        inline_indices_.clear();
        inline_draw_consumed_ = !rendering_stop_->inline_indices.empty();
        return completion.rendering.draw
            ? result(CommandStatus::Unsupported,
                     "Draw End argument is unsupported",
                     method_call,
                     engine_class::threed)
            : stopped;
    }
    DrawDescription draw{
        indexed, *begin,
        threed_registers_[indexed ? 0x5f7 : 0x35d],
        threed_registers_[indexed ? 0x5f8 : 0x35e],
        indexed ? std::bit_cast<std::int32_t>(threed_registers_[0x50d]) : 0,
        0, 1};
    if (!inline_indices_.empty()) {
        // Inline data replaces either ordinary source, but the hardware's
        // stored first-index state still applies independently of the indexed
        // event latch and must be validated explicitly.
        draw.indexed = true;
        draw.first = threed_registers_[0x5f7];
    }
    if (inline_indices_.empty() && draw.count != 0 && draw.count - 1 >
            std::numeric_limits<std::uint32_t>::max() - draw.first) {
        const auto stopped = stop_rendering(
            method_call, engine_class::threed, RenderKind::Draw);
        rendering_stop_->inline_indices = std::move(inline_indices_);
        inline_indices_.clear();
        inline_draw_consumed_ = !rendering_stop_->inline_indices.empty();
        return completion.rendering.draw
            ? result(CommandStatus::Unsupported,
                     "Draw first/count exceeds the supported 32-bit element range",
                     method_call,
                     engine_class::threed)
            : stopped;
    }
    return dispatch_draw(method_call, memory, completion, draw, continuation);
}

CommandResult EngineFrontEnd::dispatch_draw(const MethodCall& method_call,
                                            const GpuAddressSpace& memory,
                                            CompletionCallbacks completion,
                                            DrawDescription draw,
                                            bool continuation) noexcept
{
    const auto stopped = stop_rendering(method_call, engine_class::threed, RenderKind::Draw);
    const bool has_inline = !inline_indices_.empty();
    if (has_inline) {
        rendering_stop_->inline_indices = std::move(inline_indices_);
        inline_indices_.clear();
        inline_draw_consumed_ = true;
    }
    const auto unsupported = [&](CommandStatus status, std::string_view detail) {
        return completion.rendering.draw
            ? result(status, detail, method_call, engine_class::threed)
            : stopped;
    };
    if (threed_registers_[0x50e] >
        std::numeric_limits<std::uint32_t>::max() - draw_instance_) {
        return unsupported(CommandStatus::BudgetExceeded,
                           "Draw base instance and instance counter overflowed");
    }
    draw.base_instance = threed_registers_[0x50e];
    draw.instance_index = draw_instance_;
    if (has_inline) {
        if (continuation || draw_instance_ != 0) {
            return unsupported(CommandStatus::Unsupported,
                               "Inline index stream cannot be used by an instance continuation");
        }
        if (draw.indexed && draw.first != 0) {
            return unsupported(CommandStatus::Unsupported,
                               "Inline index stream requires a zero first index");
        }
        draw.indexed = true;
        draw.first = 0;
        draw.count = static_cast<std::uint32_t>(rendering_stop_->inline_indices.size());
        draw.base_vertex = std::bit_cast<std::int32_t>(threed_registers_[0x50d]);
    }
    rendering_stop_->draw = draw;
    if (!completion.rendering.draw) return stopped;
    // Zero count still reaches the backend contract: state/coherence completion
    // is its responsibility, and this front end never invents a completed draw.
    rendering_owner_ = completion.rendering.owner;
    const auto executed = completion.rendering.draw(completion.rendering.context,
                                                    *rendering_stop_, memory);
    if (executed.ok()) {
        rendering_since_coherence_ = true;
        rendering_stop_.reset();
        rendering_owner_.reset();
        return {};
    }
    return result(executed.status, executed.detail, method_call, engine_class::threed);
}

CommandResult EngineFrontEnd::execute_pending_macro(const MethodCall& method_call,
                                                    const GpuAddressSpace& memory,
                                                    CompletionCallbacks completion) noexcept
{
    const std::uint32_t slot = (pending_macro_method_ - 0xe00) >> 1;
    if (slot >= macro_entries || !macro_entry_valid_[slot] || !macro_valid_[macro_entry_[slot]]) {
        pending_macro_method_ = 0;
        pending_macro_arguments_.clear();
        return result(CommandStatus::Unsupported,
                      "macro entry points to uninitialized code",
                      method_call,
                      engine_class::threed);
    }
    const std::uint32_t entry = macro_entry_[slot];
    auto& cached_end = macro_end_[slot];
    if (cached_end.first != macro_code_version_ || cached_end.second < entry) {
        std::uint32_t scan = entry;
        while (scan < macro_code_words && macro_valid_[scan]) ++scan;
        cached_end = {macro_code_version_, scan};
    }
    const std::size_t end = cached_end.second;

    struct Context {
        EngineFrontEnd* self;
        const GpuAddressSpace* memory;
        CompletionCallbacks completion;
        MethodCall outer;
        CommandResult last{};
    } context{this, &memory, completion, method_call, {}};

    const auto read =
        [](void* raw, std::uint32_t method, std::uint32_t& value) noexcept -> MacroResult {
        auto& context = *static_cast<Context*>(raw);
        if (method >= context.self->threed_registers_.size()) {
            context.last = result(CommandStatus::Unsupported,
                                  "macro register read is out of range",
                                  context.outer,
                                  engine_class::threed);
            return {MacroResultCode::CallbackStopped, context.last.detail, 0};
        }
        value = context.self->threed_registers_[method];
        return {};
    };
    const auto emit =
        [](void* raw, std::uint32_t method, std::uint32_t value) noexcept -> MacroResult {
        auto& context = *static_cast<Context*>(raw);
        MethodCall emitted{method, context.outer.subchannel, value, context.outer.address, true};
        context.last =
            context.self->call_threed(emitted, *context.memory, context.completion, true);
        if (!context.last.ok()) {
            return {MacroResultCode::CallbackStopped, context.last.detail, 0};
        }
        return {};
    };

    const MacroResult macro_result =
        execute_macro(std::span<const std::uint32_t>(macro_code_).subspan(entry, end - entry),
                      0,
                      pending_macro_arguments_,
                      {read, emit, &context});
    pending_macro_method_ = 0;
    pending_macro_arguments_.clear();
    if (!macro_result) {
        if (!context.last.ok()) {
            return context.last;
        }
        return result(macro_result.code == MacroResultCode::InstructionBudgetExceeded
                          ? CommandStatus::BudgetExceeded
                          : CommandStatus::Unsupported,
                      macro_result.detail,
                      method_call,
                      engine_class::threed);
    }
    return {};
}

CommandResult EngineFrontEnd::stop_rendering(const MethodCall& method_call,
                                             std::uint32_t engine,
                                             RenderKind kind) noexcept
{
    auto& stop = rendering_stop_.recycle();
    stop.original = method_call;
    stop.engine = engine;
    stop.kind = kind;
    stop.effective_render_enable_mode = 1;
    stop.dirty_groups = threed_dirty::all;
    stop.state_stream = 0;
    stop.state_serial = 0;
    stop.register_stream = 0;
    stop.register_serial = 0;
    if (state_stream_ && register_serial_ != std::numeric_limits<std::uint64_t>::max()) {
        stop.register_stream = state_stream_;
        stop.register_serial = ++register_serial_;
    }
    stop.draw.reset();
    stop.inline_indices.clear();
    stop.compute_descriptor_address = 0;
    stop.compute_descriptor = {};
    stop.has_compute_descriptor = false;
    stop.compute.reset();
    const auto* source = &threed_registers_;
    if (engine == engine_class::compute) {
        source = &compute_registers_;
    } else if (engine == engine_class::twod) {
        source = &twod_registers_;
    }
    // A 3D stop following a 3D stop copies only the registers written since.
    if (engine == engine_class::threed && copied_engine_ == engine_class::threed) {
        stop.changed_registers = changed_registers_;
        for (std::size_t word = 0; word < changed_registers_.size(); ++word) {
            for (auto bits = changed_registers_[word]; bits; bits &= bits - 1) {
                const auto method = word * 64 + std::size_t(__builtin_ctzll(bits));
                stop.registers[method] = threed_registers_[method];
            }
        }
    } else {
        stop.registers = *source;
        stop.changed_registers.fill(std::numeric_limits<std::uint64_t>::max());
    }
    if (engine == engine_class::threed) changed_registers_ = {};
    copied_engine_ = engine;
    if (engine != engine_class::threed) {
        stop.color_targets = {};
        stop.depth = {};
        stop.programs = {};
        stop.constant_buffers = {};
    } else {
        stop.effective_render_enable_mode = effective_render_enable_mode_;
        stop.constant_buffers = constant_buffers_;
        for (std::size_t index = 0; index < stop.color_targets.size(); ++index) {
            const std::size_t base = 0x200 + index * 0x10;
            auto& target = stop.color_targets[index];
            target.address = address(threed_registers_[base], threed_registers_[base + 1]);
            target.width = threed_registers_[base + 2];
            target.height = threed_registers_[base + 3];
            target.format = threed_registers_[base + 4];
            target.tile_mode = threed_registers_[base + 5];
            target.depth = threed_registers_[base + 6] & 0xffff;
            target.array_pitch = threed_registers_[base + 7];
            target.base_layer = threed_registers_[base + 8];
        }
        stop.depth.address = address(threed_registers_[0x3f8], threed_registers_[0x3f9]);
        stop.depth.format = threed_registers_[0x3fa];
        stop.depth.tile_mode = threed_registers_[0x3fb];
        stop.depth.array_pitch = threed_registers_[0x3fc];
        stop.depth.width = threed_registers_[0x48a];
        stop.depth.height = threed_registers_[0x48b];
        stop.depth.enabled = threed_registers_[0x54e] != 0;
        const std::uint64_t program_base =
            address(threed_registers_[0x582], threed_registers_[0x583]);
        for (std::size_t index = 0; index < stop.programs.size(); ++index) {
            const std::size_t base = 0x800 + index * 0x10;
            auto& program = stop.programs[index];
            program.control = threed_registers_[base];
            program.offset = threed_registers_[base + 1];
            program.address = program_base + program.offset;
            program.enabled = index == 1 || (program.control & 1) != 0;
        }
        if (kind == RenderKind::Draw) {
            stop.dirty_groups = dirty_groups_;
            stop.state_stream = state_stream_;
            stop.state_serial = ++state_serial_;
            dirty_groups_ = 0;
        }
    }
    return result(
        CommandStatus::Stopped, "GPU command reached a rendering operation", method_call, engine);
}

}  // namespace armigo::graphics
