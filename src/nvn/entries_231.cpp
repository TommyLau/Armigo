// ============================================================================
// Armigo NVN - API 231 Entry Points
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// API 231 entry points: the ones the API 55 implementation serves unchanged,
// and those whose numbering, handles or objects are API 231's own.
//
// Descriptor handles: a texture handle is the texture ID in bits 0-19 and the
// sampler ID shifted left by 20, both sign-extended to 64 bits, and zero for
// texture ID zero; texel-fetch and image handles are the sign-extended ID.
// Unlike API 55 no bit 32 marks a texture. Shaders find the texture and
// sampler IDs in the same bits as API 55 (handle_texture_id and
// handle_sampler_id), which index the pools registered through
// nvnTexturePoolRegisterTexture and nvnSamplerPoolRegisterSampler.

#include "backend.h"
#include "commands.h"
#include "entry_table_231.h"
#include "host_stack.h"
#include "objects.h"
#include "present_internal.h"

#include <armigo/common/diagnostics.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <utility>

namespace armigo::nvn::v231 {
namespace {

// The storage an API 231 object has, which the API 55 state must fit: what the
// title's driver itself writes in its defaults or initialization, or where
// that is less, what the title reserves for the object.
template <typename State>
constexpr bool fits(std::size_t api231_storage) noexcept { return sizeof(State) <= api231_storage; }
static_assert(fits<DeviceBuilderState>(0x18) && fits<DeviceState>(0x3000));
static_assert(fits<MemoryPoolBuilderState>(0x38) && fits<MemoryPoolState>(0x100));
static_assert(fits<DescriptorPoolState>(0x20));
static_assert(fits<TextureBuilderState>(0x68) && fits<TextureState>(0x80) && fits<TextureViewState>(0x10));
// The driver writes 0x30 bytes of a sampler builder; the title keeps its
// builders in 0x40 bytes of stack below other locals.
static_assert(fits<SamplerBuilderState>(0x40) && fits<SamplerState>(0x40));
static_assert(fits<ProgramState>(0xa0));
static_assert(fits<QueueBuilderState>(0x48) && fits<QueueState>(0x2cb8));
static_assert(fits<SyncState>(0x20) && fits<EventBuilderState>(0x10) && fits<EventState>(0x28));
static_assert(fits<WindowBuilderState>(0x70) && fits<WindowState>(0x1c0));
static_assert(fits<CommandBufferState>(0xb0));
static_assert(fits<BlendStateBits>(8) && fits<ColorStateBits>(4) && fits<ChannelMaskStateBits>(4) &&
              fits<MultisampleStateBits>(24) && fits<PolygonStateBits>(4) && fits<DepthStencilStateBits>(8) &&
              fits<VertexAttributeStateBits>(4) && fits<VertexStreamStateBits>(8));
// Buffer states are private to the memory lane: 32 and 48 bytes, against a
// 0x28-byte builder and a 0x30-byte buffer.

void unsupported(const char* detail, std::uint64_t guest_return) noexcept {
    on_host_stack([&] { report_unsupported(detail, guest_return); });
}

#define ARMIGO_GUEST_RETURN reinterpret_cast<std::uint64_t>(__builtin_return_address(0))

// ---------------------------------------------------------------- device

PFNNVNGENERICFUNCPTRPROC device_get_proc_address(const NVNdevice*, const char* name) noexcept {
    return armigo_nvn231_bootstrap_loader(name);
}

// Properties 2-97 keep their API 55 numbers and meanings in the title's driver;
// the device answers them as the API 55 device does, except the shader code
// limits that describe API 231 binaries: the GPU code version range (201.0 to
// 201.20) and the padding after shader code (0x800 bytes). Newer properties
// stop the run naming their number until a title is seen to ask for one.
void device_get_integer(const NVNdevice* device, int info, int* out) noexcept {
    if (!out) return;
    const auto version = selected_api_version();
    switch (info) {
    case 0: *out = version ? version->major : 0; return;
    case 1: *out = version ? version->minor : 0; return;
    case 46: case 47: *out = 201; return;
    case 48: *out = 20; return;
    case 49: *out = 0; return;
    case 78: *out = 0x800; return;
    default: break;
    }
    // Properties 99 to 109, constants in the title's driver (the movie player
    // queries 103); 98 reads device state there and stays unsupported.
    static constexpr int properties_99_to_109[] = {256, 16, 65535, 65535, 256, 1, 16, 16, 16, 1, 1};
    if (info >= 99 && info <= 109) {
        *out = properties_99_to_109[info - 99];
        return;
    }
    if (info >= 2 && info <= 97) {
        if (const auto api55 = implemented_entry(nvn::Entry::nvnDeviceGetInteger)) {
            reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnDeviceGetInteger>::type>(api55)(
                device, static_cast<NVNdeviceInfo>(info), out);
            return;
        }
    }
    const auto guest_return = ARMIGO_GUEST_RETURN;
    char detail[96];
    std::snprintf(detail, sizeof detail, "Armigo NVN 231.40: device property %d is not implemented", info);
    unsupported(detail, guest_return);
    *out = 0;
}

NVNtextureHandle device_get_texture_handle(const NVNdevice*, int texture, int sampler) noexcept {
    if (texture == 0) return 0;
    return std::uint64_t(std::int64_t(texture)) | std::uint64_t(std::int64_t(sampler)) << 20;
}

NVNtextureHandle device_get_texel_fetch_handle(const NVNdevice*, int texture) noexcept {
    return std::uint64_t(std::int64_t(texture));
}

NVNimageHandle device_get_image_handle(const NVNdevice*, int image) noexcept {
    return std::uint64_t(std::int64_t(image));
}

// ---------------------------------------------------------------- programs

// API 231 builds programs from a builder that names the device and the shader
// data (16-byte records, as nvnProgramSetShaders takes them).
struct ProgramBuilderState {
    NVNdevice* device;
    std::int32_t count;
    const NVNshaderData* shaders;
};
static_assert(fits<ProgramBuilderState>(0x64));

void program_builder_set_defaults(void* builder) noexcept {
    auto* state = static_cast<ProgramBuilderState*>(builder);
    state->count = 0;
    state->shaders = nullptr;
}

void program_builder_set_device(void* builder, NVNdevice* device) noexcept {
    static_cast<ProgramBuilderState*>(builder)->device = device;
}

void program_builder_set_shaders(void* builder, int count, const NVNshaderData* shaders) noexcept {
    auto* state = static_cast<ProgramBuilderState*>(builder);
    state->count = count;
    state->shaders = shaders;
}

NVNboolean program_initialize_from_builder(NVNprogram* program, const void* builder) noexcept {
    const auto initialize = implemented_entry(nvn::Entry::nvnProgramInitialize);
    const auto set_shaders = implemented_entry(nvn::Entry::nvnProgramSetShaders);
    const auto guest_return = ARMIGO_GUEST_RETURN;
    if (!program || !builder || !initialize || !set_shaders) {
        unsupported("Armigo NVN 231.40: nvnProgramInitializeFromBuilder has no program implementation", guest_return);
        return 0;
    }
    const auto* state = static_cast<const ProgramBuilderState*>(builder);
    if (!reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnProgramInitialize>::type>(initialize)(
            program, state->device))
        return 0;
    if (state->count == 0) return 1;
    return reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnProgramSetShaders>::type>(set_shaders)(
        program, state->count, state->shaders);
}

// ---------------------------------------------------------------- memory pools

// The title gives a physical pool a compressible kind through a virtual pool
// of the same size that maps all of it once. With GPU addresses equal to
// guest addresses such a virtual pool is the physical pool's memory: it takes
// that address, the physical pool's backend registration covers it, and its
// storage class (a compression kind) has no Metal counterpart. Any other
// mapping stops the run.
NVNboolean memory_pool_map_virtual(NVNmemoryPool* pool, int count, const NVNmappingRequest* requests) noexcept {
    const auto guest_return = ARMIGO_GUEST_RETURN;
    auto* state = reinterpret_cast<MemoryPoolState*>(pool);
    const auto& request = requests ? requests[0] : NVNmappingRequest{};
    const auto* physical = reinterpret_cast<const MemoryPoolState*>(request.physicalPool);
    if (!state || !(state->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || state->memory || count != 1 || !requests ||
        !physical || !physical->memory || (physical->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) ||
        request.virtualOffset != 0 || request.size != state->size ||
        request.physicalOffset > physical->size || request.size > physical->size - request.physicalOffset) {
        unsupported("Armigo NVN 231.40: only a single mapping of a whole virtual pool onto one physical pool "
                    "is implemented",
                    guest_return);
        return 0;
    }
    state->memory = physical->memory + request.physicalOffset;
    // The aliased physical range. Marked on the call, not as a whole-pool
    // throttle, so a later whole-pool address of that storage is still recorded.
    if (backends().resources && request.size) {
        const auto address = reinterpret_cast<std::uint64_t>(state->memory);
        const auto size = request.size;
        on_host_stack([&] { backends().resources->note_addressed(address, size); });
    }
    return 1;
}

// A virtual pool aliasing a physical one owns no backend registration: it gives
// up the borrowed address before the shared cleanup of a finalize or of a
// reinitialization releases what it finds there.
NVNboolean memory_pool_initialize(NVNmemoryPool* pool, const NVNmemoryPoolBuilder* builder) noexcept {
    auto* state = reinterpret_cast<MemoryPoolState*>(pool);
    if (state && (state->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL)) state->memory = nullptr;
    const auto api55 = implemented_entry(nvn::Entry::nvnMemoryPoolInitialize);
    return api55 && reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnMemoryPoolInitialize>::type>(api55)(
                        pool, builder);
}

void memory_pool_finalize(NVNmemoryPool* pool) noexcept {
    auto* state = reinterpret_cast<MemoryPoolState*>(pool);
    if (state && (state->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL)) state->memory = nullptr;
    if (const auto api55 = implemented_entry(nvn::Entry::nvnMemoryPoolFinalize))
        reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnMemoryPoolFinalize>::type>(api55)(pool);
}

// ---------------------------------------------------------------- queues

// A queue built with a primary queue runs in the primary's GPU context in the
// title's driver. Armigo replays every queue's submissions on one backend in
// submission order, so the pairing needs no state of its own.
void queue_builder_set_primary_queue(NVNqueueBuilder*, NVNqueue*) noexcept {}

// ---------------------------------------------------------------- command buffers

// Recording modes pick the GPU engine that runs a command buffer (the driver
// starts each command with a mode-specific engine word): 0 and 3 graphics, 1
// and 4 compute, 2 copy. Replay keeps one ordered stream per queue for every
// engine, so a valid mode changes nothing.
void command_buffer_set_recording_mode(NVNcommandBuffer* command_buffer, int mode) noexcept {
    if (command_buffer && mode >= 0 && mode <= 4) return;
    const auto guest_return = ARMIGO_GUEST_RETURN;
    char detail[96];
    std::snprintf(detail, sizeof detail, "Armigo NVN 231.40: command buffer recording mode %d is not implemented",
                  mode);
    unsupported(detail, guest_return);
}

// Both memory callbacks start enabled, as the driver initializes them.
void command_buffer_set_command_memory_callback_enabled(NVNcommandBuffer* command_buffer,
                                                         std::uint32_t enabled) noexcept {
    if (command_buffer) reinterpret_cast<CommandBufferState*>(command_buffer)->command_callback_disabled = !guest_bool(enabled);
}

void command_buffer_set_control_memory_callback_enabled(NVNcommandBuffer* command_buffer,
                                                         std::uint32_t enabled) noexcept {
    if (command_buffer) reinterpret_cast<CommandBufferState*>(command_buffer)->control_callback_disabled = !guest_bool(enabled);
}

// Task and mesh retain logical resource banks 6 and 7 independently of
// ordinary graphics stages. Replay selects their physical execution slots.
template <nvn::Entry E, typename Signature = typename nvn::EntrySignature<E>::type>
struct StageBind;
template <nvn::Entry E, typename... Rest>
struct StageBind<E, void (*)(NVNcommandBuffer*, NVNshaderStage, Rest...)> {
    static void call(NVNcommandBuffer* command_buffer, NVNshaderStage stage, Rest... rest) noexcept {
        if (const auto api55 = implemented_entry(E)) {
            reinterpret_cast<typename nvn::EntrySignature<E>::type>(api55)(command_buffer, stage,
                                                                         rest...);
            return;
        }
        const auto guest_return = ARMIGO_GUEST_RETURN;
        char detail[128];
        const auto name = nvn::entry_name(E);
        std::snprintf(detail, sizeof detail, "Armigo NVN 231.40: %.*s has no API 55 implementation",
                      static_cast<int>(name.size()), name.data());
        unsupported(detail, guest_return);
    }
};

// Program bindings retain all logical stage bits independently of the
// physical task/mesh execution slots.
void command_buffer_bind_program(NVNcommandBuffer* command_buffer, const NVNprogram* program, int stages) noexcept {
    if (const auto api55 = implemented_entry(nvn::Entry::nvnCommandBufferBindProgram)) {
        reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnCommandBufferBindProgram>::type>(api55)(
            command_buffer, program, stages);
        return;
    }
    const auto guest_return = ARMIGO_GUEST_RETURN;
    unsupported("Armigo NVN 231.40: nvnCommandBufferBindProgram has no API 55 implementation", guest_return);
}

// Graphics push constants, as the title's driver emits them: at most 256
// bytes in whole words, written at byte offset (offset & 0xff) of the
// stage's push block, and fewer than 4 bytes change nothing. Shaders read the
// block as hardware constant buffer 16.
void command_buffer_push_graphics_constants(NVNcommandBuffer* command_buffer, NVNshaderStage stage,
                                            std::uint32_t offset, std::uint32_t size,
                                            const void* data) noexcept {
    const auto bytes = std::min(size, nvn::push_constant_bytes) & ~3u;
    if (!bytes) return;
    if (stage == NVN_SHADER_STAGE_COMPUTE) {
        nvn::command_refuse(command_buffer, "nvnCommandBufferPushGraphicsConstants",
                            "compute pushes go through nvnCommandBufferPushComputeConstants");
        return;
    }
    nvn::command_push_constants(command_buffer, stage, nvn::push_constant_bank, offset & 0xffu, bytes, data,
                                "nvnCommandBufferPushGraphicsConstants");
}

// Compute push constants: the same block semantics for compute's own block
// (whole words, at most 256 bytes, fewer than 4 change nothing). The first
// argument selects a 256-byte block in the title's driver; what it names is
// not established, so a nonzero value is reported once and the bytes still
// go to compute's single block.
void command_buffer_push_compute_constants(NVNcommandBuffer* command_buffer, std::uint32_t block,
                                           std::uint32_t offset, std::uint32_t size,
                                           const void* data) noexcept {
    if (block) {
        static std::once_flag reported;
        std::call_once(reported, [block] {
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN 231.40: nvnCommandBufferPushComputeConstants block %u is not established; "
                                "its bytes go to compute's push block\n", block);
        });
    }
    const auto bytes = std::min(size, nvn::push_constant_bytes) & ~3u;
    if (!bytes) return;
    nvn::command_push_constants(command_buffer, NVN_SHADER_STAGE_COMPUTE, nvn::compute_push_constant_bank, offset,
                                bytes, data, "nvnCommandBufferPushComputeConstants");
}

// Mesh draws, recorded with the title driver's arguments: DrawMeshTasks(first,
// count), which its driver emits as (count, first), and DrawMeshTasksIndirect
// (argument address). The replay draws them through the mesh pipeline with the
// translated task and mesh stages.
void command_buffer_draw_mesh_tasks(NVNcommandBuffer* command_buffer, int first, int count) noexcept {
    nvn::command_draw_mesh_tasks_at(command_buffer, first, count, ARMIGO_GUEST_RETURN);
}
void command_buffer_draw_mesh_tasks_indirect(NVNcommandBuffer* command_buffer, NVNbufferAddress address) noexcept {
    nvn::command_draw_mesh_tasks_indirect(command_buffer, address);
}

// Controls with no effect on what the commands produce: color reduction and
// stencil culling are compression and culling optimizations, and compute
// overlap and compute workload priority schedule work that the replay runs in submission order on one Metal queue. The
// conservative rasterization mode applies only while conservative
// rasterization is enabled, which is its own command. The title's driver
// accepts every value (it masks or clamps each into its GPU methods), so these
// do too; each logs its first value once.
template <std::size_t Control>
void log_control_once(const char* name, long long value) noexcept {
    static std::once_flag logged;
    std::call_once(logged, [&] {
        armigo::diag::write(armigo::diag::Level::info, "Armigo NVN 231.40: %s %lld has no effect on Metal\n",
                            name, value);
    });
}
void command_buffer_set_color_reduction_enable(NVNcommandBuffer*, std::uint32_t enable) noexcept {
    log_control_once<0>("nvnCommandBufferSetColorReductionEnable", enable & 0xff);
}
void command_buffer_set_color_reduction_thresholds(NVNcommandBuffer*, std::int32_t format_class, std::int32_t,
                                                   std::int32_t) noexcept {
    log_control_once<1>("nvnCommandBufferSetColorReductionThresholds", format_class);
}
void command_buffer_set_compute_overlap_mode(NVNcommandBuffer*, std::int32_t mode) noexcept {
    log_control_once<2>("nvnCommandBufferSetComputeOverlapMode", mode);
}
void command_buffer_set_compute_workload_priority(NVNcommandBuffer*, std::int32_t priority) noexcept {
    log_control_once<3>("nvnCommandBufferSetComputeWorkloadPriority", priority);
}
void command_buffer_set_geometry_processing_mode(NVNcommandBuffer* command_buffer, std::int32_t mode) noexcept {
    nvn::command_geometry_processing_mode(command_buffer, mode);
}
void command_buffer_set_stencil_cull_criteria(NVNcommandBuffer*, std::int32_t function, std::int32_t,
                                              std::int32_t) noexcept {
    log_control_once<5>("nvnCommandBufferSetStencilCullCriteria", function);
}
void command_buffer_set_conservative_raster_mode(NVNcommandBuffer*, std::int32_t mode, std::int32_t) noexcept {
    log_control_once<6>("nvnCommandBufferSetConservativeRasterMode", mode);
}

// Variable-rate shading: the driver accepts a shading-rate image, its palette
// and the palette mode and never applies a rate, so every fragment shades at
// full rate. That is the disabled pixel-rate state the native shader
// translation assumes (native stages may then omit their rate selector).
void command_buffer_bind_shading_rate_image(NVNcommandBuffer*, const NVNtextureView*, const NVNtexture*) noexcept {
    log_control_once<7>("nvnCommandBufferBindShadingRateImage", 0);
}
void command_buffer_set_shading_rate_palette_mode(NVNcommandBuffer*, std::int32_t mode) noexcept {
    log_control_once<8>("nvnCommandBufferSetShadingRatePaletteMode", mode);
}
void command_buffer_shading_rate_image_palette(NVNcommandBuffer*, std::int32_t viewport, std::int32_t,
                                               const void*) noexcept {
    log_control_once<9>("nvnCommandBufferShadingRateImagePalette", viewport);
}

// ---------------------------------------------------------------- windows

// API 231 windows may present from fewer textures than they hold, switching
// between double and triple buffering. The builder keeps the count after the
// API 55 builder state; zero leaves every texture active.
constexpr std::size_t window_builder_active_offset = sizeof(WindowBuilderState);
static_assert(window_builder_active_offset + sizeof(std::int32_t) <= 0x70);

std::int32_t& window_builder_active(void* builder) noexcept {
    return *reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(builder) + window_builder_active_offset);
}

void window_builder_set_defaults(NVNwindowBuilder* builder) noexcept {
    if (const auto api55 = implemented_entry(nvn::Entry::nvnWindowBuilderSetDefaults))
        reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnWindowBuilderSetDefaults>::type>(api55)(builder);
    if (builder) window_builder_active(builder) = 0;
}

void window_builder_set_num_active_textures(NVNwindowBuilder* builder, int count) noexcept {
    if (builder) window_builder_active(builder) = count;
}

NVNboolean window_initialize(NVNwindow* window, const NVNwindowBuilder* builder) noexcept {
    const auto api55 = implemented_entry(nvn::Entry::nvnWindowInitialize);
    const auto guest_return = ARMIGO_GUEST_RETURN;
    if (!api55) {
        unsupported("Armigo NVN 231.40: nvnWindowInitialize has no window implementation", guest_return);
        return 0;
    }
    if (!reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnWindowInitialize>::type>(api55)(window, builder))
        return 0;
    const auto active = window_builder_active(const_cast<NVNwindowBuilder*>(builder));
    if (active && !on_host_stack([&] { return set_window_active_textures(window, active); })) {
        char detail[112];
        std::snprintf(detail, sizeof detail, "Armigo NVN 231.40: a window cannot present from %d active textures",
                      active);
        unsupported(detail, guest_return);
        return 0;
    }
    return 1;
}

void window_set_num_active_textures(NVNwindow* window, int count) noexcept {
    const auto guest_return = ARMIGO_GUEST_RETURN;
    if (on_host_stack([&] { return set_window_active_textures(window, count); })) return;
    char detail[112];
    std::snprintf(detail, sizeof detail, "Armigo NVN 231.40: a window cannot present from %d active textures", count);
    unsupported(detail, guest_return);
}

// Entry points both APIs name whose API 55 signature matches the title's API 231
// driver (its argument registers, checked per entry point), served by the
// API 55 implementation when there is one.
constexpr std::pair<Entry, nvn::Entry> shared_entries[] = {
    // Device: the driver's builder defaults write 24 bytes; the device is 0x3000.
    {Entry::nvnDeviceBuilderGetFlags, nvn::Entry::nvnDeviceBuilderGetFlags},
    {Entry::nvnDeviceBuilderSetDefaults, nvn::Entry::nvnDeviceBuilderSetDefaults},
    {Entry::nvnDeviceBuilderSetFlags, nvn::Entry::nvnDeviceBuilderSetFlags},
    {Entry::nvnDeviceFinalize, nvn::Entry::nvnDeviceFinalize},
    {Entry::nvnDeviceFinalizeCommandHandle, nvn::Entry::nvnDeviceFinalizeCommandHandle},
    {Entry::nvnDeviceGenerateDebugDomainId, nvn::Entry::nvnDeviceGenerateDebugDomainId},
    {Entry::nvnDeviceGetCurrentTimestampInNanoseconds, nvn::Entry::nvnDeviceGetCurrentTimestampInNanoseconds},
    {Entry::nvnDeviceGetDepthMode, nvn::Entry::nvnDeviceGetDepthMode},
    {Entry::nvnDeviceGetSeparateSamplerHandle, nvn::Entry::nvnDeviceGetSeparateSamplerHandle},
    {Entry::nvnDeviceGetSeparateTextureHandle, nvn::Entry::nvnDeviceGetSeparateTextureHandle},
    {Entry::nvnDeviceGetTimestampInNanoseconds, nvn::Entry::nvnDeviceGetTimestampInNanoseconds},
    {Entry::nvnDeviceGetWindowOriginMode, nvn::Entry::nvnDeviceGetWindowOriginMode},
    {Entry::nvnDeviceInitialize, nvn::Entry::nvnDeviceInitialize},
    {Entry::nvnDeviceInstallDebugCallback, nvn::Entry::nvnDeviceInstallDebugCallback},
    {Entry::nvnDeviceIsExternalDebuggerAttached, nvn::Entry::nvnDeviceIsExternalDebuggerAttached},
    {Entry::nvnDeviceRegisterFastClearColor, nvn::Entry::nvnDeviceRegisterFastClearColor},
    {Entry::nvnDeviceRegisterFastClearColori, nvn::Entry::nvnDeviceRegisterFastClearColori},
    {Entry::nvnDeviceRegisterFastClearColorui, nvn::Entry::nvnDeviceRegisterFastClearColorui},
    {Entry::nvnDeviceRegisterFastClearDepth, nvn::Entry::nvnDeviceRegisterFastClearDepth},
    {Entry::nvnDeviceSetDebugLabel, nvn::Entry::nvnDeviceSetDebugLabel},
    {Entry::nvnDeviceSetDepthMode, nvn::Entry::nvnDeviceSetDepthMode},
    {Entry::nvnDeviceSetIntermediateShaderCache, nvn::Entry::nvnDeviceSetIntermediateShaderCache},
    {Entry::nvnDeviceSetWindowOriginMode, nvn::Entry::nvnDeviceSetWindowOriginMode},
    {Entry::nvnDeviceWalkDebugDatabase, nvn::Entry::nvnDeviceWalkDebugDatabase},
    // Memory pools: builder defaults write 0x38 bytes; a pool is 0x100.
    {Entry::nvnMemoryPoolBuilderGetDevice, nvn::Entry::nvnMemoryPoolBuilderGetDevice},
    {Entry::nvnMemoryPoolBuilderGetFlags, nvn::Entry::nvnMemoryPoolBuilderGetFlags},
    {Entry::nvnMemoryPoolBuilderGetMemory, nvn::Entry::nvnMemoryPoolBuilderGetMemory},
    {Entry::nvnMemoryPoolBuilderGetSize, nvn::Entry::nvnMemoryPoolBuilderGetSize},
    {Entry::nvnMemoryPoolBuilderSetDefaults, nvn::Entry::nvnMemoryPoolBuilderSetDefaults},
    {Entry::nvnMemoryPoolBuilderSetDevice, nvn::Entry::nvnMemoryPoolBuilderSetDevice},
    {Entry::nvnMemoryPoolBuilderSetFlags, nvn::Entry::nvnMemoryPoolBuilderSetFlags},
    {Entry::nvnMemoryPoolBuilderSetStorage, nvn::Entry::nvnMemoryPoolBuilderSetStorage},
    {Entry::nvnMemoryPoolFlushMappedRange, nvn::Entry::nvnMemoryPoolFlushMappedRange},
    {Entry::nvnMemoryPoolGetBufferAddress, nvn::Entry::nvnMemoryPoolGetBufferAddress},
    {Entry::nvnMemoryPoolGetFlags, nvn::Entry::nvnMemoryPoolGetFlags},
    {Entry::nvnMemoryPoolGetSize, nvn::Entry::nvnMemoryPoolGetSize},
    {Entry::nvnMemoryPoolInvalidateMappedRange, nvn::Entry::nvnMemoryPoolInvalidateMappedRange},
    {Entry::nvnMemoryPoolMap, nvn::Entry::nvnMemoryPoolMap},
    {Entry::nvnMemoryPoolSetDebugLabel, nvn::Entry::nvnMemoryPoolSetDebugLabel},
    // Buffers: builder defaults write 0x28 bytes; a buffer is 0x30.
    {Entry::nvnBufferBuilderGetDevice, nvn::Entry::nvnBufferBuilderGetDevice},
    {Entry::nvnBufferBuilderGetMemoryOffset, nvn::Entry::nvnBufferBuilderGetMemoryOffset},
    {Entry::nvnBufferBuilderGetMemoryPool, nvn::Entry::nvnBufferBuilderGetMemoryPool},
    {Entry::nvnBufferBuilderGetSize, nvn::Entry::nvnBufferBuilderGetSize},
    {Entry::nvnBufferBuilderSetDefaults, nvn::Entry::nvnBufferBuilderSetDefaults},
    {Entry::nvnBufferBuilderSetDevice, nvn::Entry::nvnBufferBuilderSetDevice},
    {Entry::nvnBufferBuilderSetStorage, nvn::Entry::nvnBufferBuilderSetStorage},
    {Entry::nvnBufferFinalize, nvn::Entry::nvnBufferFinalize},
    {Entry::nvnBufferFlushMappedRange, nvn::Entry::nvnBufferFlushMappedRange},
    {Entry::nvnBufferGetAddress, nvn::Entry::nvnBufferGetAddress},
    {Entry::nvnBufferGetDebugID, nvn::Entry::nvnBufferGetDebugID},
    {Entry::nvnBufferGetMemoryOffset, nvn::Entry::nvnBufferGetMemoryOffset},
    {Entry::nvnBufferGetMemoryPool, nvn::Entry::nvnBufferGetMemoryPool},
    {Entry::nvnBufferGetSize, nvn::Entry::nvnBufferGetSize},
    {Entry::nvnBufferInitialize, nvn::Entry::nvnBufferInitialize},
    {Entry::nvnBufferInvalidateMappedRange, nvn::Entry::nvnBufferInvalidateMappedRange},
    {Entry::nvnBufferMap, nvn::Entry::nvnBufferMap},
    {Entry::nvnBufferSetDebugLabel, nvn::Entry::nvnBufferSetDebugLabel},
    // Descriptor pools: each initializes 0x20 bytes.
    {Entry::nvnTexturePoolFinalize, nvn::Entry::nvnTexturePoolFinalize},
    {Entry::nvnTexturePoolGetMemoryOffset, nvn::Entry::nvnTexturePoolGetMemoryOffset},
    {Entry::nvnTexturePoolGetMemoryPool, nvn::Entry::nvnTexturePoolGetMemoryPool},
    {Entry::nvnTexturePoolGetSize, nvn::Entry::nvnTexturePoolGetSize},
    {Entry::nvnTexturePoolInitialize, nvn::Entry::nvnTexturePoolInitialize},
    {Entry::nvnTexturePoolRegisterImage, nvn::Entry::nvnTexturePoolRegisterImage},
    {Entry::nvnTexturePoolRegisterTexture, nvn::Entry::nvnTexturePoolRegisterTexture},
    {Entry::nvnTexturePoolSetDebugLabel, nvn::Entry::nvnTexturePoolSetDebugLabel},
    {Entry::nvnSamplerPoolFinalize, nvn::Entry::nvnSamplerPoolFinalize},
    {Entry::nvnSamplerPoolGetMemoryOffset, nvn::Entry::nvnSamplerPoolGetMemoryOffset},
    {Entry::nvnSamplerPoolGetMemoryPool, nvn::Entry::nvnSamplerPoolGetMemoryPool},
    {Entry::nvnSamplerPoolGetSize, nvn::Entry::nvnSamplerPoolGetSize},
    {Entry::nvnSamplerPoolInitialize, nvn::Entry::nvnSamplerPoolInitialize},
    {Entry::nvnSamplerPoolRegisterSampler, nvn::Entry::nvnSamplerPoolRegisterSampler},
    {Entry::nvnSamplerPoolRegisterSamplerBuilder, nvn::Entry::nvnSamplerPoolRegisterSamplerBuilder},
    {Entry::nvnSamplerPoolSetDebugLabel, nvn::Entry::nvnSamplerPoolSetDebugLabel},
    // Textures: builder defaults write 0x68 bytes, a texture is 0x80, a view 0x10.
    {Entry::nvnTextureBuilderGetDepth, nvn::Entry::nvnTextureBuilderGetDepth},
    {Entry::nvnTextureBuilderGetDepthStencilMode, nvn::Entry::nvnTextureBuilderGetDepthStencilMode},
    {Entry::nvnTextureBuilderGetDevice, nvn::Entry::nvnTextureBuilderGetDevice},
    {Entry::nvnTextureBuilderGetFlags, nvn::Entry::nvnTextureBuilderGetFlags},
    {Entry::nvnTextureBuilderGetFormat, nvn::Entry::nvnTextureBuilderGetFormat},
    {Entry::nvnTextureBuilderGetGLTextureName, nvn::Entry::nvnTextureBuilderGetGLTextureName},
    {Entry::nvnTextureBuilderGetHeight, nvn::Entry::nvnTextureBuilderGetHeight},
    {Entry::nvnTextureBuilderGetLevels, nvn::Entry::nvnTextureBuilderGetLevels},
    {Entry::nvnTextureBuilderGetMemoryOffset, nvn::Entry::nvnTextureBuilderGetMemoryOffset},
    {Entry::nvnTextureBuilderGetMemoryPool, nvn::Entry::nvnTextureBuilderGetMemoryPool},
    {Entry::nvnTextureBuilderGetPackagedTextureData, nvn::Entry::nvnTextureBuilderGetPackagedTextureData},
    {Entry::nvnTextureBuilderGetSamples, nvn::Entry::nvnTextureBuilderGetSamples},
    {Entry::nvnTextureBuilderGetSparseTileLayout, nvn::Entry::nvnTextureBuilderGetSparseTileLayout},
    {Entry::nvnTextureBuilderGetStorageAlignment, nvn::Entry::nvnTextureBuilderGetStorageAlignment},
    {Entry::nvnTextureBuilderGetStorageClass, nvn::Entry::nvnTextureBuilderGetStorageClass},
    {Entry::nvnTextureBuilderGetStorageSize, nvn::Entry::nvnTextureBuilderGetStorageSize},
    {Entry::nvnTextureBuilderGetStride, nvn::Entry::nvnTextureBuilderGetStride},
    {Entry::nvnTextureBuilderGetSwizzle, nvn::Entry::nvnTextureBuilderGetSwizzle},
    {Entry::nvnTextureBuilderGetTarget, nvn::Entry::nvnTextureBuilderGetTarget},
    {Entry::nvnTextureBuilderGetWidth, nvn::Entry::nvnTextureBuilderGetWidth},
    {Entry::nvnTextureBuilderGetZCullStorageSize, nvn::Entry::nvnTextureBuilderGetZCullStorageSize},
    {Entry::nvnTextureBuilderSetDefaults, nvn::Entry::nvnTextureBuilderSetDefaults},
    {Entry::nvnTextureBuilderSetDepth, nvn::Entry::nvnTextureBuilderSetDepth},
    {Entry::nvnTextureBuilderSetDepthStencilMode, nvn::Entry::nvnTextureBuilderSetDepthStencilMode},
    {Entry::nvnTextureBuilderSetDevice, nvn::Entry::nvnTextureBuilderSetDevice},
    {Entry::nvnTextureBuilderSetFlags, nvn::Entry::nvnTextureBuilderSetFlags},
    {Entry::nvnTextureBuilderSetFormat, nvn::Entry::nvnTextureBuilderSetFormat},
    {Entry::nvnTextureBuilderSetGLTextureName, nvn::Entry::nvnTextureBuilderSetGLTextureName},
    {Entry::nvnTextureBuilderSetHeight, nvn::Entry::nvnTextureBuilderSetHeight},
    {Entry::nvnTextureBuilderSetLevels, nvn::Entry::nvnTextureBuilderSetLevels},
    {Entry::nvnTextureBuilderSetPackagedTextureData, nvn::Entry::nvnTextureBuilderSetPackagedTextureData},
    {Entry::nvnTextureBuilderSetPackagedTextureLayout, nvn::Entry::nvnTextureBuilderSetPackagedTextureLayout},
    {Entry::nvnTextureBuilderSetSamples, nvn::Entry::nvnTextureBuilderSetSamples},
    {Entry::nvnTextureBuilderSetSize1D, nvn::Entry::nvnTextureBuilderSetSize1D},
    {Entry::nvnTextureBuilderSetSize2D, nvn::Entry::nvnTextureBuilderSetSize2D},
    {Entry::nvnTextureBuilderSetSize3D, nvn::Entry::nvnTextureBuilderSetSize3D},
    {Entry::nvnTextureBuilderSetStorage, nvn::Entry::nvnTextureBuilderSetStorage},
    {Entry::nvnTextureBuilderSetStride, nvn::Entry::nvnTextureBuilderSetStride},
    {Entry::nvnTextureBuilderSetSwizzle, nvn::Entry::nvnTextureBuilderSetSwizzle},
    {Entry::nvnTextureBuilderSetTarget, nvn::Entry::nvnTextureBuilderSetTarget},
    {Entry::nvnTextureBuilderSetWidth, nvn::Entry::nvnTextureBuilderSetWidth},
    {Entry::nvnTextureViewCompare, nvn::Entry::nvnTextureViewCompare},
    {Entry::nvnTextureViewGetDepthStencilMode, nvn::Entry::nvnTextureViewGetDepthStencilMode},
    {Entry::nvnTextureViewGetFormat, nvn::Entry::nvnTextureViewGetFormat},
    {Entry::nvnTextureViewGetLayers, nvn::Entry::nvnTextureViewGetLayers},
    {Entry::nvnTextureViewGetLevels, nvn::Entry::nvnTextureViewGetLevels},
    {Entry::nvnTextureViewGetSwizzle, nvn::Entry::nvnTextureViewGetSwizzle},
    {Entry::nvnTextureViewGetTarget, nvn::Entry::nvnTextureViewGetTarget},
    {Entry::nvnTextureViewSetDefaults, nvn::Entry::nvnTextureViewSetDefaults},
    {Entry::nvnTextureViewSetDepthStencilMode, nvn::Entry::nvnTextureViewSetDepthStencilMode},
    {Entry::nvnTextureViewSetFormat, nvn::Entry::nvnTextureViewSetFormat},
    {Entry::nvnTextureViewSetLayers, nvn::Entry::nvnTextureViewSetLayers},
    {Entry::nvnTextureViewSetLevels, nvn::Entry::nvnTextureViewSetLevels},
    {Entry::nvnTextureViewSetSwizzle, nvn::Entry::nvnTextureViewSetSwizzle},
    {Entry::nvnTextureViewSetTarget, nvn::Entry::nvnTextureViewSetTarget},
    {Entry::nvnTextureCompare, nvn::Entry::nvnTextureCompare},
    {Entry::nvnTextureFinalize, nvn::Entry::nvnTextureFinalize},
    {Entry::nvnTextureFlushTexels, nvn::Entry::nvnTextureFlushTexels},
    {Entry::nvnTextureGetDebugID, nvn::Entry::nvnTextureGetDebugID},
    {Entry::nvnTextureGetDepth, nvn::Entry::nvnTextureGetDepth},
    {Entry::nvnTextureGetDepthStencilMode, nvn::Entry::nvnTextureGetDepthStencilMode},
    {Entry::nvnTextureGetFlags, nvn::Entry::nvnTextureGetFlags},
    {Entry::nvnTextureGetFormat, nvn::Entry::nvnTextureGetFormat},
    {Entry::nvnTextureGetHeight, nvn::Entry::nvnTextureGetHeight},
    {Entry::nvnTextureGetLevels, nvn::Entry::nvnTextureGetLevels},
    {Entry::nvnTextureGetMemoryOffset, nvn::Entry::nvnTextureGetMemoryOffset},
    {Entry::nvnTextureGetMemoryPool, nvn::Entry::nvnTextureGetMemoryPool},
    {Entry::nvnTextureGetSamples, nvn::Entry::nvnTextureGetSamples},
    {Entry::nvnTextureGetSparseTileLayout, nvn::Entry::nvnTextureGetSparseTileLayout},
    {Entry::nvnTextureGetStorageClass, nvn::Entry::nvnTextureGetStorageClass},
    {Entry::nvnTextureGetStorageSize, nvn::Entry::nvnTextureGetStorageSize},
    {Entry::nvnTextureGetStride, nvn::Entry::nvnTextureGetStride},
    {Entry::nvnTextureGetSwizzle, nvn::Entry::nvnTextureGetSwizzle},
    {Entry::nvnTextureGetTarget, nvn::Entry::nvnTextureGetTarget},
    {Entry::nvnTextureGetTextureAddress, nvn::Entry::nvnTextureGetTextureAddress},
    {Entry::nvnTextureGetViewOffset, nvn::Entry::nvnTextureGetViewOffset},
    {Entry::nvnTextureGetWidth, nvn::Entry::nvnTextureGetWidth},
    {Entry::nvnTextureGetZCullStorageSize, nvn::Entry::nvnTextureGetZCullStorageSize},
    {Entry::nvnTextureInitialize, nvn::Entry::nvnTextureInitialize},
    {Entry::nvnTextureInvalidateTexels, nvn::Entry::nvnTextureInvalidateTexels},
    {Entry::nvnTextureReadTexels, nvn::Entry::nvnTextureReadTexels},
    {Entry::nvnTextureReadTexelsStrided, nvn::Entry::nvnTextureReadTexelsStrided},
    {Entry::nvnTextureSetDebugLabel, nvn::Entry::nvnTextureSetDebugLabel},
    {Entry::nvnTextureWriteTexels, nvn::Entry::nvnTextureWriteTexels},
    {Entry::nvnTextureWriteTexelsStrided, nvn::Entry::nvnTextureWriteTexelsStrided},
    // Samplers: builders take 0x40 bytes; a sampler is 0x40.
    {Entry::nvnSamplerBuilderGetBorderColor, nvn::Entry::nvnSamplerBuilderGetBorderColor},
    {Entry::nvnSamplerBuilderGetBorderColori, nvn::Entry::nvnSamplerBuilderGetBorderColori},
    {Entry::nvnSamplerBuilderGetBorderColorui, nvn::Entry::nvnSamplerBuilderGetBorderColorui},
    {Entry::nvnSamplerBuilderGetCompare, nvn::Entry::nvnSamplerBuilderGetCompare},
    {Entry::nvnSamplerBuilderGetDevice, nvn::Entry::nvnSamplerBuilderGetDevice},
    {Entry::nvnSamplerBuilderGetLodBias, nvn::Entry::nvnSamplerBuilderGetLodBias},
    {Entry::nvnSamplerBuilderGetLodClamp, nvn::Entry::nvnSamplerBuilderGetLodClamp},
    {Entry::nvnSamplerBuilderGetLodSnap, nvn::Entry::nvnSamplerBuilderGetLodSnap},
    {Entry::nvnSamplerBuilderGetMaxAnisotropy, nvn::Entry::nvnSamplerBuilderGetMaxAnisotropy},
    {Entry::nvnSamplerBuilderGetMinMagFilter, nvn::Entry::nvnSamplerBuilderGetMinMagFilter},
    {Entry::nvnSamplerBuilderGetReductionFilter, nvn::Entry::nvnSamplerBuilderGetReductionFilter},
    {Entry::nvnSamplerBuilderGetWrapMode, nvn::Entry::nvnSamplerBuilderGetWrapMode},
    {Entry::nvnSamplerBuilderSetBorderColor, nvn::Entry::nvnSamplerBuilderSetBorderColor},
    {Entry::nvnSamplerBuilderSetBorderColori, nvn::Entry::nvnSamplerBuilderSetBorderColori},
    {Entry::nvnSamplerBuilderSetBorderColorui, nvn::Entry::nvnSamplerBuilderSetBorderColorui},
    {Entry::nvnSamplerBuilderSetCompare, nvn::Entry::nvnSamplerBuilderSetCompare},
    {Entry::nvnSamplerBuilderSetDefaults, nvn::Entry::nvnSamplerBuilderSetDefaults},
    {Entry::nvnSamplerBuilderSetDevice, nvn::Entry::nvnSamplerBuilderSetDevice},
    {Entry::nvnSamplerBuilderSetLodBias, nvn::Entry::nvnSamplerBuilderSetLodBias},
    {Entry::nvnSamplerBuilderSetLodClamp, nvn::Entry::nvnSamplerBuilderSetLodClamp},
    {Entry::nvnSamplerBuilderSetLodSnap, nvn::Entry::nvnSamplerBuilderSetLodSnap},
    {Entry::nvnSamplerBuilderSetMaxAnisotropy, nvn::Entry::nvnSamplerBuilderSetMaxAnisotropy},
    {Entry::nvnSamplerBuilderSetMinMagFilter, nvn::Entry::nvnSamplerBuilderSetMinMagFilter},
    {Entry::nvnSamplerBuilderSetReductionFilter, nvn::Entry::nvnSamplerBuilderSetReductionFilter},
    {Entry::nvnSamplerBuilderSetWrapMode, nvn::Entry::nvnSamplerBuilderSetWrapMode},
    {Entry::nvnSamplerCompare, nvn::Entry::nvnSamplerCompare},
    {Entry::nvnSamplerFinalize, nvn::Entry::nvnSamplerFinalize},
    {Entry::nvnSamplerGetBorderColor, nvn::Entry::nvnSamplerGetBorderColor},
    {Entry::nvnSamplerGetBorderColori, nvn::Entry::nvnSamplerGetBorderColori},
    {Entry::nvnSamplerGetBorderColorui, nvn::Entry::nvnSamplerGetBorderColorui},
    {Entry::nvnSamplerGetCompare, nvn::Entry::nvnSamplerGetCompare},
    {Entry::nvnSamplerGetDebugID, nvn::Entry::nvnSamplerGetDebugID},
    {Entry::nvnSamplerGetLodBias, nvn::Entry::nvnSamplerGetLodBias},
    {Entry::nvnSamplerGetLodClamp, nvn::Entry::nvnSamplerGetLodClamp},
    {Entry::nvnSamplerGetMaxAnisotropy, nvn::Entry::nvnSamplerGetMaxAnisotropy},
    {Entry::nvnSamplerGetMinMagFilter, nvn::Entry::nvnSamplerGetMinMagFilter},
    {Entry::nvnSamplerGetReductionFilter, nvn::Entry::nvnSamplerGetReductionFilter},
    {Entry::nvnSamplerGetWrapMode, nvn::Entry::nvnSamplerGetWrapMode},
    {Entry::nvnSamplerInitialize, nvn::Entry::nvnSamplerInitialize},
    {Entry::nvnSamplerSetDebugLabel, nvn::Entry::nvnSamplerSetDebugLabel},
    // Programs: initialization clears 0xa0 bytes; shader data stay 16-byte records.
    {Entry::nvnProgramFinalize, nvn::Entry::nvnProgramFinalize},
    {Entry::nvnProgramInitialize, nvn::Entry::nvnProgramInitialize},
    {Entry::nvnProgramSetDebugLabel, nvn::Entry::nvnProgramSetDebugLabel},
    {Entry::nvnProgramSetShaders, nvn::Entry::nvnProgramSetShaders},
    {Entry::nvnProgramSetShadersExt, nvn::Entry::nvnProgramSetShadersExt},
    {Entry::nvnProgramSetSubroutineLinkage, nvn::Entry::nvnProgramSetSubroutineLinkage},
    // Queues: builder defaults write 0x48 bytes; a queue is larger than 0x2cb8.
    {Entry::nvnQueueBuilderGetQueueMemorySize, nvn::Entry::nvnQueueBuilderGetQueueMemorySize},
    {Entry::nvnQueueBuilderSetCommandFlushThreshold, nvn::Entry::nvnQueueBuilderSetCommandFlushThreshold},
    {Entry::nvnQueueBuilderSetCommandMemorySize, nvn::Entry::nvnQueueBuilderSetCommandMemorySize},
    {Entry::nvnQueueBuilderSetComputeMemorySize, nvn::Entry::nvnQueueBuilderSetComputeMemorySize},
    {Entry::nvnQueueBuilderSetControlMemorySize, nvn::Entry::nvnQueueBuilderSetControlMemorySize},
    {Entry::nvnQueueBuilderSetDefaults, nvn::Entry::nvnQueueBuilderSetDefaults},
    {Entry::nvnQueueBuilderSetDevice, nvn::Entry::nvnQueueBuilderSetDevice},
    {Entry::nvnQueueBuilderSetFlags, nvn::Entry::nvnQueueBuilderSetFlags},
    {Entry::nvnQueueBuilderSetQueueMemory, nvn::Entry::nvnQueueBuilderSetQueueMemory},
    {Entry::nvnQueueFenceSync, nvn::Entry::nvnQueueFenceSync},
    {Entry::nvnQueueFinalize, nvn::Entry::nvnQueueFinalize},
    {Entry::nvnQueueFinish, nvn::Entry::nvnQueueFinish},
    {Entry::nvnQueueFlush, nvn::Entry::nvnQueueFlush},
    {Entry::nvnQueueGetError, nvn::Entry::nvnQueueGetError},
    {Entry::nvnQueueGetTotalCommandMemoryUsed, nvn::Entry::nvnQueueGetTotalCommandMemoryUsed},
    {Entry::nvnQueueGetTotalComputeMemoryUsed, nvn::Entry::nvnQueueGetTotalComputeMemoryUsed},
    {Entry::nvnQueueGetTotalControlMemoryUsed, nvn::Entry::nvnQueueGetTotalControlMemoryUsed},
    {Entry::nvnQueueInitialize, nvn::Entry::nvnQueueInitialize},
    {Entry::nvnQueuePresentTexture, nvn::Entry::nvnQueuePresentTexture},
    {Entry::nvnQueueResetMemoryUsageCounts, nvn::Entry::nvnQueueResetMemoryUsageCounts},
    {Entry::nvnQueueSetDebugLabel, nvn::Entry::nvnQueueSetDebugLabel},
    {Entry::nvnQueueSubmitCommands, nvn::Entry::nvnQueueSubmitCommands},
    {Entry::nvnQueueWaitSync, nvn::Entry::nvnQueueWaitSync},
    // Sync objects initialize 0x20 bytes, events 0x28 from 0x10-byte builders.
    {Entry::nvnSyncFinalize, nvn::Entry::nvnSyncFinalize},
    {Entry::nvnSyncInitialize, nvn::Entry::nvnSyncInitialize},
    {Entry::nvnSyncSetDebugLabel, nvn::Entry::nvnSyncSetDebugLabel},
    {Entry::nvnSyncWait, nvn::Entry::nvnSyncWait},
    {Entry::nvnEventBuilderSetDefaults, nvn::Entry::nvnEventBuilderSetDefaults},
    {Entry::nvnEventBuilderSetStorage, nvn::Entry::nvnEventBuilderSetStorage},
    {Entry::nvnEventFinalize, nvn::Entry::nvnEventFinalize},
    {Entry::nvnEventGetValue, nvn::Entry::nvnEventGetValue},
    {Entry::nvnEventInitialize, nvn::Entry::nvnEventInitialize},
    {Entry::nvnEventSignal, nvn::Entry::nvnEventSignal},
    // Windows: builder defaults write 0x70 bytes; a window exceeds 0x1c0.
    {Entry::nvnWindowBuilderGetNativeWindow, nvn::Entry::nvnWindowBuilderGetNativeWindow},
    {Entry::nvnWindowBuilderGetPresentInterval, nvn::Entry::nvnWindowBuilderGetPresentInterval},
    {Entry::nvnWindowBuilderSetDevice, nvn::Entry::nvnWindowBuilderSetDevice},
    {Entry::nvnWindowBuilderSetNativeWindow, nvn::Entry::nvnWindowBuilderSetNativeWindow},
    {Entry::nvnWindowBuilderSetPresentInterval, nvn::Entry::nvnWindowBuilderSetPresentInterval},
    {Entry::nvnWindowBuilderSetTextures, nvn::Entry::nvnWindowBuilderSetTextures},
    {Entry::nvnWindowAcquireTexture, nvn::Entry::nvnWindowAcquireTexture},
    {Entry::nvnWindowFinalize, nvn::Entry::nvnWindowFinalize},
    {Entry::nvnWindowGetCrop, nvn::Entry::nvnWindowGetCrop},
    {Entry::nvnWindowGetNativeWindow, nvn::Entry::nvnWindowGetNativeWindow},
    {Entry::nvnWindowGetPresentInterval, nvn::Entry::nvnWindowGetPresentInterval},
    {Entry::nvnWindowSetCrop, nvn::Entry::nvnWindowSetCrop},
    {Entry::nvnWindowSetDebugLabel, nvn::Entry::nvnWindowSetDebugLabel},
    {Entry::nvnWindowSetPresentInterval, nvn::Entry::nvnWindowSetPresentInterval},
    // State objects: the driver's defaults write the API 55 sizes (blend 8, color 4,
    // channel mask 4, multisample 24, polygon 4, depth-stencil 8, vertex attribute 4,
    // vertex stream 8 bytes).
    {Entry::nvnBlendStateGetAdvancedMode, nvn::Entry::nvnBlendStateGetAdvancedMode},
    {Entry::nvnBlendStateGetAdvancedNormalizedDst, nvn::Entry::nvnBlendStateGetAdvancedNormalizedDst},
    {Entry::nvnBlendStateGetAdvancedOverlap, nvn::Entry::nvnBlendStateGetAdvancedOverlap},
    {Entry::nvnBlendStateGetAdvancedPremultipliedSrc, nvn::Entry::nvnBlendStateGetAdvancedPremultipliedSrc},
    {Entry::nvnBlendStateGetBlendEquation, nvn::Entry::nvnBlendStateGetBlendEquation},
    {Entry::nvnBlendStateGetBlendFunc, nvn::Entry::nvnBlendStateGetBlendFunc},
    {Entry::nvnBlendStateGetBlendTarget, nvn::Entry::nvnBlendStateGetBlendTarget},
    {Entry::nvnBlendStateSetAdvancedMode, nvn::Entry::nvnBlendStateSetAdvancedMode},
    {Entry::nvnBlendStateSetAdvancedNormalizedDst, nvn::Entry::nvnBlendStateSetAdvancedNormalizedDst},
    {Entry::nvnBlendStateSetAdvancedOverlap, nvn::Entry::nvnBlendStateSetAdvancedOverlap},
    {Entry::nvnBlendStateSetAdvancedPremultipliedSrc, nvn::Entry::nvnBlendStateSetAdvancedPremultipliedSrc},
    {Entry::nvnBlendStateSetBlendEquation, nvn::Entry::nvnBlendStateSetBlendEquation},
    {Entry::nvnBlendStateSetBlendFunc, nvn::Entry::nvnBlendStateSetBlendFunc},
    {Entry::nvnBlendStateSetBlendTarget, nvn::Entry::nvnBlendStateSetBlendTarget},
    {Entry::nvnBlendStateSetDefaults, nvn::Entry::nvnBlendStateSetDefaults},
    {Entry::nvnColorStateGetAlphaTest, nvn::Entry::nvnColorStateGetAlphaTest},
    {Entry::nvnColorStateGetBlendEnable, nvn::Entry::nvnColorStateGetBlendEnable},
    {Entry::nvnColorStateGetLogicOp, nvn::Entry::nvnColorStateGetLogicOp},
    {Entry::nvnColorStateSetAlphaTest, nvn::Entry::nvnColorStateSetAlphaTest},
    {Entry::nvnColorStateSetBlendEnable, nvn::Entry::nvnColorStateSetBlendEnable},
    {Entry::nvnColorStateSetDefaults, nvn::Entry::nvnColorStateSetDefaults},
    {Entry::nvnColorStateSetLogicOp, nvn::Entry::nvnColorStateSetLogicOp},
    {Entry::nvnChannelMaskStateGetChannelMask, nvn::Entry::nvnChannelMaskStateGetChannelMask},
    {Entry::nvnChannelMaskStateSetChannelMask, nvn::Entry::nvnChannelMaskStateSetChannelMask},
    {Entry::nvnChannelMaskStateSetDefaults, nvn::Entry::nvnChannelMaskStateSetDefaults},
    {Entry::nvnMultisampleStateGetAlphaToCoverageDither, nvn::Entry::nvnMultisampleStateGetAlphaToCoverageDither},
    {Entry::nvnMultisampleStateGetAlphaToCoverageEnable, nvn::Entry::nvnMultisampleStateGetAlphaToCoverageEnable},
    {Entry::nvnMultisampleStateGetCoverageModulationMode, nvn::Entry::nvnMultisampleStateGetCoverageModulationMode},
    {Entry::nvnMultisampleStateGetCoverageToColorEnable, nvn::Entry::nvnMultisampleStateGetCoverageToColorEnable},
    {Entry::nvnMultisampleStateGetCoverageToColorOutput, nvn::Entry::nvnMultisampleStateGetCoverageToColorOutput},
    {Entry::nvnMultisampleStateGetMultisampleEnable, nvn::Entry::nvnMultisampleStateGetMultisampleEnable},
    {Entry::nvnMultisampleStateGetRasterSamples, nvn::Entry::nvnMultisampleStateGetRasterSamples},
    {Entry::nvnMultisampleStateGetSampleLocationsEnable, nvn::Entry::nvnMultisampleStateGetSampleLocationsEnable},
    {Entry::nvnMultisampleStateGetSampleLocationsGrid, nvn::Entry::nvnMultisampleStateGetSampleLocationsGrid},
    {Entry::nvnMultisampleStateGetSampleLocationsGridEnable, nvn::Entry::nvnMultisampleStateGetSampleLocationsGridEnable},
    {Entry::nvnMultisampleStateGetSamples, nvn::Entry::nvnMultisampleStateGetSamples},
    {Entry::nvnMultisampleStateSetAlphaToCoverageDither, nvn::Entry::nvnMultisampleStateSetAlphaToCoverageDither},
    {Entry::nvnMultisampleStateSetAlphaToCoverageEnable, nvn::Entry::nvnMultisampleStateSetAlphaToCoverageEnable},
    {Entry::nvnMultisampleStateSetCoverageModulationMode, nvn::Entry::nvnMultisampleStateSetCoverageModulationMode},
    {Entry::nvnMultisampleStateSetCoverageToColorEnable, nvn::Entry::nvnMultisampleStateSetCoverageToColorEnable},
    {Entry::nvnMultisampleStateSetCoverageToColorOutput, nvn::Entry::nvnMultisampleStateSetCoverageToColorOutput},
    {Entry::nvnMultisampleStateSetDefaults, nvn::Entry::nvnMultisampleStateSetDefaults},
    {Entry::nvnMultisampleStateSetMultisampleEnable, nvn::Entry::nvnMultisampleStateSetMultisampleEnable},
    {Entry::nvnMultisampleStateSetRasterSamples, nvn::Entry::nvnMultisampleStateSetRasterSamples},
    {Entry::nvnMultisampleStateSetSampleLocations, nvn::Entry::nvnMultisampleStateSetSampleLocations},
    {Entry::nvnMultisampleStateSetSampleLocationsEnable, nvn::Entry::nvnMultisampleStateSetSampleLocationsEnable},
    {Entry::nvnMultisampleStateSetSampleLocationsGridEnable, nvn::Entry::nvnMultisampleStateSetSampleLocationsGridEnable},
    {Entry::nvnMultisampleStateSetSamples, nvn::Entry::nvnMultisampleStateSetSamples},
    {Entry::nvnPolygonStateGetCullFace, nvn::Entry::nvnPolygonStateGetCullFace},
    {Entry::nvnPolygonStateGetFrontFace, nvn::Entry::nvnPolygonStateGetFrontFace},
    {Entry::nvnPolygonStateGetPolygonMode, nvn::Entry::nvnPolygonStateGetPolygonMode},
    {Entry::nvnPolygonStateGetPolygonOffsetEnables, nvn::Entry::nvnPolygonStateGetPolygonOffsetEnables},
    {Entry::nvnPolygonStateSetCullFace, nvn::Entry::nvnPolygonStateSetCullFace},
    {Entry::nvnPolygonStateSetDefaults, nvn::Entry::nvnPolygonStateSetDefaults},
    {Entry::nvnPolygonStateSetFrontFace, nvn::Entry::nvnPolygonStateSetFrontFace},
    {Entry::nvnPolygonStateSetPolygonMode, nvn::Entry::nvnPolygonStateSetPolygonMode},
    {Entry::nvnPolygonStateSetPolygonOffsetEnables, nvn::Entry::nvnPolygonStateSetPolygonOffsetEnables},
    {Entry::nvnDepthStencilStateGetDepthFunc, nvn::Entry::nvnDepthStencilStateGetDepthFunc},
    {Entry::nvnDepthStencilStateGetDepthTestEnable, nvn::Entry::nvnDepthStencilStateGetDepthTestEnable},
    {Entry::nvnDepthStencilStateGetDepthWriteEnable, nvn::Entry::nvnDepthStencilStateGetDepthWriteEnable},
    {Entry::nvnDepthStencilStateGetStencilFunc, nvn::Entry::nvnDepthStencilStateGetStencilFunc},
    {Entry::nvnDepthStencilStateGetStencilOp, nvn::Entry::nvnDepthStencilStateGetStencilOp},
    {Entry::nvnDepthStencilStateGetStencilTestEnable, nvn::Entry::nvnDepthStencilStateGetStencilTestEnable},
    {Entry::nvnDepthStencilStateSetDefaults, nvn::Entry::nvnDepthStencilStateSetDefaults},
    {Entry::nvnDepthStencilStateSetDepthFunc, nvn::Entry::nvnDepthStencilStateSetDepthFunc},
    {Entry::nvnDepthStencilStateSetDepthTestEnable, nvn::Entry::nvnDepthStencilStateSetDepthTestEnable},
    {Entry::nvnDepthStencilStateSetDepthWriteEnable, nvn::Entry::nvnDepthStencilStateSetDepthWriteEnable},
    {Entry::nvnDepthStencilStateSetStencilFunc, nvn::Entry::nvnDepthStencilStateSetStencilFunc},
    {Entry::nvnDepthStencilStateSetStencilOp, nvn::Entry::nvnDepthStencilStateSetStencilOp},
    {Entry::nvnDepthStencilStateSetStencilTestEnable, nvn::Entry::nvnDepthStencilStateSetStencilTestEnable},
    {Entry::nvnVertexAttribStateGetFormat, nvn::Entry::nvnVertexAttribStateGetFormat},
    {Entry::nvnVertexAttribStateGetStreamIndex, nvn::Entry::nvnVertexAttribStateGetStreamIndex},
    {Entry::nvnVertexAttribStateSetDefaults, nvn::Entry::nvnVertexAttribStateSetDefaults},
    {Entry::nvnVertexAttribStateSetFormat, nvn::Entry::nvnVertexAttribStateSetFormat},
    {Entry::nvnVertexAttribStateSetStreamIndex, nvn::Entry::nvnVertexAttribStateSetStreamIndex},
    {Entry::nvnVertexStreamStateGetDivisor, nvn::Entry::nvnVertexStreamStateGetDivisor},
    {Entry::nvnVertexStreamStateGetStride, nvn::Entry::nvnVertexStreamStateGetStride},
    {Entry::nvnVertexStreamStateSetDefaults, nvn::Entry::nvnVertexStreamStateSetDefaults},
    {Entry::nvnVertexStreamStateSetDivisor, nvn::Entry::nvnVertexStreamStateSetDivisor},
    {Entry::nvnVertexStreamStateSetStride, nvn::Entry::nvnVertexStreamStateSetStride},
    // Command buffers: initialization clears 0xb0 bytes.
    {Entry::nvnCommandBufferAddCommandMemory, nvn::Entry::nvnCommandBufferAddCommandMemory},
    {Entry::nvnCommandBufferAddControlMemory, nvn::Entry::nvnCommandBufferAddControlMemory},
    {Entry::nvnCommandBufferBarrier, nvn::Entry::nvnCommandBufferBarrier},
    {Entry::nvnCommandBufferBeginRecording, nvn::Entry::nvnCommandBufferBeginRecording},
    {Entry::nvnCommandBufferBeginTransformFeedback, nvn::Entry::nvnCommandBufferBeginTransformFeedback},
    {Entry::nvnCommandBufferBindBlendState, nvn::Entry::nvnCommandBufferBindBlendState},
    {Entry::nvnCommandBufferBindChannelMaskState, nvn::Entry::nvnCommandBufferBindChannelMaskState},
    {Entry::nvnCommandBufferBindColorState, nvn::Entry::nvnCommandBufferBindColorState},
    {Entry::nvnCommandBufferBindCoverageModulationTable, nvn::Entry::nvnCommandBufferBindCoverageModulationTable},
    {Entry::nvnCommandBufferBindDepthStencilState, nvn::Entry::nvnCommandBufferBindDepthStencilState},
    {Entry::nvnCommandBufferBindMultisampleState, nvn::Entry::nvnCommandBufferBindMultisampleState},
    {Entry::nvnCommandBufferBindPolygonState, nvn::Entry::nvnCommandBufferBindPolygonState},
    {Entry::nvnCommandBufferBindTransformFeedbackBuffer, nvn::Entry::nvnCommandBufferBindTransformFeedbackBuffer},
    {Entry::nvnCommandBufferBindTransformFeedbackBuffers, nvn::Entry::nvnCommandBufferBindTransformFeedbackBuffers},
    {Entry::nvnCommandBufferBindVertexAttribState, nvn::Entry::nvnCommandBufferBindVertexAttribState},
    {Entry::nvnCommandBufferBindVertexBuffer, nvn::Entry::nvnCommandBufferBindVertexBuffer},
    {Entry::nvnCommandBufferBindVertexBuffers, nvn::Entry::nvnCommandBufferBindVertexBuffers},
    {Entry::nvnCommandBufferBindVertexStreamState, nvn::Entry::nvnCommandBufferBindVertexStreamState},
    {Entry::nvnCommandBufferCallCommands, nvn::Entry::nvnCommandBufferCallCommands},
    {Entry::nvnCommandBufferClearBuffer, nvn::Entry::nvnCommandBufferClearBuffer},
    {Entry::nvnCommandBufferClearColor, nvn::Entry::nvnCommandBufferClearColor},
    {Entry::nvnCommandBufferClearColori, nvn::Entry::nvnCommandBufferClearColori},
    {Entry::nvnCommandBufferClearColorui, nvn::Entry::nvnCommandBufferClearColorui},
    {Entry::nvnCommandBufferClearDepthStencil, nvn::Entry::nvnCommandBufferClearDepthStencil},
    {Entry::nvnCommandBufferClearTexture, nvn::Entry::nvnCommandBufferClearTexture},
    {Entry::nvnCommandBufferClearTexturei, nvn::Entry::nvnCommandBufferClearTexturei},
    {Entry::nvnCommandBufferClearTextureui, nvn::Entry::nvnCommandBufferClearTextureui},
    {Entry::nvnCommandBufferCopyBufferToBuffer, nvn::Entry::nvnCommandBufferCopyBufferToBuffer},
    {Entry::nvnCommandBufferCopyBufferToTexture, nvn::Entry::nvnCommandBufferCopyBufferToTexture},
    {Entry::nvnCommandBufferCopyCommands, nvn::Entry::nvnCommandBufferCopyCommands},
    {Entry::nvnCommandBufferCopyTextureToBuffer, nvn::Entry::nvnCommandBufferCopyTextureToBuffer},
    {Entry::nvnCommandBufferCopyTextureToTexture, nvn::Entry::nvnCommandBufferCopyTextureToTexture},
    {Entry::nvnCommandBufferDiscardColor, nvn::Entry::nvnCommandBufferDiscardColor},
    {Entry::nvnCommandBufferDiscardDepthStencil, nvn::Entry::nvnCommandBufferDiscardDepthStencil},
    {Entry::nvnCommandBufferDispatchCompute, nvn::Entry::nvnCommandBufferDispatchCompute},
    {Entry::nvnCommandBufferDispatchComputeIndirect, nvn::Entry::nvnCommandBufferDispatchComputeIndirect},
    {Entry::nvnCommandBufferDownsample, nvn::Entry::nvnCommandBufferDownsample},
    {Entry::nvnCommandBufferDownsampleTextureView, nvn::Entry::nvnCommandBufferDownsampleTextureView},
    {Entry::nvnCommandBufferDrawArrays, nvn::Entry::nvnCommandBufferDrawArrays},
    {Entry::nvnCommandBufferDrawArraysIndirect, nvn::Entry::nvnCommandBufferDrawArraysIndirect},
    {Entry::nvnCommandBufferDrawArraysInstanced, nvn::Entry::nvnCommandBufferDrawArraysInstanced},
    {Entry::nvnCommandBufferDrawElements, nvn::Entry::nvnCommandBufferDrawElements},
    {Entry::nvnCommandBufferDrawElementsBaseVertex, nvn::Entry::nvnCommandBufferDrawElementsBaseVertex},
    {Entry::nvnCommandBufferDrawElementsIndirect, nvn::Entry::nvnCommandBufferDrawElementsIndirect},
    {Entry::nvnCommandBufferDrawElementsInstanced, nvn::Entry::nvnCommandBufferDrawElementsInstanced},
    {Entry::nvnCommandBufferDrawTexture, nvn::Entry::nvnCommandBufferDrawTexture},
    {Entry::nvnCommandBufferDrawTransformFeedback, nvn::Entry::nvnCommandBufferDrawTransformFeedback},
    {Entry::nvnCommandBufferEndRecording, nvn::Entry::nvnCommandBufferEndRecording},
    {Entry::nvnCommandBufferEndTransformFeedback, nvn::Entry::nvnCommandBufferEndTransformFeedback},
    {Entry::nvnCommandBufferFenceSync, nvn::Entry::nvnCommandBufferFenceSync},
    {Entry::nvnCommandBufferFinalize, nvn::Entry::nvnCommandBufferFinalize},
    {Entry::nvnCommandBufferGetCommandMemoryFree, nvn::Entry::nvnCommandBufferGetCommandMemoryFree},
    {Entry::nvnCommandBufferGetCommandMemorySize, nvn::Entry::nvnCommandBufferGetCommandMemorySize},
    {Entry::nvnCommandBufferGetCommandMemoryUsed, nvn::Entry::nvnCommandBufferGetCommandMemoryUsed},
    {Entry::nvnCommandBufferGetControlMemoryFree, nvn::Entry::nvnCommandBufferGetControlMemoryFree},
    {Entry::nvnCommandBufferGetControlMemorySize, nvn::Entry::nvnCommandBufferGetControlMemorySize},
    {Entry::nvnCommandBufferGetControlMemoryUsed, nvn::Entry::nvnCommandBufferGetControlMemoryUsed},
    {Entry::nvnCommandBufferGetCopyImageStride, nvn::Entry::nvnCommandBufferGetCopyImageStride},
    {Entry::nvnCommandBufferGetCopyRowStride, nvn::Entry::nvnCommandBufferGetCopyRowStride},
    {Entry::nvnCommandBufferGetMemoryCallback, nvn::Entry::nvnCommandBufferGetMemoryCallback},
    {Entry::nvnCommandBufferGetMemoryCallbackData, nvn::Entry::nvnCommandBufferGetMemoryCallbackData},
    {Entry::nvnCommandBufferInitialize, nvn::Entry::nvnCommandBufferInitialize},
    {Entry::nvnCommandBufferInsertDebugMarker, nvn::Entry::nvnCommandBufferInsertDebugMarker},
    {Entry::nvnCommandBufferIsRecording, nvn::Entry::nvnCommandBufferIsRecording},
    {Entry::nvnCommandBufferMultiDrawArraysIndirectCount, nvn::Entry::nvnCommandBufferMultiDrawArraysIndirectCount},
    {Entry::nvnCommandBufferMultiDrawElementsIndirectCount, nvn::Entry::nvnCommandBufferMultiDrawElementsIndirectCount},
    {Entry::nvnCommandBufferPauseTransformFeedback, nvn::Entry::nvnCommandBufferPauseTransformFeedback},
    {Entry::nvnCommandBufferPopDebugGroup, nvn::Entry::nvnCommandBufferPopDebugGroup},
    {Entry::nvnCommandBufferPushDebugGroup, nvn::Entry::nvnCommandBufferPushDebugGroup},
    {Entry::nvnCommandBufferReportCounter, nvn::Entry::nvnCommandBufferReportCounter},
    {Entry::nvnCommandBufferReportValue, nvn::Entry::nvnCommandBufferReportValue},
    {Entry::nvnCommandBufferResetCounter, nvn::Entry::nvnCommandBufferResetCounter},
    {Entry::nvnCommandBufferResolveDepthBuffer, nvn::Entry::nvnCommandBufferResolveDepthBuffer},
    {Entry::nvnCommandBufferRestoreZCullData, nvn::Entry::nvnCommandBufferRestoreZCullData},
    {Entry::nvnCommandBufferResumeTransformFeedback, nvn::Entry::nvnCommandBufferResumeTransformFeedback},
    {Entry::nvnCommandBufferSaveZCullData, nvn::Entry::nvnCommandBufferSaveZCullData},
    {Entry::nvnCommandBufferSetAlphaRef, nvn::Entry::nvnCommandBufferSetAlphaRef},
    {Entry::nvnCommandBufferSetBlendColor, nvn::Entry::nvnCommandBufferSetBlendColor},
    {Entry::nvnCommandBufferSetConservativeRasterDilate, nvn::Entry::nvnCommandBufferSetConservativeRasterDilate},
    {Entry::nvnCommandBufferSetConservativeRasterEnable, nvn::Entry::nvnCommandBufferSetConservativeRasterEnable},
    {Entry::nvnCommandBufferSetCopyImageStride, nvn::Entry::nvnCommandBufferSetCopyImageStride},
    {Entry::nvnCommandBufferSetCopyRowStride, nvn::Entry::nvnCommandBufferSetCopyRowStride},
    {Entry::nvnCommandBufferSetDebugLabel, nvn::Entry::nvnCommandBufferSetDebugLabel},
    {Entry::nvnCommandBufferSetDepthBounds, nvn::Entry::nvnCommandBufferSetDepthBounds},
    {Entry::nvnCommandBufferSetDepthClamp, nvn::Entry::nvnCommandBufferSetDepthClamp},
    {Entry::nvnCommandBufferSetDepthRange, nvn::Entry::nvnCommandBufferSetDepthRange},
    {Entry::nvnCommandBufferSetDepthRanges, nvn::Entry::nvnCommandBufferSetDepthRanges},
    {Entry::nvnCommandBufferSetInnerTessellationLevels, nvn::Entry::nvnCommandBufferSetInnerTessellationLevels},
    {Entry::nvnCommandBufferSetLineWidth, nvn::Entry::nvnCommandBufferSetLineWidth},
    {Entry::nvnCommandBufferSetMemoryCallback, nvn::Entry::nvnCommandBufferSetMemoryCallback},
    {Entry::nvnCommandBufferSetMemoryCallbackData, nvn::Entry::nvnCommandBufferSetMemoryCallbackData},
    {Entry::nvnCommandBufferSetOuterTessellationLevels, nvn::Entry::nvnCommandBufferSetOuterTessellationLevels},
    {Entry::nvnCommandBufferSetPatchSize, nvn::Entry::nvnCommandBufferSetPatchSize},
    {Entry::nvnCommandBufferSetPointSize, nvn::Entry::nvnCommandBufferSetPointSize},
    {Entry::nvnCommandBufferSetPolygonOffsetClamp, nvn::Entry::nvnCommandBufferSetPolygonOffsetClamp},
    {Entry::nvnCommandBufferSetPrimitiveRestart, nvn::Entry::nvnCommandBufferSetPrimitiveRestart},
    {Entry::nvnCommandBufferSetProgramSubroutines, nvn::Entry::nvnCommandBufferSetProgramSubroutines},
    {Entry::nvnCommandBufferSetRasterizerDiscard, nvn::Entry::nvnCommandBufferSetRasterizerDiscard},
    {Entry::nvnCommandBufferSetRenderEnable, nvn::Entry::nvnCommandBufferSetRenderEnable},
    {Entry::nvnCommandBufferSetRenderEnableConditional, nvn::Entry::nvnCommandBufferSetRenderEnableConditional},
    {Entry::nvnCommandBufferSetRenderTargets, nvn::Entry::nvnCommandBufferSetRenderTargets},
    {Entry::nvnCommandBufferSetSampleMask, nvn::Entry::nvnCommandBufferSetSampleMask},
    {Entry::nvnCommandBufferSetSamplerPool, nvn::Entry::nvnCommandBufferSetSamplerPool},
    {Entry::nvnCommandBufferSetScissor, nvn::Entry::nvnCommandBufferSetScissor},
    {Entry::nvnCommandBufferSetScissors, nvn::Entry::nvnCommandBufferSetScissors},
    {Entry::nvnCommandBufferSetShaderScratchMemory, nvn::Entry::nvnCommandBufferSetShaderScratchMemory},
    {Entry::nvnCommandBufferSetStencilMask, nvn::Entry::nvnCommandBufferSetStencilMask},
    {Entry::nvnCommandBufferSetStencilRef, nvn::Entry::nvnCommandBufferSetStencilRef},
    {Entry::nvnCommandBufferSetStencilValueMask, nvn::Entry::nvnCommandBufferSetStencilValueMask},
    {Entry::nvnCommandBufferSetSubpixelPrecisionBias, nvn::Entry::nvnCommandBufferSetSubpixelPrecisionBias},
    {Entry::nvnCommandBufferSetTexturePool, nvn::Entry::nvnCommandBufferSetTexturePool},
    {Entry::nvnCommandBufferSetTiledCacheAction, nvn::Entry::nvnCommandBufferSetTiledCacheAction},
    {Entry::nvnCommandBufferSetTiledCacheTileSize, nvn::Entry::nvnCommandBufferSetTiledCacheTileSize},
    {Entry::nvnCommandBufferSetViewport, nvn::Entry::nvnCommandBufferSetViewport},
    {Entry::nvnCommandBufferSetViewportSwizzles, nvn::Entry::nvnCommandBufferSetViewportSwizzles},
    {Entry::nvnCommandBufferSetViewports, nvn::Entry::nvnCommandBufferSetViewports},
    {Entry::nvnCommandBufferSignalEvent, nvn::Entry::nvnCommandBufferSignalEvent},
    {Entry::nvnCommandBufferTiledDownsample, nvn::Entry::nvnCommandBufferTiledDownsample},
    {Entry::nvnCommandBufferTiledDownsampleTextureView, nvn::Entry::nvnCommandBufferTiledDownsampleTextureView},
    {Entry::nvnCommandBufferUpdateUniformBuffer, nvn::Entry::nvnCommandBufferUpdateUniformBuffer},
    {Entry::nvnCommandBufferWaitEvent, nvn::Entry::nvnCommandBufferWaitEvent},
    {Entry::nvnCommandBufferWaitSync, nvn::Entry::nvnCommandBufferWaitSync},
};

} // namespace

void register_entries() noexcept {
    for (const auto& [entry, implementation] : shared_entries) share(entry, implementation);
    install(Entry::nvnDeviceGetProcAddress, &device_get_proc_address);
    install(Entry::nvnDeviceGetInteger, &device_get_integer);
    install(Entry::nvnDeviceGetTextureHandle, &device_get_texture_handle);
    install(Entry::nvnDeviceGetTexelFetchHandle, &device_get_texel_fetch_handle);
    install(Entry::nvnDeviceGetImageHandle, &device_get_image_handle);
    install(Entry::nvnProgramBuilderSetDefaults, &program_builder_set_defaults);
    install(Entry::nvnProgramBuilderSetDevice, &program_builder_set_device);
    install(Entry::nvnProgramBuilderSetShaders, &program_builder_set_shaders);
    install(Entry::nvnProgramInitializeFromBuilder, &program_initialize_from_builder);
    install(Entry::nvnMemoryPoolMapVirtual, &memory_pool_map_virtual);
    install(Entry::nvnMemoryPoolInitialize, &memory_pool_initialize);
    install(Entry::nvnMemoryPoolFinalize, &memory_pool_finalize);
    install(Entry::nvnQueueBuilderSetPrimaryQueue, &queue_builder_set_primary_queue);
    install(Entry::nvnCommandBufferSetRecordingMode, &command_buffer_set_recording_mode);
    install(Entry::nvnCommandBufferSetCommandMemoryCallbackEnabled,
            &command_buffer_set_command_memory_callback_enabled);
    install(Entry::nvnCommandBufferSetControlMemoryCallbackEnabled,
            &command_buffer_set_control_memory_callback_enabled);
    install(Entry::nvnWindowBuilderSetDefaults, &window_builder_set_defaults);
    install(Entry::nvnWindowBuilderSetNumActiveTextures, &window_builder_set_num_active_textures);
    install(Entry::nvnWindowInitialize, &window_initialize);
    install(Entry::nvnWindowSetNumActiveTextures, &window_set_num_active_textures);
#define ARMIGO_NVN231_STAGE_BIND(name) install(Entry::name, &StageBind<nvn::Entry::name>::call);
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindImage)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindImages)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindSeparateSampler)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindSeparateSamplers)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindSeparateTexture)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindSeparateTextures)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindStorageBuffer)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindStorageBuffers)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindTexture)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindTextures)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindTexturesDeferred)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindUniformBuffer)
    ARMIGO_NVN231_STAGE_BIND(nvnCommandBufferBindUniformBuffers)
#undef ARMIGO_NVN231_STAGE_BIND
    install(Entry::nvnCommandBufferBindProgram, &command_buffer_bind_program);
    install(Entry::nvnCommandBufferPushGraphicsConstants, &command_buffer_push_graphics_constants);
    install(Entry::nvnCommandBufferPushComputeConstants, &command_buffer_push_compute_constants);
    install(Entry::nvnCommandBufferDrawMeshTasks, &command_buffer_draw_mesh_tasks);
    install(Entry::nvnCommandBufferDrawMeshTasksIndirect, &command_buffer_draw_mesh_tasks_indirect);
    install(Entry::nvnCommandBufferSetColorReductionEnable, &command_buffer_set_color_reduction_enable);
    install(Entry::nvnCommandBufferSetColorReductionThresholds, &command_buffer_set_color_reduction_thresholds);
    install(Entry::nvnCommandBufferSetComputeOverlapMode, &command_buffer_set_compute_overlap_mode);
    install(Entry::nvnCommandBufferSetComputeWorkloadPriority, &command_buffer_set_compute_workload_priority);
    install(Entry::nvnCommandBufferSetGeometryProcessingMode, &command_buffer_set_geometry_processing_mode);
    install(Entry::nvnCommandBufferSetStencilCullCriteria, &command_buffer_set_stencil_cull_criteria);
    install(Entry::nvnCommandBufferSetConservativeRasterMode, &command_buffer_set_conservative_raster_mode);
    install(Entry::nvnCommandBufferBindShadingRateImage, &command_buffer_bind_shading_rate_image);
    install(Entry::nvnCommandBufferSetShadingRatePaletteMode, &command_buffer_set_shading_rate_palette_mode);
    install(Entry::nvnCommandBufferShadingRateImagePalette, &command_buffer_shading_rate_image_palette);
}

} // namespace armigo::nvn::v231
