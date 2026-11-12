// ============================================================================
// Armigo Tests - NVN API 231 Entry Points
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The API 231 table as a title resolves it through the redirected bootstrap
// loader: admission, explicit stubs, shared API 55 implementations, the API 231
// device queries and handles, program builders, whole-pool virtual mappings,
// command buffer memory callbacks and window active textures.

#include "backend.h"
#include "commands.h"
#include "entry_table_231.h"
#include "objects.h"
#include "present.h"
#include "programs.h"
#include "redirect.h"
#include <armigo/services/video_interface.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace armigo::nvn {
void register_memory_entries() noexcept;
void register_texture_entries();
} // namespace armigo::nvn

namespace {

using namespace armigo::nvn;
using namespace std::chrono_literals;

unsigned failures = 0;
#define CHECK(condition)                                                  \
    do {                                                                  \
        if (!(condition)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); \
            ++failures;                                                   \
        }                                                                 \
    } while (false)

template <typename Signature>
Signature entry(const char* name) {
    return reinterpret_cast<Signature>(armigo_nvn231_bootstrap_loader(name));
}

int query(const NVNdevice* device, int info) {
    int value = -1;
    entry<void (*)(const NVNdevice*, int, int*)>("nvnDeviceGetInteger")(device, info, &value);
    return value;
}

void test_admission() {
    CHECK(known_api_version("22_2_8") == (ApiVersion{231, 40}));
    CHECK(bootstrap_loader_for({231, 40}) == &armigo_nvn231_bootstrap_loader);
    CHECK(bootstrap_loader_for({55, 15}) == &armigo_nvn_bootstrap_loader);
    // Nothing is handed out before API 231 is selected, nor by the other table.
    CHECK(armigo_nvn231_bootstrap_loader("nvnDeviceInitialize") == nullptr);
    select_api_version({55, 15});
    CHECK(armigo_nvn231_bootstrap_loader("nvnDeviceInitialize") == nullptr);
    CHECK(armigo_nvn_bootstrap_loader("nvnDeviceInitialize") != nullptr);
    select_api_version({231, 40});
    CHECK(selected_api_version() == (ApiVersion{231, 40}));
    CHECK(armigo_nvn_bootstrap_loader("nvnDeviceInitialize") == nullptr);
}

void test_names_and_stubs() {
    using v231::Entry;
    CHECK(v231::entry_count == 765);
    CHECK(v231::find_entry("nvnDeviceGetProcAddress") == Entry::nvnDeviceGetProcAddress);
    CHECK(v231::entry_name(Entry::nvnQueueBuilderSetPrimaryQueue) == "nvnQueueBuilderSetPrimaryQueue");
    // Names the driver does not resolve stay null, as they do natively.
    CHECK(armigo_nvn231_bootstrap_loader("nvnNotAnEntryPoint") == nullptr);
    CHECK(armigo_nvn231_bootstrap_loader("nvnBootstrapLoader") == nullptr);
    CHECK(armigo_nvn231_bootstrap_loader("nvnQueueAcquireTexture") == nullptr);
    // An unimplemented entry point reports itself; without a runtime handler
    // it returns zero.
    const auto mesh = entry<std::uint64_t (*)()>("nvnCommandBufferDrawMeshTasks3D");
    CHECK(!v231::implemented(Entry::nvnCommandBufferDrawMeshTasks3D));
    CHECK(mesh != nullptr && mesh() == 0);
    // The shading-rate commands are accepted and never applied, which is the
    // disabled pixel-rate state native stages translate with.
    static_assert(!v231::applies_shading_rate);
    CHECK(v231::implemented(Entry::nvnCommandBufferBindShadingRateImage));
    CHECK(v231::implemented(Entry::nvnCommandBufferSetShadingRatePaletteMode));
    CHECK(v231::implemented(Entry::nvnCommandBufferShadingRateImagePalette));
    CHECK(v231::implemented(Entry::nvnDeviceGetProcAddress));
    // nvnDeviceGetProcAddress resolves through the same table.
    const auto get_proc = entry<GenericFunction (*)(const NVNdevice*, const char*)>("nvnDeviceGetProcAddress");
    CHECK(get_proc(nullptr, "nvnDeviceInitialize") == armigo_nvn231_bootstrap_loader("nvnDeviceInitialize"));
}

std::uint64_t replaced_lod_bias_calls = 0;
float replaced_lod_bias(const NVNsampler*) noexcept {
    ++replaced_lod_bias_calls;
    return 2.5f;
}

void test_shared_entries_follow_late_installs() {
    // A shared entry point serves whatever API 55 implements when it resolves.
    install<armigo::nvn::Entry::nvnSamplerGetLodBias>(&replaced_lod_bias);
    const auto lod_bias = entry<float (*)(const NVNsampler*)>("nvnSamplerGetLodBias");
    CHECK(reinterpret_cast<void*>(lod_bias) == reinterpret_cast<void*>(&replaced_lod_bias));
    CHECK(lod_bias(nullptr) == 2.5f && replaced_lod_bias_calls == 1);
}

void test_device() {
    NVNdeviceBuilder builder{};
    NVNdevice device{};
    entry<void (*)(NVNdeviceBuilder*)>("nvnDeviceBuilderSetDefaults")(&builder);
    entry<void (*)(NVNdeviceBuilder*, int)>("nvnDeviceBuilderSetFlags")(&builder, 0);
    CHECK(entry<NVNboolean (*)(NVNdevice*, const NVNdeviceBuilder*)>("nvnDeviceInitialize")(&device, &builder) == 1);
    CHECK(query(&device, 0) == 231 && query(&device, 1) == 40);
    CHECK(reinterpret_cast<const armigo::nvn::DeviceState*>(&device)->gob_order == 1);
    // The API 231 shader code range and padding; other properties 2-97 keep
    // their API 55 numbers and values.
    CHECK(query(&device, 46) == 201 && query(&device, 47) == 201);
    CHECK(query(&device, 48) == 20 && query(&device, 49) == 0);
    CHECK(query(&device, 78) == 0x800);
    CHECK(query(&device, NVN_DEVICE_INFO_UNIFORM_BUFFER_ALIGNMENT) == 256);
    CHECK(query(&device, NVN_DEVICE_INFO_TEXTURE_DESCRIPTOR_SIZE) == 32);
    // Properties 99-109 answer the title driver's constants.
    CHECK(query(&device, 99) == 256 && query(&device, 100) == 16 && query(&device, 101) == 65535);
    CHECK(query(&device, 102) == 65535 && query(&device, 103) == 256 && query(&device, 104) == 1);
    CHECK(query(&device, 105) == 16 && query(&device, 107) == 16 && query(&device, 109) == 1);
    // A property this driver does not answer is reported; zero without a handler.
    CHECK(query(&device, 98) == 0 && query(&device, 110) == 0 && query(&device, 169) == 0);

    // Handles: texture ID in bits 0-19, sampler ID from bit 20, sign-extended;
    // texture zero has no handle; no bit 32.
    using Handle = std::uint64_t (*)(const NVNdevice*, int, int);
    const auto texture_handle = entry<Handle>("nvnDeviceGetTextureHandle");
    CHECK(texture_handle(&device, 5, 3) == (5u | (3u << 20)));
    CHECK(texture_handle(&device, 0, 7) == 0);
    CHECK(texture_handle(&device, 0xfffff, 0xfff) == 0xffffffffull);
    CHECK(texture_handle(&device, 1, -1) == 0xfffffffffff00001ull);
    CHECK(handle_texture_id(texture_handle(&device, 0x12345, 0x678)) == 0x12345);
    CHECK(handle_sampler_id(texture_handle(&device, 0x12345, 0x678)) == 0x678);
    CHECK(entry<std::uint64_t (*)(const NVNdevice*, int)>("nvnDeviceGetTexelFetchHandle")(&device, 9) == 9);
    CHECK(entry<std::uint64_t (*)(const NVNdevice*, int)>("nvnDeviceGetImageHandle")(&device, -2) ==
          0xfffffffffffffffeull);
}

struct ProgramCalls {
    NVNprogram* program = nullptr;
    NVNdevice* device = nullptr;
    int count = -1;
    const NVNshaderData* shaders = nullptr;
} program_calls;

NVNboolean record_program_initialize(NVNprogram* program, NVNdevice* device) noexcept {
    program_calls.program = program;
    program_calls.device = device;
    return 1;
}

NVNboolean record_program_set_shaders(NVNprogram* program, int count, const NVNshaderData* shaders) noexcept {
    program_calls.count = program == program_calls.program ? count : -2;
    program_calls.shaders = shaders;
    return 1;
}

void test_program_builder() {
    install<armigo::nvn::Entry::nvnProgramInitialize>(&record_program_initialize);
    install<armigo::nvn::Entry::nvnProgramSetShaders>(&record_program_set_shaders);
    alignas(8) std::byte builder[0x80]{};
    NVNprogram program{};
    NVNdevice device{};
    const NVNshaderData shaders[2]{{0x1000, nullptr}, {0x2000, nullptr}};
    entry<void (*)(void*)>("nvnProgramBuilderSetDefaults")(builder);
    entry<void (*)(void*, NVNdevice*)>("nvnProgramBuilderSetDevice")(builder, &device);
    entry<void (*)(void*, int, const NVNshaderData*)>("nvnProgramBuilderSetShaders")(builder, 2, shaders);
    CHECK(entry<NVNboolean (*)(NVNprogram*, const void*)>("nvnProgramInitializeFromBuilder")(&program, builder) == 1);
    CHECK(program_calls.program == &program && program_calls.device == &device);
    CHECK(program_calls.count == 2 && program_calls.shaders == shaders);
}

struct Pool {
    NVNmemoryPool pool{};
    NVNmemoryPoolBuilder builder{};
    Pool(void* storage, std::size_t size, int flags) {
        entry<void (*)(NVNmemoryPoolBuilder*)>("nvnMemoryPoolBuilderSetDefaults")(&builder);
        entry<void (*)(NVNmemoryPoolBuilder*, NVNdevice*)>("nvnMemoryPoolBuilderSetDevice")(&builder, nullptr);
        entry<void (*)(NVNmemoryPoolBuilder*, void*, std::size_t)>("nvnMemoryPoolBuilderSetStorage")(&builder, storage,
                                                                                                    size);
        entry<void (*)(NVNmemoryPoolBuilder*, int)>("nvnMemoryPoolBuilderSetFlags")(&builder, flags);
        CHECK(entry<NVNboolean (*)(NVNmemoryPool*, const NVNmemoryPoolBuilder*)>("nvnMemoryPoolInitialize")(
                  &pool, &builder) == 1);
    }
    const MemoryPoolState& state() const { return *reinterpret_cast<const MemoryPoolState*>(&pool); }
};

void test_virtual_pool() {
    alignas(4096) static std::byte storage[0x4000];
    Pool physical(storage, sizeof storage, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_GPU_CACHED |
                                               NVN_MEMORY_POOL_FLAGS_PHYSICAL);
    const int virtual_flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_GPU_CACHED |
                              NVN_MEMORY_POOL_FLAGS_VIRTUAL | NVN_MEMORY_POOL_FLAGS_COMPRESSIBLE;
    Pool alias(nullptr, sizeof storage, virtual_flags);
    CHECK(alias.state().memory == nullptr);
    const auto map = entry<NVNboolean (*)(NVNmemoryPool*, int, const NVNmappingRequest*)>("nvnMemoryPoolMapVirtual");
    // Anything but one mapping of the whole virtual pool is refused.
    NVNmappingRequest partial{&physical.pool, 0, 0, 0x1000, {}, 0};
    CHECK(map(&alias.pool, 1, &partial) == 0 && alias.state().memory == nullptr);
    NVNmappingRequest outside{&physical.pool, 0x1000, 0, sizeof storage, {}, 0};
    CHECK(map(&alias.pool, 1, &outside) == 0 && alias.state().memory == nullptr);
    NVNmappingRequest whole{&physical.pool, 0, 0, sizeof storage, {}, 0};
    CHECK(map(&alias.pool, 1, &whole) == 1);
    CHECK(alias.state().memory == reinterpret_cast<std::uint8_t*>(storage));
    // Mapping again is refused: the alias exists.
    CHECK(map(&alias.pool, 1, &whole) == 0);
    entry<void (*)(NVNmemoryPool*)>("nvnMemoryPoolFinalize")(&alias.pool);
    CHECK(alias.state().memory == nullptr);
    CHECK(physical.state().memory == reinterpret_cast<std::uint8_t*>(storage));
    entry<void (*)(NVNmemoryPool*)>("nvnMemoryPoolFinalize")(&physical.pool);
}

void test_command_buffers() {
    NVNdevice device{};
    alignas(8) std::byte storage[0xb0]{};
    auto* command_buffer = reinterpret_cast<NVNcommandBuffer*>(storage);
    CHECK(entry<NVNboolean (*)(NVNcommandBuffer*, NVNdevice*)>("nvnCommandBufferInitialize")(command_buffer, &device));
    const auto& state = *reinterpret_cast<const CommandBufferState*>(command_buffer);
    CHECK(!state.command_callback_disabled && !state.control_callback_disabled);
    const auto command_enabled =
        entry<void (*)(NVNcommandBuffer*, std::uint32_t)>("nvnCommandBufferSetCommandMemoryCallbackEnabled");
    const auto control_enabled =
        entry<void (*)(NVNcommandBuffer*, std::uint32_t)>("nvnCommandBufferSetControlMemoryCallbackEnabled");
    // Only the low byte of a guest boolean is defined.
    command_enabled(command_buffer, 0x100);
    CHECK(state.command_callback_disabled && !state.control_callback_disabled);
    control_enabled(command_buffer, 0);
    CHECK(state.control_callback_disabled);
    command_enabled(command_buffer, 1);
    CHECK(!state.command_callback_disabled && state.control_callback_disabled);
    const auto recording_mode = entry<void (*)(NVNcommandBuffer*, int)>("nvnCommandBufferSetRecordingMode");
    recording_mode(command_buffer, 2);
    recording_mode(command_buffer, 5);  // reported; no handler, so the call returns
    // Front-end controls the title's driver accepts with any value.
    for (const char* name : {"nvnCommandBufferSetComputeOverlapMode", "nvnCommandBufferSetComputeWorkloadPriority",
                             "nvnCommandBufferSetGeometryProcessingMode", "nvnCommandBufferSetShadingRatePaletteMode",
                             "nvnCommandBufferSetColorReductionEnable"}) {
        CHECK(v231::implemented(*v231::find_entry(name)));
        entry<void (*)(NVNcommandBuffer*, int)>(name)(command_buffer, 3);
    }
}

// Buffer and handle bindings at NVN2's task and mesh stages (6 and 7) reach
// the API 55 implementation at the tessellation slots (3 and 4); other stages
// pass through unchanged.
NVNshaderStage bound_stage = NVNshaderStage(-1);
int bound_index = -1;
void record_uniform_bind(NVNcommandBuffer*, NVNshaderStage stage, int index, NVNbufferAddress, size_t) noexcept {
    bound_stage = stage;
    bound_index = index;
}
void record_textures_deferred(NVNcommandBuffer*, NVNshaderStage stage, int first, int, const NVNtextureHandle*) noexcept {
    bound_stage = stage;
    bound_index = first;
}

const NVNprogram* bound_program = nullptr;
int bound_stages = -1;
void record_program_bind(NVNcommandBuffer*, const NVNprogram* program, int stages) noexcept {
    bound_program = program;
    bound_stages = stages;
}

void test_stage_banks() {
    const auto previous_uniform = installed_entry(armigo::nvn::Entry::nvnCommandBufferBindUniformBuffer);
    const auto previous_deferred = installed_entry(armigo::nvn::Entry::nvnCommandBufferBindTexturesDeferred);
    install<armigo::nvn::Entry::nvnCommandBufferBindUniformBuffer>(&record_uniform_bind);
    install<armigo::nvn::Entry::nvnCommandBufferBindTexturesDeferred>(&record_textures_deferred);
    const auto bind = entry<void (*)(NVNcommandBuffer*, NVNshaderStage, int, NVNbufferAddress, size_t)>(
        "nvnCommandBufferBindUniformBuffer");
    const auto deferred = entry<void (*)(NVNcommandBuffer*, NVNshaderStage, int, int, const NVNtextureHandle*)>(
        "nvnCommandBufferBindTexturesDeferred");
    bind(nullptr, NVNshaderStage(6), 2, 0x1000, 0x100);
    CHECK(bound_stage == NVNshaderStage(6) && bound_index == 2);
    bind(nullptr, NVNshaderStage(7), 5, 0x1000, 0x100);
    CHECK(bound_stage == NVNshaderStage(7) && bound_index == 5);
    bind(nullptr, NVN_SHADER_STAGE_FRAGMENT, 1, 0x1000, 0x100);
    CHECK(bound_stage == NVN_SHADER_STAGE_FRAGMENT && bound_index == 1);
    deferred(nullptr, NVNshaderStage(7), 4, 1, nullptr);
    CHECK(bound_stage == NVNshaderStage(7) && bound_index == 4);
    install_entry(armigo::nvn::Entry::nvnCommandBufferBindUniformBuffer, previous_uniform);
    install_entry(armigo::nvn::Entry::nvnCommandBufferBindTexturesDeferred, previous_deferred);
    // Program bindings retain the public task and mesh mask bits.
    const auto previous_program = installed_entry(armigo::nvn::Entry::nvnCommandBufferBindProgram);
    install<armigo::nvn::Entry::nvnCommandBufferBindProgram>(&record_program_bind);
    const auto bind_program = entry<void (*)(NVNcommandBuffer*, const NVNprogram*, int)>("nvnCommandBufferBindProgram");
    NVNprogram program{};
    bind_program(nullptr, &program, 0xc0 | 0x2);
    CHECK(bound_program == &program && bound_stages == (0xc0 | 0x2));
    bind_program(nullptr, &program, 0x21);
    CHECK(bound_stages == 0x21);
    install_entry(armigo::nvn::Entry::nvnCommandBufferBindProgram, previous_program);
}

// Graphics push constants as the title's driver emits them: whole words up to
// 256 bytes at byte offset (offset & 0xff), for hardware constant buffer 16;
// fewer than 4 bytes record nothing. Task and mesh retain separate banks,
// and compute is refused.
bool collect_mesh_draws(const CommandRecordView& view, void* context) {
    auto& seen = *static_cast<std::vector<std::pair<Op, std::uint64_t>>*>(context);
    if (view.op == Op::draw_mesh_tasks && view.payload_size >= sizeof(DrawMeshTasks)) {
        DrawMeshTasks draw{};
        std::memcpy(&draw, view.payload, sizeof draw);
        seen.emplace_back(view.op, std::uint64_t(std::uint32_t(draw.first)) << 32 | std::uint32_t(draw.count));
        CHECK(draw.api_return != 0);
    } else if (view.op == Op::draw_mesh_tasks_indirect && view.payload_size >= sizeof(DrawMeshTasksIndirect)) {
        DrawMeshTasksIndirect draw{};
        std::memcpy(&draw, view.payload, sizeof draw);
        seen.emplace_back(view.op, draw.address);
    }
    return true;
}

// Mesh draws are recorded with their arguments; the replay refuses them.
void test_mesh_draws() {
    NVNdevice device{};
    alignas(8) std::byte storage[0xb0]{};
    auto* command_buffer = reinterpret_cast<NVNcommandBuffer*>(storage);
    CHECK(entry<NVNboolean (*)(NVNcommandBuffer*, NVNdevice*)>("nvnCommandBufferInitialize")(command_buffer, &device));
    const auto begin = entry<void (*)(NVNcommandBuffer*)>("nvnCommandBufferBeginRecording");
    const auto end = entry<NVNcommandHandle (*)(NVNcommandBuffer*)>("nvnCommandBufferEndRecording");
    const auto draw = entry<void (*)(NVNcommandBuffer*, int, int)>("nvnCommandBufferDrawMeshTasks");
    const auto indirect = entry<void (*)(NVNcommandBuffer*, NVNbufferAddress)>("nvnCommandBufferDrawMeshTasksIndirect");
    begin(command_buffer);
    draw(command_buffer, 2, 5);
    indirect(command_buffer, 0x12340);
    const auto handle = end(command_buffer);
    std::vector<std::pair<Op, std::uint64_t>> seen;
    CHECK(handle && walk_commands(handle, collect_mesh_draws, &seen) == CommandWalkStatus::complete);
    CHECK(seen.size() == 2);
    if (seen.size() == 2) {
        CHECK(seen[0].first == Op::draw_mesh_tasks && seen[0].second == (std::uint64_t{2} << 32 | 5));
        CHECK(seen[1].first == Op::draw_mesh_tasks_indirect && seen[1].second == 0x12340);
    }
    begin(command_buffer);
    draw(command_buffer, -1, 5);
    CHECK(!end(command_buffer));
}

bool collect_pushes(const CommandRecordView& view, void* context) {
    if (view.op == Op::push_constants && view.payload_size >= sizeof(PushConstants)) {
        PushConstants push{};
        std::memcpy(&push, view.payload, sizeof push);
        static_cast<std::vector<PushConstants>*>(context)->push_back(push);
    }
    return true;
}

void test_push_constants() {
    NVNdevice device{};
    alignas(8) std::byte storage[0xb0]{};
    auto* command_buffer = reinterpret_cast<NVNcommandBuffer*>(storage);
    CHECK(entry<NVNboolean (*)(NVNcommandBuffer*, NVNdevice*)>("nvnCommandBufferInitialize")(command_buffer, &device));
    const auto begin = entry<void (*)(NVNcommandBuffer*)>("nvnCommandBufferBeginRecording");
    const auto end = entry<NVNcommandHandle (*)(NVNcommandBuffer*)>("nvnCommandBufferEndRecording");
    const auto push = entry<void (*)(NVNcommandBuffer*, NVNshaderStage, std::uint32_t, std::uint32_t, const void*)>(
        "nvnCommandBufferPushGraphicsConstants");
    CHECK(v231::implemented(*v231::find_entry("nvnCommandBufferPushGraphicsConstants")));
    std::array<std::uint32_t, 72> words{};
    for (std::uint32_t i = 0; i < words.size(); ++i) words[i] = 0x1000u + i;
    begin(command_buffer);
    push(command_buffer, NVNshaderStage(7), 0x10, 12, words.data());
    push(command_buffer, NVN_SHADER_STAGE_FRAGMENT, 0, 0x110, words.data());
    push(command_buffer, NVN_SHADER_STAGE_VERTEX, 0, 3, words.data());
    push(command_buffer, NVN_SHADER_STAGE_VERTEX, 0x104, 10, words.data() + 1);
    const auto handle = end(command_buffer);
    std::vector<PushConstants> pushes;
    CHECK(handle && walk_commands(handle, collect_pushes, &pushes) == CommandWalkStatus::complete);
    CHECK(pushes.size() == 3);
    if (pushes.size() == 3) {
        CHECK(pushes[0].stage == 7 && pushes[0].bank == 16 && pushes[0].offset == 0x10 && pushes[0].size == 12 &&
              std::memcmp(pushes[0].data, words.data(), 12) == 0);
        CHECK(pushes[1].stage == NVN_SHADER_STAGE_FRAGMENT && pushes[1].offset == 0 &&
              pushes[1].size == push_constant_bytes &&
              std::memcmp(pushes[1].data, words.data(), push_constant_bytes) == 0);
        CHECK(pushes[2].stage == NVN_SHADER_STAGE_VERTEX && pushes[2].offset == 4 && pushes[2].size == 8 &&
              std::memcmp(pushes[2].data, words.data() + 1, 8) == 0);
    }
    // A block overrun or the compute stage fails the recording.
    begin(command_buffer);
    push(command_buffer, NVN_SHADER_STAGE_VERTEX, 0xf0, 0x20, words.data());
    CHECK(!end(command_buffer));
    begin(command_buffer);
    push(command_buffer, NVN_SHADER_STAGE_COMPUTE, 0, 16, words.data());
    CHECK(!end(command_buffer));
    // Compute has its own block (bank 24); a nonzero block word is reported
    // and still lands there.
    const auto push_compute = entry<void (*)(NVNcommandBuffer*, std::uint32_t, std::uint32_t, std::uint32_t,
                                             const void*)>("nvnCommandBufferPushComputeConstants");
    CHECK(v231::implemented(*v231::find_entry("nvnCommandBufferPushComputeConstants")));
    begin(command_buffer);
    push_compute(command_buffer, 0, 8, 16, words.data());
    push_compute(command_buffer, 1, 0, 2, words.data());
    push_compute(command_buffer, 1, 0, 4, words.data() + 4);
    const auto compute = end(command_buffer);
    pushes.clear();
    CHECK(compute && walk_commands(compute, collect_pushes, &pushes) == CommandWalkStatus::complete);
    CHECK(pushes.size() == 2);
    if (pushes.size() == 2) {
        CHECK(pushes[0].stage == NVN_SHADER_STAGE_COMPUTE && pushes[0].bank == 24 && pushes[0].offset == 8 &&
              pushes[0].size == 16 && std::memcmp(pushes[0].data, words.data(), 16) == 0);
        CHECK(pushes[1].offset == 0 && pushes[1].size == 4 && std::memcmp(pushes[1].data, words.data() + 4, 4) == 0);
    }
    begin(command_buffer);
    push_compute(command_buffer, 0, 0xf8, 16, words.data());
    CHECK(!end(command_buffer));
}

class Replay final : public ReplayBackend {
public:
    std::uint64_t submit(std::span<const NVNcommandHandle>) override { return 0; }
    void flush() override {}
    std::uint64_t completed_fence() const override { return 0; }
    bool wait_fence(std::uint64_t, std::uint64_t) override { return true; }
};

void test_window_active_textures() {
    Replay replay;
    set_backends({nullptr, &replay});
    armigo::horizon::HandleTable handles(16);
    armigo::services::vi::DisplaySession display(handles);
    display.set_vsync_source(std::make_shared<armigo::services::vi::VsyncSource>(2ms));
    int native_token;
    CHECK(display.bind_native_window(&native_token));
    set_texture_exporter([](const TextureState& texture) -> std::optional<armigo::services::vi::NativeTexture> {
        armigo::services::vi::NativeTexture out;
        out.serial = texture.serial;
        out.width = 4;
        out.height = 4;
        out.read_pixels = [](std::array<std::int32_t, 4>) -> armigo::graphics::Rgba8ReadbackResult {
            return armigo::graphics::Rgba8Image{};
        };
        return out;
    });
    NVNdevice device{};
    alignas(8) NVNtexture textures[3]{};
    NVNtexture* pointers[3];
    for (int i = 0; i < 3; ++i) {
        auto& texture = *reinterpret_cast<TextureState*>(&textures[i]);
        texture.tag = object_tag('T', 'E', 'X', 'R');
        texture.serial = 31 + i;
        texture.description.width = 4;
        texture.description.height = 4;
        texture.memory = reinterpret_cast<std::uint8_t*>(0x40000 + i * 0x1000);
        texture.size = 64;
        pointers[i] = &textures[i];
    }
    alignas(8) std::byte builder_storage[0x70]{};
    auto* builder = reinterpret_cast<NVNwindowBuilder*>(builder_storage);
    alignas(8) std::byte window_storage[0x1c0]{};
    auto* window = reinterpret_cast<NVNwindow*>(window_storage);
    entry<void (*)(NVNwindowBuilder*, NVNdevice*)>("nvnWindowBuilderSetDevice")(builder, &device);
    entry<void (*)(NVNwindowBuilder*)>("nvnWindowBuilderSetDefaults")(builder);
    entry<void (*)(NVNwindowBuilder*, void*)>("nvnWindowBuilderSetNativeWindow")(builder, &native_token);
    entry<void (*)(NVNwindowBuilder*, int, NVNtexture* const*)>("nvnWindowBuilderSetTextures")(builder, 3, pointers);
    entry<void (*)(NVNwindowBuilder*, int)>("nvnWindowBuilderSetNumActiveTextures")(builder, 2);
    CHECK(entry<NVNboolean (*)(NVNwindow*, const NVNwindowBuilder*)>("nvnWindowInitialize")(window, builder) == 1);

    const auto acquire = entry<NVNwindowAcquireTextureResult (*)(NVNwindow*, NVNsync*, int*)>("nvnWindowAcquireTexture");
    int first = -1, second = -1;
    CHECK(acquire(window, nullptr, &first) == NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS);
    CHECK(acquire(window, nullptr, &second) == NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS);
    CHECK(first == 0 && second == 1);
    // The third texture is inactive: nothing is free to acquire.
    const auto producer = armigo::services::vi::find_native_window(&native_token);
    CHECK(producer && !producer->free_event()->is_signaled());
    const auto set_active = entry<void (*)(NVNwindow*, int)>("nvnWindowSetNumActiveTextures");
    set_active(window, 3);
    CHECK(producer && producer->free_event()->is_signaled());
    int third = -1;
    CHECK(acquire(window, nullptr, &third) == NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS && third == 2);
    // Out of range counts are refused and leave the window as it was.
    CHECK(producer && !producer->set_active_textures(4) && !producer->set_active_textures(1));
    set_active(window, 4);  // reported; no handler, so the call returns
    armigo::services::vi::unbind_native_window(&native_token);
    set_backends({});
}

// Counts backend pool registrations by range.
class CountingResources final : public ResourceBackend {
public:
    unsigned registered = 0, released = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> addressed;
    bool register_pool(const MemoryPoolState&) override { ++registered; return true; }
    void note_addressed(std::uint64_t address, std::uint64_t size) override {
        addressed.push_back({address, size});
    }
    void release_pool(const MemoryPoolState&) override { ++released; }
    void flush_pool(const MemoryPoolState&, std::uint64_t, std::uint64_t) override {}
    void invalidate_pool(const MemoryPoolState&, std::uint64_t, std::uint64_t) override {}
    void register_texture(const TextureState&) override {}
    void release_texture(const TextureState&) override {}
    void clear_texture_descriptor(std::int32_t) override { CHECK(false); }
    void register_texture_descriptor(std::int32_t, const TextureState&, const TextureViewState*) override {}
    void register_sampler_descriptor(std::int32_t, const SamplerParameters&) override {}
    bool read_texels(const TextureState&, const TexelTransfer&, std::byte*) override { return false; }
    bool write_texels(const TextureState&, const TexelTransfer&, const std::byte*) override { return false; }
};

// A virtual alias never releases the physical pool's registration: not when it
// is finalized and not when it is initialized again while mapped.
void test_virtual_alias_registration() {
    register_memory_entries();
    CountingResources resources;
    set_backends({&resources, nullptr});
    alignas(4096) static std::byte storage[0x4000];
    {
        Pool physical(storage, sizeof storage, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_GPU_CACHED |
                                                   NVN_MEMORY_POOL_FLAGS_PHYSICAL);
        const int virtual_flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_GPU_CACHED |
                                  NVN_MEMORY_POOL_FLAGS_VIRTUAL;
        Pool alias(nullptr, sizeof storage, virtual_flags);
        CHECK(resources.registered == 1);
        const auto map = entry<NVNboolean (*)(NVNmemoryPool*, int, const NVNmappingRequest*)>("nvnMemoryPoolMapVirtual");
        NVNmappingRequest partial{&physical.pool, 0, 0, 0x1000, {}, 0};
        CHECK(map(&alias.pool, 1, &partial) == 0 && resources.addressed.empty());
        NVNmappingRequest whole{&physical.pool, 0, 0, sizeof storage, {}, 0};
        CHECK(map(&alias.pool, 1, &whole) == 1);
        CHECK(resources.addressed.size() == 1);
        CHECK(resources.addressed[0].first == reinterpret_cast<std::uint64_t>(storage) &&
              resources.addressed[0].second == sizeof storage);
        // Initialized again while mapped: the alias gives up its address first.
        CHECK(entry<NVNboolean (*)(NVNmemoryPool*, const NVNmemoryPoolBuilder*)>("nvnMemoryPoolInitialize")(
                  &alias.pool, &alias.builder) == 1);
        CHECK(resources.released == 0 && alias.state().memory == nullptr);
        CHECK(map(&alias.pool, 1, &whole) == 1);
        CHECK(resources.addressed.size() == 2);
        entry<void (*)(NVNmemoryPool*)>("nvnMemoryPoolFinalize")(&alias.pool);
        CHECK(resources.released == 0);
        entry<void (*)(NVNmemoryPool*)>("nvnMemoryPoolFinalize")(&physical.pool);
        CHECK(resources.released == 1);
    }
    set_backends({});
}

bool resolve_any(void*, const DeviceState*, std::span<const NVNshaderData> shaders,
                 std::array<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>, 6>& stages,
                 std::string&) {
    for (std::size_t i = 0; i < shaders.size(); ++i)
        stages[i] = std::make_shared<armigo::platform::CompiledGraphicsStage>();
    return true;
}

// API 231 program constants (byte size at control +0x3c, offset from the
// shader data at +0x40) bind as hardware constant buffer 1 of the stage the
// selector byte at +0x56 names; a stage without constants binds none.
void test_program_constants_bound() {
    register_program_entries();
    set_program_resolver(&resolve_any, nullptr);
    NVNdevice device{};
    NVNprogram program{};
    CHECK(entry<NVNboolean (*)(NVNprogram*, NVNdevice*)>("nvnProgramInitialize")(&program, &device) == 1);
    alignas(8) std::byte vertex[0xe0]{}, fragment[0xe0]{};
    const std::uint32_t size = 0x40, offset = 0x380;
    std::memcpy(fragment + 0x3c, &size, 4);
    std::memcpy(fragment + 0x40, &offset, 4);
    fragment[0x56] = std::byte{1};
    const NVNshaderData shaders[2]{{0x10000, vertex}, {0x20000, fragment}};
    const auto set = entry<NVNboolean (*)(NVNprogram*, int, const NVNshaderData*)>("nvnProgramSetShaders");
    CHECK(set(&program, 2, shaders) == 1);
    const auto resolved = retain_program(*reinterpret_cast<const ProgramState*>(&program));
    CHECK(resolved != nullptr);
    if (resolved) {
        CHECK(resolved->constants[1].address == 0x20000 + 0x380 && resolved->constants[1].size == 0x40);
        CHECK(resolved->constants[0].address == 0 && resolved->constants[0].size == 0);
    }
    entry<void (*)(NVNprogram*)>("nvnProgramFinalize")(&program);
    set_program_resolver(nullptr, nullptr);
}

} // namespace

// Texture initialization and packaged texture data are API 55's on both
// paths: the API 231 entries resolve to the same implementations, so
// packaged data takes the copy the API 55 tests check.
void test_shared_texture_initialization() {
    register_texture_entries();
    for (const auto [name, implemented] :
         {std::pair{"nvnTextureInitialize", Entry::nvnTextureInitialize},
          std::pair{"nvnTextureBuilderSetPackagedTextureData", Entry::nvnTextureBuilderSetPackagedTextureData},
          std::pair{"nvnTextureBuilderGetPackagedTextureData", Entry::nvnTextureBuilderGetPackagedTextureData}}) {
        const auto resolved = armigo_nvn231_bootstrap_loader(name);
        CHECK(resolved && reinterpret_cast<void*>(resolved) == reinterpret_cast<void*>(implemented_entry(implemented)));
    }
}

int main() {
    test_admission();
    register_null_entries();
    register_present_entries();
    v231::register_entries();
    test_names_and_stubs();
    test_shared_entries_follow_late_installs();
    test_device();
    test_program_builder();
    test_virtual_pool();
    test_command_buffers();
    test_stage_banks();
    test_window_active_textures();
    test_virtual_alias_registration();
    test_program_constants_bound();
    // Recording proper: the API 55 command entries the shared ones resolve to.
    register_command_entries();
    test_push_constants();
    test_mesh_draws();
    test_shared_texture_initialization();
    std::printf("NVN API 231 failures: %u\n", failures);
    return failures ? 1 : 0;
}
