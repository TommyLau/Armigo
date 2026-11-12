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
#include <armigo/execution/guest_timeline.h>
#include <TargetConditionals.h>
#include "nvn_replay.h"
#include "../../nvn/program_capture.h"
#include "nvn_image_formats.h"
#include "nvn_pool_tables.h"
#include "nvn_draw_state.h"
#include "nvn_color_aliases.h"
#include "nvn_pass_dump.h"
#include "nvn_submission_segments.h"
#include "nvn_primitive_expansion.h"
#include "nvn_gpu_frame_work.h"
#include "nvn_guest_memory.h"
#include "nvn_encode_ring.h"
#include "nvn_attachment_actions.h"
#include "nvn_write_ranges.h"
#include "nvn/flat_map.h"
#include <armigo/memory/write_tracker.h>
#include <armigo/platform/presentation_thread.h>
#include <armigo/platform/periodic_thread_budget.h>
#include <objc/runtime.h>

#include "../../nvn/commands.h"
#include "../../nvn/conditional_reports.h"
#include "../../nvn/programs.h"
#ifdef ARMIGO_HAS_NATIVE_SHADERS
#include "nvn_native_stages.h"
#endif
#include "../../nvn/present.h"
#include <armigo/nvn/driver.h>
#include <armigo/nvn/timeline.h>
#include <CoreFoundation/CoreFoundation.h>
#include "clear_encoder.h"
#include "attachments.h"
#include "texture_format.h"
#include "pso_archive.h"
#include "pso_manifest.h"
#include <armigo/platform/metal_shader_package.h>
#ifdef ARMIGO_HAS_NATIVE_PACKAGE
#include <armigo/shader_native/package.h>
#endif
#include <armigo/shader/vertex_fetch.h>
#include "shader/pipeline/handle_program.h"

#include <armigo/graphics/texture_format.h>
#include <armigo/graphics/render_target.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <ctime>
#include <condition_variable>
#include <CommonCrypto/CommonDigest.h>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sys/sysctl.h>
#include <unordered_set>
#include <memory>
#include <thread>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <vector>
#include <mach/mach_time.h>
#include <pthread.h>
#include <map>
#include <set>
#include <tuple>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace armigo::platform::metal::nvn_driver {
namespace {

using namespace nvn;
using namespace nvn::api;

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void* pool);

// A stage's Metal library: its compiled library, or for a stage translated at
// run time its source, compiled with the options the offline package build
// uses (Metal 4.0, the compiler's default fast math). Metal keeps compiled
// source in its own cache, so a later run compiles the same source quickly.
id<MTLLibrary> load_stage_library(id<MTLDevice> device, const platform::CompiledGraphicsStage& stage,
                                  std::string& detail) {
    NSError* error = nil;
    id<MTLLibrary> library = nil;
    if (!stage.metallib.empty()) {
        dispatch_data_t data = dispatch_data_create(stage.metallib.data(), stage.metallib.size(),
                                                    nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        library = [device newLibraryWithData:data error:&error];
    } else if (!stage.msl_source.empty() || !stage.msl_source_file.empty()) {
        auto text = platform::stage_source(stage);
        if (const auto* failure = std::get_if<platform::MetalShaderPackageError>(&text)) {
            detail = "shader source load failed: " + failure->detail;
            return nil;
        }
        const auto& code = std::get<std::string>(text);
        auto* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        // Positions marked [[invariant]] agree across stages only so.
        options.preserveInvariance = YES;
        NSString* source = [[NSString alloc] initWithBytes:code.data() length:code.size()
                                                  encoding:NSUTF8StringEncoding];
        if (source) {
            // A source compile can take Metal's compiler arbitrarily long (some
            // stages never finish), and the replay thread must not wait for it
            // without bound: the guest waits on this submission's fence. The
            // compile runs asynchronously and is waited for up to a deadline;
            // past it the stage is unavailable for now (its draws are skipped)
            // and the library is taken once the compile finishes.
            struct Compile {
                std::mutex mutex;
                std::condition_variable done_signal;
                bool done = false;
                id<MTLLibrary> library = nil;
                NSError* error = nil;
            };
            static std::mutex compiles_mutex;
            static std::unordered_map<std::string, std::shared_ptr<Compile>> compiles;
            std::shared_ptr<Compile> compile;
            bool start = false;
            {
                const std::lock_guard lock(compiles_mutex);
                auto& entry = compiles[code];
                if (!entry) {
                    entry = std::make_shared<Compile>();
                    start = true;
                }
                compile = entry;
            }
            if (start)
                [device newLibraryWithSource:source options:options
                           completionHandler:^(id<MTLLibrary> result, NSError* failure) {
                    const std::lock_guard lock(compile->mutex);
                    compile->library = result;
                    compile->error = failure;
                    compile->done = true;
                    compile->done_signal.notify_all();
                }];
            static const auto deadline = [] {
                const char* value = std::getenv("ARMIGO_NVN_SOURCE_COMPILE_TIMEOUT_MS");
                const auto milliseconds = value ? std::strtoull(value, nullptr, 10) : 5000ull;
                return std::chrono::milliseconds(milliseconds ? milliseconds : 5000ull);
            }();
            std::unique_lock lock(compile->mutex);
            if (!compile->done_signal.wait_for(lock, start ? deadline : std::chrono::milliseconds(0),
                                               [&] { return compile->done; })) {
                detail = "the Metal source compile has not finished after " +
                         std::to_string(deadline.count()) + " ms; the stage is unavailable until it does";
                return nil;
            }
            library = compile->library;
            error = compile->error;
            if (library) {
                lock.unlock();
                const std::lock_guard forget(compiles_mutex);
                compiles.erase(code);
            }
        }
    }
    if (library && !stage.discard_entry_point.empty() &&
        ![library.functionNames containsObject:[NSString stringWithUTF8String:stage.discard_entry_point.c_str()]]) {
        detail = "shader library is missing its declared rasterizer-discard entry";
        return nil;
    }
    if (!library)
        detail = std::string("shader library load failed: ") +
                 (error ? error.localizedDescription.UTF8String : "the stage has no Metal library or source");
    return library;
}

std::uint64_t thread_cpu_ns() {
    timespec value{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value);
    return std::uint64_t(value.tv_sec) * 1'000'000'000 + value.tv_nsec;
}

void periodic_graphics_tick(std::uint64_t period_ns, ReplayThread thread,
                            const ReplaySchedulingPolicy& scheduling) {
    const char* role = thread == ReplayThread::worker ? "replay"
        : thread == ReplayThread::encode ? "encode" : "guest present";
    thread_local platform::PeriodicThreadBudget budget;
    thread_local bool warned = false;
    const auto result = budget.tick(std::chrono::nanoseconds(period_ns),
        std::chrono::nanoseconds(thread_cpu_ns()), [&](auto period, auto computation) {
            if (scheduling.apply)
                return scheduling.apply(scheduling.context, thread, period.count(), computation.count());
            if (period != std::chrono::nanoseconds::zero())
                return platform::declare_periodic_thread(period, computation);
            thread_standard_policy_data_t policy{};
            const auto status = thread_policy_set(pthread_mach_thread_np(pthread_self()),
                THREAD_STANDARD_POLICY, reinterpret_cast<thread_policy_t>(&policy),
                THREAD_STANDARD_POLICY_COUNT);
            // Mach policy changes permanently opt the thread out of pthread
            // QoS. Disabling this hint returns it to standard scheduling.
            return status == KERN_SUCCESS;
        });
    if (result == platform::PeriodicThreadBudget::Result::rejected && !warned) {
        armigo::diag::write(armigo::diag::Level::error,  "nvn_replay: %s scheduling declaration failed; retaining prior policy\n", role);
        warned = true;
    } else if (result == platform::PeriodicThreadBudget::Result::applied) {
        warned = false;
        timeline_event("thread_requested_period", static_cast<std::uint64_t>(thread), budget.period().count());
        timeline_event("thread_requested_budget", static_cast<std::uint64_t>(thread), budget.computation().count());
    }
}

constexpr std::size_t stage_count = 6;

// The title's barrier policy is game and ARMIGO_NVN_BARRIERS=full does not
// force the full policy back (any other value is reported once and ignored).
bool game_barriers() {
    static const bool forced_full = [] {
        const char* value = std::getenv("ARMIGO_NVN_BARRIERS");
        if (!value || !*value) return false;
        if (std::string_view(value) == "full") return true;
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo NVN: ARMIGO_NVN_BARRIERS accepts only full; ignoring %s\n", value);
        return false;
    }();
    return !forced_full && nvn::title_policy().barriers == nvn::TitlePolicy::Barriers::game;
}
constexpr std::size_t buffer_slots = 16;

// Hardware constant buffer 16 of an API 231 graphics stage is the push block
// once the stage has a push, and NVN uniform slot 13 before. Which one the
// title means is not settled, so the first draw of each stage in each case
// (push only, slot 13 only, both, neither) is reported.
void note_push_bank_reads(std::size_t stage, bool pushed, bool bound) {
    static std::atomic<std::uint32_t> reported{0};
    const auto which = (pushed ? 1u : 0u) | (bound ? 2u : 0u);
    const auto bit = 1u << (which * 8 + stage);
    if (reported.fetch_or(bit) & bit) return;
    static constexpr const char* cases[] = {
        "neither a push nor uniform slot 13, and reads zero", "the push block", "uniform slot 13",
        "both a push and uniform slot 13, and reads the push block"};
    armigo::diag::write(which == 1 || which == 2 ? armigo::diag::Level::info : armigo::diag::Level::warning,
                        "Armigo NVN 231.40: stage %zu reads constant buffer 16 with %s\n", stage, cases[which]);
}
constexpr std::size_t texture_slots = 128;
constexpr std::size_t bound_texture_slots = 256;
constexpr std::size_t separate_texture_slot_base = 128;
// Graphics driver data: the word GLSLC adds to the hardware instance ID.
constexpr std::size_t graphics_base_instance_offset = 0x644;
constexpr std::size_t sampler_slots = 128;
constexpr std::size_t image_slots = 32;
// Several queue submissions can belong to one frame. Keep room for the next
// frame to encode while earlier submissions still own their GPU storage: six
// slots, the title's nvn_policy.submission_slots, or ARMIGO_NVN_SUBMISSION_SLOTS
// (1 to 16) for measurement. Read when a replay encodes its first submission.
constexpr std::size_t max_submission_slots = 16;
std::size_t submission_slot_count() {
    if (const char* text = std::getenv("ARMIGO_NVN_SUBMISSION_SLOTS")) {
        const auto parsed = std::strtoull(text, nullptr, 10);
        if (parsed >= 1 && parsed <= max_submission_slots) return static_cast<std::size_t>(parsed);
    }
    if (const auto slots = nvn::title_policy().submission_slots; slots >= 1 && slots <= max_submission_slots)
        return slots;
    return 6;
}
// Pipelines a capturing session writes to one archive file before it starts
// another (see ReplayImpl::capture).
constexpr std::uint64_t archive_chunk_pipelines = 4096;

// A record's payload as a T; bytes a shorter record does not carry read zero.
// Only those are cleared: viewport and render-target records are hundreds of
// bytes, and clearing what the copy overwrites showed up in the walk.
template <class T>
T payload(const CommandRecordView& record) {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_default_constructible_v<T>);
    T value;
    const auto copied = std::min(sizeof value, record.payload_size);
    std::memcpy(&value, record.payload, copied);
    if (copied < sizeof value) std::memset(reinterpret_cast<std::byte*>(&value) + copied, 0, sizeof value - copied);
    return value;
}


std::optional<graphics::TextureFormat> texture_format(std::int32_t value) {
    using F = graphics::TextureFormat;
    switch (value) {
        case NVN_FORMAT_R8: return F::R8Unorm;
        case NVN_FORMAT_R8SN: return F::R8Snorm;
        case NVN_FORMAT_R8UI: return F::R8Uint;
        case NVN_FORMAT_R8I: return F::R8Sint;
        case NVN_FORMAT_R16F: return F::R16Float;
        case NVN_FORMAT_R16: return F::R16Unorm;
        case NVN_FORMAT_R16SN: return F::R16Snorm;
        case NVN_FORMAT_R16UI: return F::R16Uint;
        case NVN_FORMAT_R16I: return F::R16Sint;
        case NVN_FORMAT_R32F: return F::R32Float;
        case NVN_FORMAT_R32UI: return F::R32Uint;
        case NVN_FORMAT_R32I: return F::R32Sint;
        case NVN_FORMAT_RG8: return F::R8G8Unorm;
        case NVN_FORMAT_RG8SN: return F::R8G8Snorm;
        case NVN_FORMAT_RG8UI: return F::R8G8Uint;
        case NVN_FORMAT_RG8I: return F::R8G8Sint;
        case NVN_FORMAT_RG16F: return F::R16G16Float;
        case NVN_FORMAT_RG16: return F::R16G16Unorm;
        case NVN_FORMAT_RG16SN: return F::R16G16Snorm;
        case NVN_FORMAT_RG16UI: return F::R16G16Uint;
        case NVN_FORMAT_RG16I: return F::R16G16Sint;
        case NVN_FORMAT_RG32F: return F::R32G32Float;
        case NVN_FORMAT_RG32UI: return F::R32G32Uint;
        case NVN_FORMAT_RG32I: return F::R32G32Sint;
        case NVN_FORMAT_RGB32F: return F::R32G32B32Float;
        case NVN_FORMAT_RGB32UI: return F::R32G32B32Uint;
        case NVN_FORMAT_RGB32I: return F::R32G32B32Sint;
        case NVN_FORMAT_RGBA8:
        case NVN_FORMAT_RGBX8: return F::R8G8B8A8Unorm;
        case NVN_FORMAT_RGBA8SN:
        case NVN_FORMAT_RGBX8SN: return F::R8G8B8A8Snorm;
        case NVN_FORMAT_RGBA8UI:
        case NVN_FORMAT_RGBX8UI: return F::R8G8B8A8Uint;
        case NVN_FORMAT_RGBA8I:
        case NVN_FORMAT_RGBX8I: return F::R8G8B8A8Sint;
        case NVN_FORMAT_RGBA8_SRGB:
        case NVN_FORMAT_RGBX8_SRGB: return F::R8G8B8A8Srgb;
        case NVN_FORMAT_BGRA8:
        case NVN_FORMAT_BGRX8: return F::B8G8R8A8Unorm;
        case NVN_FORMAT_BGRA8_SRGB:
        case NVN_FORMAT_BGRX8_SRGB: return F::B8G8R8A8Srgb;
        case NVN_FORMAT_RGBA16F:
        case NVN_FORMAT_RGBX16F: return F::R16G16B16A16Float;
        case NVN_FORMAT_RGBA16:
        case NVN_FORMAT_RGBX16: return F::R16G16B16A16Unorm;
        case NVN_FORMAT_RGBA16SN:
        case NVN_FORMAT_RGBX16SN: return F::R16G16B16A16Snorm;
        case NVN_FORMAT_RGBA16UI:
        case NVN_FORMAT_RGBX16UI: return F::R16G16B16A16Uint;
        case NVN_FORMAT_RGBA16I:
        case NVN_FORMAT_RGBX16I: return F::R16G16B16A16Sint;
        case NVN_FORMAT_RGBA32F:
        case NVN_FORMAT_RGBX32F: return F::R32G32B32A32Float;
        case NVN_FORMAT_RGBA32UI:
        case NVN_FORMAT_RGBX32UI: return F::R32G32B32A32Uint;
        case NVN_FORMAT_RGBA32I:
        case NVN_FORMAT_RGBX32I: return F::R32G32B32A32Sint;
        case NVN_FORMAT_RGBA4: return F::R4G4B4A4Unorm;
        case NVN_FORMAT_RGB5A1: return F::R5G5B5A1Unorm;
        case NVN_FORMAT_RGB565: return F::R5G6B5Unorm;
        case NVN_FORMAT_RGB10A2: return F::R10G10B10A2Unorm;
        case NVN_FORMAT_RGB10A2UI: return F::R10G10B10A2Uint;
        case NVN_FORMAT_R11G11B10F: return F::R11G11B10Float;
        case NVN_FORMAT_RGB9E5F: return F::R9G9B9E5Float;
        case NVN_FORMAT_DEPTH16: return F::D16Unorm;
        case NVN_FORMAT_DEPTH32F: return F::D32Float;
        case NVN_FORMAT_DEPTH24_STENCIL8: return F::D24UnormS8Uint;
        case NVN_FORMAT_DEPTH32F_STENCIL8: return F::D32FloatS8Uint;
        case NVN_FORMAT_RGBA_DXT1:
        case NVN_FORMAT_RGB_DXT1: return F::Bc1RgbaUnorm;
        case NVN_FORMAT_RGBA_DXT1_SRGB:
        case NVN_FORMAT_RGB_DXT1_SRGB: return F::Bc1RgbaSrgb;
        case NVN_FORMAT_RGBA_DXT3: return F::Bc2Unorm;
        case NVN_FORMAT_RGBA_DXT3_SRGB: return F::Bc2Srgb;
        case NVN_FORMAT_RGBA_DXT5: return F::Bc3Unorm;
        case NVN_FORMAT_RGBA_DXT5_SRGB: return F::Bc3Srgb;
        case NVN_FORMAT_RGTC1_UNORM: return F::Bc4Unorm;
        case NVN_FORMAT_RGTC1_SNORM: return F::Bc4Snorm;
        case NVN_FORMAT_RGTC2_UNORM: return F::Bc5Unorm;
        case NVN_FORMAT_RGTC2_SNORM: return F::Bc5Snorm;
        case NVN_FORMAT_BPTC_UNORM: return F::Bc7Unorm;
        case NVN_FORMAT_BPTC_UNORM_SRGB: return F::Bc7Srgb;
        case NVN_FORMAT_BPTC_UFLOAT: return F::Bc6HUfloat;
        case NVN_FORMAT_BPTC_SFLOAT: return F::Bc6HSfloat;
        default: return std::nullopt;
    }
}

std::optional<MTLVertexFormat> vertex_format(std::uint32_t value) {
    switch (value) {
        case NVN_FORMAT_R8: return MTLVertexFormatUCharNormalized;
        case NVN_FORMAT_R8SN: return MTLVertexFormatCharNormalized;
        case NVN_FORMAT_R8UI: return MTLVertexFormatUChar;
        case NVN_FORMAT_R8I: return MTLVertexFormatChar;
        case NVN_FORMAT_R16F: return MTLVertexFormatHalf;
        case NVN_FORMAT_R16: return MTLVertexFormatUShortNormalized;
        case NVN_FORMAT_R16SN: return MTLVertexFormatShortNormalized;
        case NVN_FORMAT_R16UI: return MTLVertexFormatUShort;
        case NVN_FORMAT_R16I: return MTLVertexFormatShort;
        case NVN_FORMAT_R32F: return MTLVertexFormatFloat;
        case NVN_FORMAT_R32UI: return MTLVertexFormatUInt;
        case NVN_FORMAT_R32I: return MTLVertexFormatInt;
        case NVN_FORMAT_RG8: return MTLVertexFormatUChar2Normalized;
        case NVN_FORMAT_RG8SN: return MTLVertexFormatChar2Normalized;
        case NVN_FORMAT_RG8UI: return MTLVertexFormatUChar2;
        case NVN_FORMAT_RG8I: return MTLVertexFormatChar2;
        case NVN_FORMAT_RG16F: return MTLVertexFormatHalf2;
        case NVN_FORMAT_RG16: return MTLVertexFormatUShort2Normalized;
        case NVN_FORMAT_RG16SN: return MTLVertexFormatShort2Normalized;
        case NVN_FORMAT_RG16UI: return MTLVertexFormatUShort2;
        case NVN_FORMAT_RG16I: return MTLVertexFormatShort2;
        case NVN_FORMAT_RG32F: return MTLVertexFormatFloat2;
        case NVN_FORMAT_RG32UI: return MTLVertexFormatUInt2;
        case NVN_FORMAT_RG32I: return MTLVertexFormatInt2;
        case NVN_FORMAT_RGB8: return MTLVertexFormatUChar3Normalized;
        case NVN_FORMAT_RGB8SN: return MTLVertexFormatChar3Normalized;
        case NVN_FORMAT_RGB8UI: return MTLVertexFormatUChar3;
        case NVN_FORMAT_RGB8I: return MTLVertexFormatChar3;
        case NVN_FORMAT_RGB16F: return MTLVertexFormatHalf3;
        case NVN_FORMAT_RGB16: return MTLVertexFormatUShort3Normalized;
        case NVN_FORMAT_RGB16SN: return MTLVertexFormatShort3Normalized;
        case NVN_FORMAT_RGB16UI: return MTLVertexFormatUShort3;
        case NVN_FORMAT_RGB16I: return MTLVertexFormatShort3;
        case NVN_FORMAT_RGB10A2: return MTLVertexFormatUInt1010102Normalized;
        case NVN_FORMAT_RGB10A2SN: return MTLVertexFormatInt1010102Normalized;
        case NVN_FORMAT_R11G11B10F: return MTLVertexFormatFloatRG11B10;
        case NVN_FORMAT_BGRA8: return MTLVertexFormatUChar4Normalized_BGRA;
        case NVN_FORMAT_RGB32F: return MTLVertexFormatFloat3;
        case NVN_FORMAT_RGB32UI: return MTLVertexFormatUInt3;
        case NVN_FORMAT_RGB32I: return MTLVertexFormatInt3;
        case NVN_FORMAT_RGBA8: return MTLVertexFormatUChar4Normalized;
        case NVN_FORMAT_RGBA8SN: return MTLVertexFormatChar4Normalized;
        case NVN_FORMAT_RGBA8UI: return MTLVertexFormatUChar4;
        case NVN_FORMAT_RGBA8I: return MTLVertexFormatChar4;
        case NVN_FORMAT_RGBA16F: return MTLVertexFormatHalf4;
        case NVN_FORMAT_RGBA16: return MTLVertexFormatUShort4Normalized;
        case NVN_FORMAT_RGBA16SN: return MTLVertexFormatShort4Normalized;
        case NVN_FORMAT_RGBA16UI: return MTLVertexFormatUShort4;
        case NVN_FORMAT_RGBA16I: return MTLVertexFormatShort4;
        case NVN_FORMAT_RGBA32F: return MTLVertexFormatFloat4;
        case NVN_FORMAT_RGBA32UI: return MTLVertexFormatUInt4;
        case NVN_FORMAT_RGBA32I: return MTLVertexFormatInt4;
        // Integer-to-float formats take the integer layout; Metal converts an
        // integer vertex format read by a float input to the same float value.
        case NVN_FORMAT_R8_UI2F: return MTLVertexFormatUChar;
        case NVN_FORMAT_R8_I2F: return MTLVertexFormatChar;
        case NVN_FORMAT_R16_UI2F: return MTLVertexFormatUShort;
        case NVN_FORMAT_R16_I2F: return MTLVertexFormatShort;
        case NVN_FORMAT_R32_UI2F: return MTLVertexFormatUInt;
        case NVN_FORMAT_R32_I2F: return MTLVertexFormatInt;
        case NVN_FORMAT_RG8_UI2F: return MTLVertexFormatUChar2;
        case NVN_FORMAT_RG8_I2F: return MTLVertexFormatChar2;
        case NVN_FORMAT_RG16_UI2F: return MTLVertexFormatUShort2;
        case NVN_FORMAT_RG16_I2F: return MTLVertexFormatShort2;
        case NVN_FORMAT_RG32_UI2F: return MTLVertexFormatUInt2;
        case NVN_FORMAT_RG32_I2F: return MTLVertexFormatInt2;
        case NVN_FORMAT_RGB8_UI2F: return MTLVertexFormatUChar3;
        case NVN_FORMAT_RGB8_I2F: return MTLVertexFormatChar3;
        case NVN_FORMAT_RGB16_UI2F: return MTLVertexFormatUShort3;
        case NVN_FORMAT_RGB16_I2F: return MTLVertexFormatShort3;
        case NVN_FORMAT_RGB32_UI2F: return MTLVertexFormatUInt3;
        case NVN_FORMAT_RGB32_I2F: return MTLVertexFormatInt3;
        case NVN_FORMAT_RGBA8_UI2F: return MTLVertexFormatUChar4;
        case NVN_FORMAT_RGBA8_I2F: return MTLVertexFormatChar4;
        case NVN_FORMAT_RGBA16_UI2F: return MTLVertexFormatUShort4;
        case NVN_FORMAT_RGBA16_I2F: return MTLVertexFormatShort4;
        case NVN_FORMAT_RGBA32_UI2F: return MTLVertexFormatUInt4;
        case NVN_FORMAT_RGBA32_I2F: return MTLVertexFormatInt4;
        default: return std::nullopt;
    }
}

// The fetch-table storage of an NVN vertex format: its kind, component count,
// and whether it scales integers to float (the *_UI2F and *_I2F formats; the
// plain integer formats write the integer itself).
struct VertexFetchFormat {
    shader::VertexFetchKind kind;
    std::uint32_t components;
    bool scaled;
};
std::optional<VertexFetchFormat> vertex_fetch_format(std::uint32_t value) {
    using K = shader::VertexFetchKind;
    switch (value) {
        case NVN_FORMAT_R8: return VertexFetchFormat{K::unorm8, 1u, false};
        case NVN_FORMAT_R8SN: return VertexFetchFormat{K::snorm8, 1u, false};
        case NVN_FORMAT_R8UI: return VertexFetchFormat{K::uint8, 1u, false};
        case NVN_FORMAT_R8_UI2F: return VertexFetchFormat{K::uint8, 1u, true};
        case NVN_FORMAT_R8I: return VertexFetchFormat{K::sint8, 1u, false};
        case NVN_FORMAT_R8_I2F: return VertexFetchFormat{K::sint8, 1u, true};
        case NVN_FORMAT_R16F: return VertexFetchFormat{K::float16, 1u, false};
        case NVN_FORMAT_R16: return VertexFetchFormat{K::unorm16, 1u, false};
        case NVN_FORMAT_R16SN: return VertexFetchFormat{K::snorm16, 1u, false};
        case NVN_FORMAT_R16UI: return VertexFetchFormat{K::uint16, 1u, false};
        case NVN_FORMAT_R16_UI2F: return VertexFetchFormat{K::uint16, 1u, true};
        case NVN_FORMAT_R16I: return VertexFetchFormat{K::sint16, 1u, false};
        case NVN_FORMAT_R16_I2F: return VertexFetchFormat{K::sint16, 1u, true};
        case NVN_FORMAT_R32F: return VertexFetchFormat{K::float32, 1u, false};
        case NVN_FORMAT_R32UI: return VertexFetchFormat{K::uint32, 1u, false};
        case NVN_FORMAT_R32_UI2F: return VertexFetchFormat{K::uint32, 1u, true};
        case NVN_FORMAT_R32I: return VertexFetchFormat{K::sint32, 1u, false};
        case NVN_FORMAT_R32_I2F: return VertexFetchFormat{K::sint32, 1u, true};
        case NVN_FORMAT_RG8: return VertexFetchFormat{K::unorm8, 2u, false};
        case NVN_FORMAT_RG8SN: return VertexFetchFormat{K::snorm8, 2u, false};
        case NVN_FORMAT_RG8UI: return VertexFetchFormat{K::uint8, 2u, false};
        case NVN_FORMAT_RG8_UI2F: return VertexFetchFormat{K::uint8, 2u, true};
        case NVN_FORMAT_RG8I: return VertexFetchFormat{K::sint8, 2u, false};
        case NVN_FORMAT_RG8_I2F: return VertexFetchFormat{K::sint8, 2u, true};
        case NVN_FORMAT_RG16F: return VertexFetchFormat{K::float16, 2u, false};
        case NVN_FORMAT_RG16: return VertexFetchFormat{K::unorm16, 2u, false};
        case NVN_FORMAT_RG16SN: return VertexFetchFormat{K::snorm16, 2u, false};
        case NVN_FORMAT_RG16UI: return VertexFetchFormat{K::uint16, 2u, false};
        case NVN_FORMAT_RG16_UI2F: return VertexFetchFormat{K::uint16, 2u, true};
        case NVN_FORMAT_RG16I: return VertexFetchFormat{K::sint16, 2u, false};
        case NVN_FORMAT_RG16_I2F: return VertexFetchFormat{K::sint16, 2u, true};
        case NVN_FORMAT_RG32F: return VertexFetchFormat{K::float32, 2u, false};
        case NVN_FORMAT_RG32UI: return VertexFetchFormat{K::uint32, 2u, false};
        case NVN_FORMAT_RG32_UI2F: return VertexFetchFormat{K::uint32, 2u, true};
        case NVN_FORMAT_RG32I: return VertexFetchFormat{K::sint32, 2u, false};
        case NVN_FORMAT_RG32_I2F: return VertexFetchFormat{K::sint32, 2u, true};
        case NVN_FORMAT_RGB8: return VertexFetchFormat{K::unorm8, 3u, false};
        case NVN_FORMAT_RGB8SN: return VertexFetchFormat{K::snorm8, 3u, false};
        case NVN_FORMAT_RGB8UI: return VertexFetchFormat{K::uint8, 3u, false};
        case NVN_FORMAT_RGB8_UI2F: return VertexFetchFormat{K::uint8, 3u, true};
        case NVN_FORMAT_RGB8I: return VertexFetchFormat{K::sint8, 3u, false};
        case NVN_FORMAT_RGB8_I2F: return VertexFetchFormat{K::sint8, 3u, true};
        case NVN_FORMAT_RGB16F: return VertexFetchFormat{K::float16, 3u, false};
        case NVN_FORMAT_RGB16: return VertexFetchFormat{K::unorm16, 3u, false};
        case NVN_FORMAT_RGB16SN: return VertexFetchFormat{K::snorm16, 3u, false};
        case NVN_FORMAT_RGB16UI: return VertexFetchFormat{K::uint16, 3u, false};
        case NVN_FORMAT_RGB16_UI2F: return VertexFetchFormat{K::uint16, 3u, true};
        case NVN_FORMAT_RGB16I: return VertexFetchFormat{K::sint16, 3u, false};
        case NVN_FORMAT_RGB16_I2F: return VertexFetchFormat{K::sint16, 3u, true};
        case NVN_FORMAT_RGB32F: return VertexFetchFormat{K::float32, 3u, false};
        case NVN_FORMAT_RGB32UI: return VertexFetchFormat{K::uint32, 3u, false};
        case NVN_FORMAT_RGB32_UI2F: return VertexFetchFormat{K::uint32, 3u, true};
        case NVN_FORMAT_RGB32I: return VertexFetchFormat{K::sint32, 3u, false};
        case NVN_FORMAT_RGB32_I2F: return VertexFetchFormat{K::sint32, 3u, true};
        case NVN_FORMAT_RGBA8: return VertexFetchFormat{K::unorm8, 4u, false};
        case NVN_FORMAT_RGBA8SN: return VertexFetchFormat{K::snorm8, 4u, false};
        case NVN_FORMAT_RGBA8UI: return VertexFetchFormat{K::uint8, 4u, false};
        case NVN_FORMAT_RGBA8_UI2F: return VertexFetchFormat{K::uint8, 4u, true};
        case NVN_FORMAT_RGBA8I: return VertexFetchFormat{K::sint8, 4u, false};
        case NVN_FORMAT_RGBA8_I2F: return VertexFetchFormat{K::sint8, 4u, true};
        case NVN_FORMAT_RGBA16F: return VertexFetchFormat{K::float16, 4u, false};
        case NVN_FORMAT_RGBA16: return VertexFetchFormat{K::unorm16, 4u, false};
        case NVN_FORMAT_RGBA16SN: return VertexFetchFormat{K::snorm16, 4u, false};
        case NVN_FORMAT_RGBA16UI: return VertexFetchFormat{K::uint16, 4u, false};
        case NVN_FORMAT_RGBA16_UI2F: return VertexFetchFormat{K::uint16, 4u, true};
        case NVN_FORMAT_RGBA16I: return VertexFetchFormat{K::sint16, 4u, false};
        case NVN_FORMAT_RGBA16_I2F: return VertexFetchFormat{K::sint16, 4u, true};
        case NVN_FORMAT_RGBA32F: return VertexFetchFormat{K::float32, 4u, false};
        case NVN_FORMAT_RGBA32UI: return VertexFetchFormat{K::uint32, 4u, false};
        case NVN_FORMAT_RGBA32_UI2F: return VertexFetchFormat{K::uint32, 4u, true};
        case NVN_FORMAT_RGBA32I: return VertexFetchFormat{K::sint32, 4u, false};
        case NVN_FORMAT_RGBA32_I2F: return VertexFetchFormat{K::sint32, 4u, true};
        case NVN_FORMAT_RGB10A2: return VertexFetchFormat{K::unorm10_10_10_2, 4u, false};
        case NVN_FORMAT_RGB10A2SN: return VertexFetchFormat{K::snorm10_10_10_2, 4u, false};
        case NVN_FORMAT_R11G11B10F: return VertexFetchFormat{K::float11_11_10, 3u, false};
        case NVN_FORMAT_BGRA8: return VertexFetchFormat{K::unorm8_bgra, 4u, false};
        default: return std::nullopt;
    }
}

std::optional<MTLPrimitiveType> primitive_type(std::int32_t primitive) {
    switch (primitive) {
        case NVN_DRAW_PRIMITIVE_POINTS: return MTLPrimitiveTypePoint;
        case NVN_DRAW_PRIMITIVE_LINES: return MTLPrimitiveTypeLine;
        case NVN_DRAW_PRIMITIVE_LINE_STRIP: return MTLPrimitiveTypeLineStrip;
        case NVN_DRAW_PRIMITIVE_TRIANGLES: return MTLPrimitiveTypeTriangle;
        case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP: return MTLPrimitiveTypeTriangleStrip;
        default: return std::nullopt;
    }
}

// Metal has no quads or triangle fans; both draw as triangle lists
// (nvn_primitive_expansion.h).
bool expands_to_triangles(std::int32_t primitive) {
    return primitive == NVN_DRAW_PRIMITIVE_QUADS || primitive == NVN_DRAW_PRIMITIVE_TRIANGLE_FAN;
}

MTLPrimitiveTopologyClass topology_class(std::int32_t primitive) {
    if (primitive == NVN_DRAW_PRIMITIVE_POINTS) return MTLPrimitiveTopologyClassPoint;
    if (primitive == NVN_DRAW_PRIMITIVE_LINES || primitive == NVN_DRAW_PRIMITIVE_LINE_STRIP)
        return MTLPrimitiveTopologyClassLine;
    return MTLPrimitiveTopologyClassTriangle;
}

std::optional<MTLCompareFunction> compare_function(std::uint32_t value) {
    switch (value) {
        case NVN_COMPARE_FUNC_NEVER: return MTLCompareFunctionNever;
        case NVN_COMPARE_FUNC_LESS: return MTLCompareFunctionLess;
        case NVN_COMPARE_FUNC_EQUAL: return MTLCompareFunctionEqual;
        case NVN_COMPARE_FUNC_LEQUAL: return MTLCompareFunctionLessEqual;
        case NVN_COMPARE_FUNC_GREATER: return MTLCompareFunctionGreater;
        case NVN_COMPARE_FUNC_NOTEQUAL: return MTLCompareFunctionNotEqual;
        case NVN_COMPARE_FUNC_GEQUAL: return MTLCompareFunctionGreaterEqual;
        case NVN_COMPARE_FUNC_ALWAYS: return MTLCompareFunctionAlways;
        default: return std::nullopt;
    }
}

std::optional<MTLStencilOperation> stencil_operation(std::uint32_t value) {
    switch (value) {
        case NVN_STENCIL_OP_KEEP: return MTLStencilOperationKeep;
        case NVN_STENCIL_OP_ZERO: return MTLStencilOperationZero;
        case NVN_STENCIL_OP_REPLACE: return MTLStencilOperationReplace;
        case NVN_STENCIL_OP_INCR: return MTLStencilOperationIncrementClamp;
        case NVN_STENCIL_OP_DECR: return MTLStencilOperationDecrementClamp;
        case NVN_STENCIL_OP_INVERT: return MTLStencilOperationInvert;
        case NVN_STENCIL_OP_INCR_WRAP: return MTLStencilOperationIncrementWrap;
        case NVN_STENCIL_OP_DECR_WRAP: return MTLStencilOperationDecrementWrap;
        default: return std::nullopt;
    }
}

std::optional<MTLBlendFactor> blend_factor(std::uint32_t value) {
    switch (value) {
        case NVN_BLEND_FUNC_ZERO: return MTLBlendFactorZero;
        case NVN_BLEND_FUNC_ONE: return MTLBlendFactorOne;
        case NVN_BLEND_FUNC_SRC_COLOR: return MTLBlendFactorSourceColor;
        case NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR: return MTLBlendFactorOneMinusSourceColor;
        case NVN_BLEND_FUNC_SRC_ALPHA: return MTLBlendFactorSourceAlpha;
        case NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
        case NVN_BLEND_FUNC_DST_ALPHA: return MTLBlendFactorDestinationAlpha;
        case NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA: return MTLBlendFactorOneMinusDestinationAlpha;
        case NVN_BLEND_FUNC_DST_COLOR: return MTLBlendFactorDestinationColor;
        case NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR: return MTLBlendFactorOneMinusDestinationColor;
        case NVN_BLEND_FUNC_SRC_ALPHA_SATURATE: return MTLBlendFactorSourceAlphaSaturated;
        case NVN_BLEND_FUNC_CONSTANT_COLOR: return MTLBlendFactorBlendColor;
        case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR: return MTLBlendFactorOneMinusBlendColor;
        case NVN_BLEND_FUNC_CONSTANT_ALPHA: return MTLBlendFactorBlendAlpha;
        case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA: return MTLBlendFactorOneMinusBlendAlpha;
        default: return std::nullopt;
    }
}

std::optional<MTLBlendOperation> blend_operation(std::uint32_t value) {
    switch (value) {
        case NVN_BLEND_EQUATION_ADD: return MTLBlendOperationAdd;
        case NVN_BLEND_EQUATION_SUB: return MTLBlendOperationSubtract;
        case NVN_BLEND_EQUATION_REVERSE_SUB: return MTLBlendOperationReverseSubtract;
        case NVN_BLEND_EQUATION_MIN: return MTLBlendOperationMin;
        case NVN_BLEND_EQUATION_MAX: return MTLBlendOperationMax;
        default: return std::nullopt;
    }
}

NSUInteger texture_layers(id<MTLTexture> texture) {
    return texture.textureType == MTLTextureTypeCube || texture.textureType == MTLTextureTypeCubeArray
        ? texture.arrayLength * 6 : texture.arrayLength;
}

bool array_texture(id<MTLTexture> texture) {
    return texture.textureType == MTLTextureType1DArray || texture.textureType == MTLTextureType2DArray ||
           texture.textureType == MTLTextureTypeCube || texture.textureType == MTLTextureTypeCubeArray;
}

bool valid_region(id<MTLTexture> texture, const NVNcopyRegion& region) {
    if (region.xoffset < 0 || region.yoffset < 0 || region.zoffset < 0 ||
        region.width < 0 || region.height < 0 || region.depth < 0) return false;
    return std::uint64_t(region.xoffset) + region.width <= texture.width &&
           std::uint64_t(region.yoffset) + region.height <= texture.height &&
           std::uint64_t(region.zoffset) + region.depth <=
               (array_texture(texture) ? texture_layers(texture) : texture.depth);
}

MTLScissorRect clip_scissor(const Scissor& source, NSUInteger width, NSUInteger height) {
    const auto x = std::clamp<std::int64_t>(source.x, 0, width);
    const auto y = std::clamp<std::int64_t>(source.y, 0, height);
    const auto end_x = std::clamp<std::int64_t>(std::int64_t(source.x) + source.width, x, width);
    const auto end_y = std::clamp<std::int64_t>(std::int64_t(source.y) + source.height, y, height);
    return {static_cast<NSUInteger>(x), static_cast<NSUInteger>(y),
            static_cast<NSUInteger>(end_x - x), static_cast<NSUInteger>(end_y - y)};
}

bool replace_state(void* destination, const void* source, std::size_t size) {
    if (std::memcmp(destination, source, size) == 0) return false;
    std::memcpy(destination, source, size);
    return true;
}

struct BoundBuffer {
    std::uint64_t address = 0;
    std::uint64_t size = 0;
};

// Logical bindings survive changes to the executing geometry pipeline.
// The six Metal slots below are projections, not the public binding namespace.
struct ReplayPrograms {
    struct Cached {
        std::shared_ptr<const ResolvedProgram> program;
        bool failed = false;
    };
    std::array<Cached, stage_count> ordinary;
    Cached task, task_mesh, standalone_mesh, auxiliary_mesh;
    std::array<std::shared_ptr<const ResolvedProgram>, stage_count> stage_programs;
    std::uint32_t failed_stages = 0, write_stages = 0;
    std::int32_t geometry_mode = 0;

    static bool stage_writes(const ResolvedProgram* program, std::size_t stage) {
        const auto* shader = program ? program->stages[stage].get() : nullptr;
        return shader && ((shader->guest_memory && shader->guest_memory->writes) ||
            std::ranges::any_of(shader->storage_buffers, [](const auto& resource) {
                return resource.access != platform::CompiledShaderResourceAccess::Read;
            }));
    }
    void update_write_stage(std::size_t stage) {
        const auto bit = 1u << stage;
        write_stages = stage_writes(stage_programs[stage].get(), stage)
            ? write_stages | bit : write_stages & ~bit;
    }
    bool select() {
        bool changed = false;
        std::uint32_t failures = 0;
        for (std::size_t stage = 0; stage != stage_count; ++stage) {
            const Cached* selected = &ordinary[stage];
            if (geometry_mode && stage != NVN_SHADER_STAGE_FRAGMENT && stage != NVN_SHADER_STAGE_COMPUTE) {
                selected = nullptr;
                if (stage == NVN_SHADER_STAGE_TESS_CONTROL && geometry_mode == 1) selected = &task;
                if (stage == NVN_SHADER_STAGE_TESS_EVALUATION)
                    selected = geometry_mode == 1 ? &task_mesh : &standalone_mesh;
            }
            const auto* next = selected && !selected->failed ? selected->program.get() : nullptr;
            if (stage_programs[stage].get() != next) {
                stage_programs[stage] = next ? selected->program : nullptr;
                update_write_stage(stage);
                changed = true;
            }
            if (selected && selected->failed) failures |= 1u << stage;
        }
        changed |= failed_stages != failures;
        failed_stages = failures;
        return changed;
    }
    static void update(Cached& cache, const std::shared_ptr<const ResolvedProgram>& program,
                       bool present, bool failed) {
        const auto* next = present ? program.get() : nullptr;
        if (cache.program.get() != next) cache.program = present ? program : nullptr;
        cache.failed = failed;
    }
    bool bind(const BindProgram& command, const std::shared_ptr<const ResolvedProgram>& program) {
        const auto mask = static_cast<std::uint32_t>(command.stages);
        if (!mask) return false;
        const auto present = program ? program->logical_stages : 0u;
        constexpr auto compute_bit = 1u << NVN_SHADER_STAGE_COMPUTE;
        const bool compute = (present & mask & compute_bit) != 0;
        constexpr auto graphics_bits = ((1u << logical_binding_stage_count) - 1) & ~compute_bit;
        const auto published = compute ? compute_bit : graphics_bits;
        // Publish the complete graphics list, including its opposite-route
        // clear, before applying the requested mask's missing-stage clears.
        const bool has_task = (present & (1u << 6)) != 0;
        const bool has_mesh = (present & (1u << 7)) != 0;
        if (!compute && has_mesh) {
            auto& selected = program->mesh_uses_task ? task_mesh : standalone_mesh;
            auto& opposite = program->mesh_uses_task ? standalone_mesh : task_mesh;
            opposite = {};
            update(selected, program, true,
                (program->failed_stages & (1u << NVN_SHADER_STAGE_TESS_EVALUATION)) != 0);
            update(auxiliary_mesh, program, program->mesh_primitive_header,
                program->mesh_primitive_header && selected.failed);
        }
        if (!compute && has_task)
            update(task, program, true,
                (program->failed_stages & (1u << NVN_SHADER_STAGE_TESS_CONTROL)) != 0);
        for (std::size_t stage = 0; stage != stage_count; ++stage) {
            const auto bit = 1u << stage;
            if (!(present & published & bit)) continue;
            update(ordinary[stage], program, true, (program->failed_stages & bit) != 0);
        }
        const auto missing = mask & ~present;
        for (std::size_t stage = 0; stage != stage_count; ++stage)
            if (missing & (1u << stage)) ordinary[stage] = {};
        if (missing & (1u << 6)) task = {};
        if (missing & (1u << 7)) task_mesh = standalone_mesh = auxiliary_mesh = {};
        return select();
    }
    std::array<program_capture::Cache, 10> capture_caches() const {
        std::array<program_capture::Cache, 10> result;
        const auto copy = [&](std::size_t index, const Cached& entry, std::size_t physical) {
            auto& out = result[index]; out.failed = entry.failed;
            if (!entry.program) return;
            out.generation = entry.program->state.serial; out.capture_call = entry.program->capture_call;
            const auto& shader = entry.program->stages[physical];
            if (!shader) return;
            if (!shader->native_stage_identity.empty()) {
                out.stage_identity = shader->native_stage_identity; out.identity_domain = "native-control-payload";
            } else { out.stage_identity = shader->identity; out.identity_domain = "compiled-stage"; }
            if (index == 9) out.identity_domain += ":auxiliary-metadata";
        };
        for (std::size_t i = 0; i != ordinary.size(); ++i) copy(i, ordinary[i], i);
        copy(6, task, 3); copy(7, task_mesh, 4); copy(8, standalone_mesh, 4); copy(9, auxiliary_mesh, 4);
        return result;
    }
    std::size_t resource_stage(std::size_t physical) const {
        if (geometry_mode && physical == NVN_SHADER_STAGE_TESS_CONTROL) return 6;
        if (geometry_mode && physical == NVN_SHADER_STAGE_TESS_EVALUATION) return 7;
        return physical;
    }
    std::optional<std::size_t> execution_stage(std::size_t logical) const {
        if (!geometry_mode) return logical < stage_count ? std::optional{logical} : std::nullopt;
        if (logical == 6 && geometry_mode == 1) return NVN_SHADER_STAGE_TESS_CONTROL;
        if (logical == 7) return NVN_SHADER_STAGE_TESS_EVALUATION;
        if (logical == NVN_SHADER_STAGE_FRAGMENT || logical == NVN_SHADER_STAGE_COMPUTE) return logical;
        return std::nullopt;
    }
};

struct ReplayState : ReplayPrograms {
    SetRenderTargets targets{};
    std::array<std::shared_ptr<const TextureState>, 8> target_owners;
    std::shared_ptr<const TextureState> depth_owner;
    std::array<std::array<BoundBuffer, buffer_slots>, logical_binding_stage_count> uniform{};
    std::array<std::array<BoundBuffer, buffer_slots>, logical_binding_stage_count> storage{};
    std::array<std::array<std::uint64_t, bound_texture_slots>, logical_binding_stage_count> textures{};
    std::array<std::uint64_t, logical_binding_stage_count> texture_prefix{};
    std::array<std::array<std::uint64_t, sampler_slots>, logical_binding_stage_count> samplers{};
    std::array<std::array<std::uint64_t, image_slots>, logical_binding_stage_count> images{};
    // Each graphics stage's push-constant block and the hardware constant
    // buffer its shaders read it as (0 before any push).
    std::array<std::array<std::byte, push_constant_bytes>, logical_binding_stage_count> push_constants{};
    std::array<std::uint32_t, logical_binding_stage_count> push_bank{};
    auto& uniform_at(std::size_t physical) { return uniform[resource_stage(physical)]; }
    const auto& uniform_at(std::size_t physical) const { return uniform[resource_stage(physical)]; }
    auto& storage_at(std::size_t physical) { return storage[resource_stage(physical)]; }
    const auto& storage_at(std::size_t physical) const { return storage[resource_stage(physical)]; }
    auto& textures_at(std::size_t physical) { return textures[resource_stage(physical)]; }
    const auto& textures_at(std::size_t physical) const { return textures[resource_stage(physical)]; }
    auto& prefix_at(std::size_t physical) { return texture_prefix[resource_stage(physical)]; }
    const auto& prefix_at(std::size_t physical) const { return texture_prefix[resource_stage(physical)]; }
    auto& samplers_at(std::size_t physical) { return samplers[resource_stage(physical)]; }
    const auto& samplers_at(std::size_t physical) const { return samplers[resource_stage(physical)]; }
    auto& images_at(std::size_t physical) { return images[resource_stage(physical)]; }
    const auto& images_at(std::size_t physical) const { return images[resource_stage(physical)]; }
    auto& push_at(std::size_t physical) { return push_constants[resource_stage(physical)]; }
    const auto& push_at(std::size_t physical) const { return push_constants[resource_stage(physical)]; }
    auto& push_bank_at(std::size_t physical) { return push_bank[resource_stage(physical)]; }
    const auto& push_bank_at(std::size_t physical) const { return push_bank[resource_stage(physical)]; }
    // Sampler registrations made before the current draw was recorded, from
    // the latest stamp or binding record (0 before any).
    std::uint64_t sampler_registrations = 0;
    std::array<NVNbufferRange, buffer_slots> vertex_buffers{};
    BoundBuffer index_buffer{};
    std::array<VertexAttributeStateBits, buffer_slots> attributes{};
    std::array<VertexStreamStateBits, buffer_slots> streams{};
    std::array<BlendStateBits, 8> blend{};
    ColorStateBits color{};
    ChannelMaskStateBits channel_mask{0xffffffffu};
    MultisampleStateBits multisample{};
    PolygonStateBits polygon{};
    DepthStencilStateBits depth_stencil{};
    std::array<MTLViewport, 16> viewports = [] {
        std::array<MTLViewport, 16> values{};
        for (auto& value : values) value.zfar = 1.0;
        return values;
    }();
    std::size_t viewport_count = 0;
    std::array<Scissor, 16> scissors{};
    std::size_t scissor_count = 0;
    float depth_near = 0.0f, depth_far = 1.0f;
    float polygon_factor = 0.0f, polygon_units = 0.0f, polygon_clamp = 0.0f;
    std::array<float, 4> blend_color{};
    std::uint32_t stencil_front_read = ~0u, stencil_back_read = ~0u;
    std::uint32_t stencil_front_write = ~0u, stencil_back_write = ~0u;
    std::uint32_t stencil_front_ref = 0, stencil_back_ref = 0;
    std::int64_t copy_row_stride = 0;
    std::int64_t copy_image_stride = 0;
    bool depth_clamp = false;
    bool render_enabled = true;
    // A GPU-predicated conditional block was still open when its submission
    // ended: the next submission reads the predicate once that one retired.
    bool predicate_pending = false;
    std::int32_t predicate_mode = 0;
    std::uint64_t predicate_address = 0;
    bool rasterizer_discard = false;
    bool primitive_restart = false;
    std::uint32_t restart_index = UINT32_MAX;
    std::uint32_t sample_mask = UINT32_MAX;
    bool samples_active = false;
    float line_width = 1.0f, point_size = 1.0f;
};

// Conditional rendering on the GPU. Metal has no predicated draws, so a draw
// in a conditional block is encoded as an indirect draw whose argument record
// the CPU writes, and a kernel that runs before the block compares the two
// counters and, when the block must not render, zeroes the instance count of
// every record of the block. A counter is read from memory, or, for a
// samples-passed report of the same submission made after a reset, summed
// from its visibility-result slots. The layout matches the kernel's struct.
struct PredicateJob {
    std::uint64_t visibility;
    std::uint64_t memory[2];
    std::uint32_t begin[2], end[2];
    std::uint32_t render_if_equal, record_count;
    std::uint64_t arguments;
};
static_assert(sizeof(PredicateJob) == 56);
// Argument records per block: 32 bytes each, the instance count in word 1 of
// both the indexed and the non-indexed layout.
constexpr std::uint32_t predicate_capacity = 128;
constexpr std::size_t predicate_record = 32;
static_assert(offsetof(MTLDrawPrimitivesIndirectArguments, instanceCount) == 4 &&
              offsetof(MTLDrawIndexedPrimitivesIndirectArguments, instanceCount) == 4 &&
              sizeof(MTLDrawIndexedPrimitivesIndirectArguments) <= predicate_record);
constexpr const char* predicate_source = R"msl(
#include <metal_stdlib>
using namespace metal;
struct PredicateJob {
    device const ulong* visibility;
    device const ulong* memory[2];
    uint begin[2];
    uint end[2];
    uint render_if_equal;
    uint record_count;
    device uint* arguments;
};
static void evaluate(device const PredicateJob& job) {
    ulong value[2];
    for (uint i = 0; i < 2; ++i) {
        if (job.memory[i] != nullptr) {
            value[i] = *job.memory[i];
        } else {
            ulong sum = 0;
            for (uint slot = job.begin[i]; slot < job.end[i]; ++slot) sum += job.visibility[slot];
            value[i] = sum;
        }
    }
    const bool equal = value[0] == value[1];
    if ((job.render_if_equal != 0) == equal) return;
    for (uint record = 0; record < job.record_count; ++record) job.arguments[record * 8 + 1] = 0;
}
kernel void armigo_nvn_render_predicate(device const PredicateJob& job [[buffer(0)]]) { evaluate(job); }
// The same evaluation as a point draw with rasterization disabled, at the
// start of the render pass that holds the block's draws.
vertex void armigo_nvn_render_predicate_vertex(device const PredicateJob& job [[buffer(0)]]) { evaluate(job); }
)msl";

// Why an open render pass ended. With ARMIGO_NVN_PASS_REASONS=1 the replay
// counts the reasons, with the segment cuts and the conditional-render work,
// and prints them every 300 submissions ("nvn_pass_reasons fence=N cuts=C
// reason=count ... conditionals: gpu=G cpu=C if_equal=E if_not_equal=N
// reports: samples=S other=O").
enum class PassBreak : std::uint8_t {
    render_targets, transfer, clear, dispatch, barrier, wait_sync, fence_sync, counter, depth_needed,
    storage_after_depth_barrier, color_alias, depth_sampled, sample_locations, downsample, discard, skip,
    cpu_drain, finish, predicate, buffer_writes, relaxed_hazard, wait_event, draw_dump, attachment_feedback, count
};
constexpr const char* pass_break_names[] = {
    "render_targets", "transfer", "clear", "dispatch", "barrier", "wait_sync", "fence_sync", "counter",
    "depth_needed", "storage_after_depth_barrier", "color_alias", "depth_sampled", "sample_locations",
    "downsample", "discard", "skip", "cpu_drain", "finish", "predicate", "buffer_writes", "relaxed_hazard",
    "wait_event", "draw_dump", "attachment_feedback"};
static_assert(std::size(pass_break_names) == std::size_t(PassBreak::count));

// What opened a transfer (compute) encoder. With ARMIGO_NVN_PASS_REASONS=1 the
// replay counts each new transfer encoder by the operation that asked for it,
// and how many of them ended an open render pass ("transfers: site=opened/split").
enum class TransferSite : std::uint8_t {
    other, update_uniforms, uniform_window, graphics_writes, uniform_access, storage_bind, dispatch, copy, upload,
    clear_buffer, clear_texture, report, mesh_indirect, predicate, counter, synchronize, cpu_read, finish, count
};
constexpr const char* transfer_site_names[] = {
    "other", "update_uniforms", "uniform_window", "graphics_writes", "uniform_access", "storage_bind", "dispatch",
    "copy", "upload", "clear_buffer", "clear_texture", "report", "mesh_indirect", "predicate", "counter", "synchronize",
    "cpu_read", "finish"};
static_assert(std::size(transfer_site_names) == std::size_t(TransferSite::count));

// ARMIGO_NVN_EVENT_WAIT_DRAIN switches the event-wait shortcuts off, for
// A/B runs of picture faults: "1" or "all" drains the GPU at every event wait
// and releases a held wait only when its writer retires; otherwise a
// comma-separated list of "leading" (a released leading wait drains),
// "own" (a wait this submission's own write meets drains instead of taking a
// barrier), "pending" (every later wait drains, whether or not a writer is
// pending) and "release" (a held wait goes on when its writer retires, not
// when an encoded write meets it).
struct EventWaitDrains {
    bool leading = false, own = false, pending = false, release = false;
};

const EventWaitDrains& event_wait_drains() {
    static const EventWaitDrains drains = [] {
        EventWaitDrains result;
        const char* value = std::getenv("ARMIGO_NVN_EVENT_WAIT_DRAIN");
        if (!value || !*value) return result;
        const std::string_view list{value};
        if (list == "1" || list == "all") return EventWaitDrains{true, true, true, true};
        for (std::size_t begin = 0; begin <= list.size();) {
            const auto end = std::min(list.find(',', begin), list.size());
            const auto name = list.substr(begin, end - begin);
            if (name == "leading") result.leading = true;
            else if (name == "own") result.own = true;
            else if (name == "pending") result.pending = true;
            else if (name == "release") result.release = true;
            else if (!name.empty())
                armigo::diag::write(armigo::diag::Level::warning,
                                    "Armigo NVN: ARMIGO_NVN_EVENT_WAIT_DRAIN names an unknown shortcut\n");
            begin = end + 1;
        }
        return result;
    }();
    return drains;
}

bool pass_reasons_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("ARMIGO_NVN_PASS_REASONS");
        return value && *value == '1';
    }();
    return enabled;
}

// Bound verbose command-order diagnostics by replay time and submission count.
// Submission fences are not presents: consumers must align them with the timeline.
std::uint64_t trace_pass_submission() {
    struct Configuration {
        bool enabled = false;
        std::uint64_t start = 0, delay = 0, remaining = 0, sequence = 0;
        Configuration() {
            const char* flag = std::getenv("ARMIGO_NVN_PASS_TRACE");
            enabled = flag && *flag == '1';
            if (!enabled) return;
            const auto number = [](const char* name, std::uint64_t fallback) {
                const char* value = std::getenv(name);
                std::uint64_t result = fallback;
                if (value) {
                    const auto parsed = std::from_chars(value, value + std::strlen(value), result);
                    if (parsed.ec != std::errc{} || *parsed.ptr) result = fallback;
                }
                return result;
            };
            start = timeline_now_ns();
            delay = std::min(number("ARMIGO_NVN_PASS_TRACE_DELAY_SECONDS", 0), std::uint64_t{86400}) * 1'000'000'000;
            remaining = number("ARMIGO_NVN_PASS_TRACE_SUBMISSIONS", 600);
        }
    };
    static Configuration config;
    if (!config.enabled) return 0;
    static std::mutex mutex;
    const std::lock_guard lock(mutex);
    if (!config.remaining || timeline_now_ns() - config.start < config.delay) return 0;
    --config.remaining;
    return ++config.sequence;
}

struct PassBreakCounts {
    std::array<std::atomic<std::uint64_t>, std::size_t(PassBreak::count)> reasons{};
    std::atomic<std::uint64_t> cuts{0};
    // Conditionals predicated on the GPU or read on the CPU, by mode, and
    // counter reports by kind.
    std::atomic<std::uint64_t> conditional_gpu{0}, conditional_cpu{0}, if_equal{0}, if_not_equal{0};
    std::atomic<std::uint64_t> report_samples{0}, report_other{0};
    std::array<std::atomic<std::uint64_t>, std::size_t(TransferSite::count)> transfers{}, transfer_splits{};
    // Clears drawn in a render encoder of their own, and clears taken as a
    // pass's load action.
    std::atomic<std::uint64_t> clears_own{0}, clears_load{0};
};

PassBreakCounts& pass_break_counts() {
    static auto& counts = *new PassBreakCounts;
    return counts;
}

void note_pass_break(PassBreak why) {
    if (pass_reasons_enabled())
        pass_break_counts().reasons[std::size_t(why)].fetch_add(1, std::memory_order_relaxed);
}

void note_clear(bool own) {
    if (pass_reasons_enabled()) (own ? pass_break_counts().clears_own : pass_break_counts().clears_load)
        .fetch_add(1, std::memory_order_relaxed);
}

void note_transfer_open(TransferSite site, bool split) {
    if (!pass_reasons_enabled()) return;
    auto& counts = pass_break_counts();
    counts.transfers[std::size_t(site)].fetch_add(1, std::memory_order_relaxed);
    if (split) counts.transfer_splits[std::size_t(site)].fetch_add(1, std::memory_order_relaxed);
}

void note_conditional(bool gpu, bool if_equal) {
    if (!pass_reasons_enabled()) return;
    auto& counts = pass_break_counts();
    (gpu ? counts.conditional_gpu : counts.conditional_cpu).fetch_add(1, std::memory_order_relaxed);
    (if_equal ? counts.if_equal : counts.if_not_equal).fetch_add(1, std::memory_order_relaxed);
}

void note_report(bool samples) {
    if (!pass_reasons_enabled()) return;
    auto& counts = pass_break_counts();
    (samples ? counts.report_samples : counts.report_other).fetch_add(1, std::memory_order_relaxed);
}

// A compute stage's name in the dispatch profile: its identity (or entry
// point) and threadgroup size.
std::string profile_label(const platform::CompiledGraphicsStage& shader) {
    const auto& size = shader.compute_local_size;
    return (shader.identity.empty() ? shader.entry_point : shader.identity.substr(0, 16)) + " " +
           std::to_string(size[0]) + "x" + std::to_string(size[1]) + "x" + std::to_string(size[2]);
}

// ARMIGO_NVN_DISPATCH_PROFILE=1 commits every dispatch as a segment of its
// own and reports the GPU time of each compute stage from the segments'
// commit feedback. It serializes the GPU: for attribution runs only.
bool dispatch_profile() {
    static const bool enabled = [] {
        const char* value = std::getenv("ARMIGO_NVN_DISPATCH_PROFILE");
        return value && *value == '1';
    }();
    return enabled;
}

void note_segment_cut() {
    if (pass_reasons_enabled()) pass_break_counts().cuts.fetch_add(1, std::memory_order_relaxed);
}

void report_pass_breaks(std::uint64_t fence) {
    if (!pass_reasons_enabled() || fence % 300) return;
    auto& counts = pass_break_counts();
    std::string line = "nvn_pass_reasons fence=" + std::to_string(fence) +
        " cuts=" + std::to_string(counts.cuts.exchange(0, std::memory_order_relaxed));
    for (std::size_t index = 0; index != counts.reasons.size(); ++index)
        if (const auto value = counts.reasons[index].exchange(0, std::memory_order_relaxed))
            line += std::string(" ") + pass_break_names[index] + "=" + std::to_string(value);
    const auto take = [](std::atomic<std::uint64_t>& value) {
        return std::to_string(value.exchange(0, std::memory_order_relaxed));
    };
    line += " conditionals: gpu=" + take(counts.conditional_gpu) + " cpu=" + take(counts.conditional_cpu) +
        " if_equal=" + take(counts.if_equal) + " if_not_equal=" + take(counts.if_not_equal) +
        " reports: samples=" + take(counts.report_samples) + " other=" + take(counts.report_other);
    line += " clears: own=" + take(counts.clears_own) + " load=" + take(counts.clears_load);
    line += " transfers:";
    for (std::size_t index = 0; index != counts.transfers.size(); ++index)
        if (const auto opened = counts.transfers[index].exchange(0, std::memory_order_relaxed))
            line += std::string(" ") + transfer_site_names[index] + "=" + std::to_string(opened) + "/" +
                std::to_string(counts.transfer_splits[index].exchange(0, std::memory_order_relaxed));
    armigo::diag::opt_in( "%s\n", line.c_str());
}

bool same_targets(const SetRenderTargets& a, const SetRenderTargets& b) {
    const auto attachment = [](const TextureState* x, const TextureState* y,
                               bool xv, bool yv, const TextureViewState& vx,
                               const TextureViewState& vy) {
        return ((!x && !y) || (x && y && x->serial == y->serial)) &&
            xv == yv && (!xv || std::memcmp(&vx, &vy, sizeof vx) == 0);
    };
    if (a.count != b.count || a.count < 0 || a.count > 8) return false;
    for (int i = 0; i < a.count; ++i)
        if (!attachment(a.color[i], b.color[i], a.has_color_view[i],
                        b.has_color_view[i], a.color_views[i], b.color_views[i])) return false;
    return attachment(a.depth, b.depth, a.has_depth_view, b.has_depth_view,
                      a.depth_view, b.depth_view);
}

// A pipeline compiled in the background. The replay walks on while the
// compiler works, so the misses of one submission compile in parallel;
// encoding waits only when it reaches a draw or dispatch whose pipeline is
// not ready.
template <class State>
struct BackgroundCompile {
    std::mutex mutex;
    std::condition_variable done_signal;
    bool done = false;
    State state = nil;
    std::string error;
    State wait() {
        std::unique_lock lock(mutex);
        if (!done_signal.wait_for(lock, std::chrono::seconds(2), [&] { return done; })) {
            armigo::diag::write(armigo::diag::Level::warning,  "nvn_replay: a pipeline compile has taken over 2 s\n");
            done_signal.wait(lock, [&] { return done; });
        }
        return state;
    }
    State wait_silent() {
        std::unique_lock lock(mutex);
        done_signal.wait(lock, [&] { return done; });
        return state;
    }
    // Completion handler body: publishes the result and wakes the waiter.
    void finish(State result, std::string message) {
        std::lock_guard lock(mutex);
        state = result;
        error = std::move(message);
        done = true;
        done_signal.notify_all();
    }
};
using PipelineCompile = BackgroundCompile<id<MTLRenderPipelineState>>;
using ComputeCompile = BackgroundCompile<id<MTLComputePipelineState>>;

struct NativePipeline {
    std::array<std::int32_t, buffer_slots> vertex_slots{};
    // Buffer index feeding the vertex inputs the shader reads but the vertex
    // state does not enable: they read constant zero. -1 when every input is
    // enabled.
    std::int32_t zero_attribute_slot = -1;
    std::shared_ptr<const platform::CompiledGraphicsStage> vertex_stage;
    std::shared_ptr<const platform::CompiledGraphicsStage> fragment_stage;
    // Keep the source stages that the mesh cache key names alive, including
    // when the mesh library is a variant tailored to its fragment inputs.
    std::shared_ptr<const platform::CompiledGraphicsStage> mesh_source_stage, task_stage;
    id<MTLLibrary> task_library = nil;
    id<MTLLibrary> vertex_library = nil;
    id<MTLLibrary> fragment_library = nil;
    // Set once when the pipeline is created and never reassigned: records
    // hold its raw pointer. A failed compile stays failed for this key.
    std::shared_ptr<PipelineCompile> compile;
    // Started by the look-ahead; whether a draw has used it yet.
    bool predicted = false, used = false;
};

struct NativeComputePipeline {
    std::shared_ptr<const platform::CompiledGraphicsStage> stage;
    id<MTLLibrary> library = nil;
    // Set once when the pipeline is created; records hold its raw pointer.
    std::shared_ptr<ComputeCompile> compile;
};

struct CompletionEffect {
    enum class Kind { accumulate, reset, report, signal } kind = Kind::accumulate;
    std::size_t visibility_begin = 0, visibility_end = 0;
    BufferSpan destination{};
    // The retained shared buffer keeps this CPU mapping stable through retirement.
    std::byte* destination_cpu = nullptr;
    id<MTL4CounterHeap> timestamp = nil;
    std::uint64_t cpu_before = 0, gpu_before = 0;
    std::size_t timestamp_index = 0;
    volatile std::uint32_t* event = nullptr;
    std::uint32_t value = 0;
    bool samples = false, add = false;
    // A timestamp report under the segment policy: the GPU start or end of
    // the submission's segment `segment` (0 first), not a heap timestamp.
    bool segment_time = false, segment_start = false;
    std::uint32_t segment = 0;
};

struct UploadArena {
    id<MTLBuffer> buffer = nil;
    std::byte* cpu = nullptr;
    std::uint64_t gpu = 0;
    std::size_t capacity = 0, used = 0;
    std::uint64_t fence = 0;
};

struct UploadSlice {
    id<MTLBuffer> buffer = nil;
    std::byte* cpu = nullptr;
    std::uint64_t gpu = 0;
    std::size_t offset = 0;
    explicit operator bool() const { return buffer != nil; }
};

// Allocations already made resident by one submission. The residency set and
// the retain array both cost a hash insertion (the set also a protocol check)
// per call, and a bind repeats the same pool buffers thousands of times per
// frame, so each allocation is inserted once: a flat open-addressed set keyed by
// the allocation pointer answers the repeat without touching either.
//
// A handful of allocations (the pool buffers) answer most calls, so a small
// direct-mapped filter of the last keys seen answers them from one cache line
// before the set is probed. The set starts at the size the previous submission
// needed instead of growing from 256 by rehashing.
struct ResidentTable {
    std::vector<void*> slots;
    std::size_t count = 0;
    std::array<void*, 64> recent{};

    void reserve(std::size_t expected) {
        if (!slots.empty()) return;
        std::size_t size = 256;
        while (size < expected * 2 + 2) size *= 2;
        slots.assign(size, nullptr);
    }

    bool insert_unique(void* key) {
        void*& hot = recent[(std::uintptr_t(key) >> 4) & (recent.size() - 1)];
        if (hot == key) return false;
        const bool inserted = insert_slow(key);
        hot = key;
        return inserted;
    }

private:
    bool insert_slow(void* key) {
        if (slots.empty()) slots.assign(256, nullptr);
        if ((count + 1) * 2 > slots.size()) {
            std::vector<void*> previous = std::move(slots);
            slots.assign(previous.size() * 2, nullptr);
            count = 0;
            for (void* entry : previous)
                if (entry) insert_slow(entry);
        }
        const std::size_t mask = slots.size() - 1;
        std::size_t index = (std::uintptr_t(key) >> 5) & mask;
        for (;;) {
            if (!slots[index]) {
                slots[index] = key;
                ++count;
                return true;
            }
            if (slots[index] == key) return false;
            index = (index + 1) & mask;
        }
    }
};

struct UniformUpdate {
    std::uint64_t address, size;
    const std::byte* bytes;
    std::uintptr_t cpu = 0;
    std::uint64_t gpu = 0;
};

// Offsets inside the first range, without overflowing either end address.
std::pair<std::uint64_t, std::uint64_t> overlap_offsets(
    std::uint64_t begin, std::uint64_t size, std::uint64_t other, std::uint64_t length) {
    if (!size || !length) return {};
    if (other >= begin) {
        const auto offset = other - begin;
        return offset < size ? std::pair{offset, offset + std::min(length, size - offset)}
                             : std::pair<std::uint64_t, std::uint64_t>{};
    }
    const auto offset = begin - other;
    return offset < length ? std::pair<std::uint64_t, std::uint64_t>{0, std::min(size, length - offset)}
                           : std::pair<std::uint64_t, std::uint64_t>{};
}

UniformUpdate uniform_slice(const UniformUpdate& source, std::uint64_t offset, std::uint64_t size) {
    return {source.address + offset, size, source.bytes + offset,
            source.cpu ? source.cpu + offset : 0, source.gpu ? source.gpu + offset : 0};
}

class EncodeThread;

// One residency set for every submission. An allocation joins it at its first
// use in a submission, and the set is committed only when it changed, before
// the next segment commit; every command buffer uses it. An allocation leaves
// it once every submission that used it retired and no submission used it for
// `window` submissions. A set per submission was made, filled, committed at
// every segment and released again about 26 times a frame.
class SharedResidency {
public:
    static constexpr std::uint64_t window = 256;

    bool ready() const { return set_ != nil; }

    bool create(id<MTLDevice> device) {
        auto* descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = 4096;
        NSError* error = nil;
        set_ = [device newResidencySetWithDescriptor:descriptor error:&error];
        return set_ != nil;
    }

    id<MTLResidencySet> set() const { return set_; }

    // Changes whenever a sweep removes an allocation: a carried resident
    // table from before is no longer proof of residency.
    std::uint64_t removals() const { return removals_.load(std::memory_order_acquire); }

    // `sequence` is the encoding submission's, which is in flight. A
    // submission's allocations arrive in one batch per segment commit, so
    // the lock is taken once for all of them.
    void add(std::span<const __unsafe_unretained id<MTLAllocation>> allocations, std::uint64_t sequence) {
        if (allocations.empty()) return;
        std::lock_guard lock(mutex_);
        for (__unsafe_unretained id<MTLAllocation> allocation : allocations) {
            auto [entry, inserted] = entries_.try_emplace((__bridge void*)allocation);
            entry->second.last = sequence;
            if (!inserted) continue;
            entry->second.allocation = allocation;
            [set_ addAllocation:allocation];
            dirty_ = true;
        }
    }

    void commit() {
        std::lock_guard lock(mutex_);
        if (!dirty_) return;
        [set_ commit];
        dirty_ = false;
    }

    // Removes the allocations last used before `oldest_running`, the
    // sequence of the oldest submission still in flight, and not used for
    // `window` submissions before `newest`.
    void sweep(std::uint64_t oldest_running, std::uint64_t newest) {
        std::lock_guard lock(mutex_);
        for (auto entry = entries_.begin(); entry != entries_.end();) {
            const auto last = entry->second.last;
            if (last < oldest_running && last + window <= newest) {
                [set_ removeAllocation:entry->second.allocation];
                dirty_ = true;
                removals_.fetch_add(1, std::memory_order_release);
                entry = entries_.erase(entry);
            } else {
                ++entry;
            }
        }
    }

private:
    struct Entry {
        id<MTLAllocation> allocation = nil;
        std::uint64_t last = 0;
    };
    std::mutex mutex_;
    id<MTLResidencySet> set_ = nil;
    std::unordered_map<void*, Entry> entries_;
    bool dirty_ = false;
    std::atomic<std::uint64_t> removals_{0};
};

// ARMIGO_NVN_GPUPROF=every:<n> (diagnostic): the GPU time of every encoder the
// replay opens in one frame of every n. A relaxed timestamp is written into a
// counter heap when an encoder begins (after its first active render stage)
// and when it ends (the latest available render-stage completion), so no
// synchronization is added
// beyond the replay's own barriers. Once the frame's last fence completed, the
// log lists each encoder ("nvn_gpuprof") and the frame's totals
// ("nvn_gpuprof_frame"). Encode calls (clears, resolves) make encoders of their
// own: a command-buffer timestamp before and after each such call times it
// as a whole ("kind=call").
class GpuProfile {
public:
    static constexpr std::size_t capacity = 4096;  // the largest timestamp heap Metal makes
    static constexpr std::size_t timestamp_slots = 5;
    static constexpr std::array<MTLRenderStages, 4> end_stages{
        MTLRenderStageVertex, MTLRenderStageObject, MTLRenderStageMesh, MTLRenderStageFragment};

    struct Label {
        char kind = 'R';  // R render pass, C compute / copy, X encode call
        MTLRenderStages begin_stage = 0;
        bool end_recorded = false;
        std::uint32_t width = 0, height = 0, colors = 0;
        bool depth = false;
        std::uint64_t fence = 0, pass = 0;
        std::uint32_t segment = 0;  // the segment of its submission
        std::uint64_t draws = 0, dispatches = 0, fills = 0;
        // The package identity of the program the encoder opened with: the
        // fragment (else mesh, else vertex) stage of a render pass, the
        // compute stage of a compute encoder; and a render pass's mesh (else
        // vertex) stage. Empty when not from a package.
        std::array<char, 72> program{}, geometry{};
    };

    explicit GpuProfile(id<MTLDevice> device) : device_(device) {
        const char* value = std::getenv("ARMIGO_NVN_GPUPROF");
        if (value && !std::strncmp(value, "every:", 6)) every_ = std::strtoull(value + 6, nullptr, 10);
    }

    // Worker only, at each frame boundary.
    void frame_boundary(std::uint64_t last_encoded_fence, std::uint64_t completed) {
        if (!every_) return;
        if (state_ == State::capturing) {
            state_ = State::waiting;
            target_fence_ = last_encoded_fence;
        }
        if (state_ == State::waiting && completed >= target_fence_) report();
        ++frame_;
        if (state_ == State::idle && frame_ % every_ == 0 && heap()) {
            [heap_ invalidateCounterRange:NSMakeRange(0, capacity)];
            labels_.clear();
            dropped_ = 0;
            captured_frame_ = frame_;
            state_ = State::capturing;
        }
    }

    // A slot for an encoder being opened, or -1 when this frame is not
    // profiled or the heap is full.
    int begin(const Label& label) {
        if (state_ != State::capturing) return -1;
        if (timestamp_slots * (labels_.size() + 1) > capacity) {
            ++dropped_;
            return -1;
        }
        labels_.push_back(label);
        return static_cast<int>(labels_.size() - 1);
    }

    void end(int slot, std::uint64_t draws, std::uint64_t dispatches, std::uint64_t fills) {
        if (slot < 0 || std::size_t(slot) >= labels_.size()) return;
        auto& label = labels_[std::size_t(slot)];
        label.end_recorded = true;
        label.draws = draws - label.draws;
        label.dispatches = dispatches - label.dispatches;
        label.fills = fills - label.fills;
    }

    void mark_call_end(int slot) {
        labels_[std::size_t(slot)].end_recorded = true;
    }

    id<MTL4CounterHeap> heap() {
        if (heap_ || failed_) return heap_;
        auto* descriptor = [MTL4CounterHeapDescriptor new];
        descriptor.type = MTL4CounterHeapTypeTimestamp;
        descriptor.count = capacity;
        NSError* error = nil;
        heap_ = [device_ newCounterHeapWithDescriptor:descriptor error:&error];
        if (!heap_) {
            failed_ = true;
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN: ARMIGO_NVN_GPUPROF has no counter heap; profiling is off\n");
        }
        return heap_;
    }

    static void stamp_render(id<MTL4RenderCommandEncoder> encoder, id<MTL4CounterHeap> heap, std::size_t index,
                             bool end, MTLRenderStages begin_stage = MTLRenderStageVertex) {
        if (!end) {
            [encoder writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
                                        afterStage:begin_stage intoHeap:heap atIndex:index];
            return;
        }
        // A stage with no work can leave its timestamp unwritten. Collect
        // every programmable stage and use the latest available completion.
        for (std::size_t stage = 0; stage < end_stages.size(); ++stage)
            [encoder writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
                                        afterStage:end_stages[stage] intoHeap:heap atIndex:index + stage];
    }

    static void stamp_compute(id<MTL4ComputeCommandEncoder> encoder, id<MTL4CounterHeap> heap, std::size_t index) {
        [encoder writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:heap atIndex:index];
    }

private:
    enum class State { idle, capturing, waiting };

    void report() {
        state_ = State::idle;
        const auto count = labels_.size();
        if (!count) return;
        NSData* data = [heap_ resolveCounterRange:NSMakeRange(0, timestamp_slots * count)];
        const auto* stamps = static_cast<const std::uint64_t*>(data.bytes);
        if (!stamps || data.length < timestamp_slots * count * sizeof(std::uint64_t)) return;
        const double tick_ns = 1e9 / double([device_ queryTimestampFrequency]);
        double render_us = 0, compute_us = 0, call_us = 0;
        std::uint64_t first = 0, last = 0;
        std::size_t timed = 0, passes = 0;
        const auto completion = [&](std::size_t index) {
            const auto* first_end = stamps + timestamp_slots * index + 1;
            const auto ends = labels_[index].kind == 'R' ? end_stages.size() : 1;
            return std::max_element(first_end, first_end + ends);
        };
        for (std::size_t index = 0; index < count; ++index) {
            const auto begin = stamps[timestamp_slots * index], end = *completion(index);
            if (!begin || !end || end < begin) continue;
            if (!first || begin < first) first = begin;
            last = std::max(last, end);
        }
        for (std::size_t index = 0; index < count; ++index) {
            const auto begin = stamps[timestamp_slots * index], end = *completion(index);
            const auto& label = labels_[index];
            if (!begin || !end || end < begin) {
                armigo::diag::write(armigo::diag::Level::info,
                    "nvn_gpuprof_untimed frame=%llu index=%zu kind=%c fence=%llu segment=%u "
                    "begin_stage=%llu end_recorded=%u begin_ticks=%llu end_ticks=%llu end_vertex=%llu end_object=%llu "
                    "end_mesh=%llu end_fragment=%llu program=%s geometry=%s\n",
                    static_cast<unsigned long long>(captured_frame_), index, label.kind,
                    static_cast<unsigned long long>(label.fence), label.segment,
                    static_cast<unsigned long long>(label.begin_stage),
                    unsigned(label.end_recorded),
                    static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end),
                    static_cast<unsigned long long>(stamps[timestamp_slots * index + 1]),
                    static_cast<unsigned long long>(stamps[timestamp_slots * index + 2]),
                    static_cast<unsigned long long>(stamps[timestamp_slots * index + 3]),
                    static_cast<unsigned long long>(stamps[timestamp_slots * index + 4]),
                    label.program[0] ? label.program.data() : "-",
                    label.geometry[0] ? label.geometry.data() : "-");
                continue;
            }
            const double us = double(end - begin) * tick_ns / 1e3;
            ++timed;
            if (label.kind == 'R') {
                render_us += us;
                ++passes;
                armigo::diag::write(armigo::diag::Level::info,
                    "nvn_gpuprof frame=%llu index=%zu us=%.1f kind=render start_us=%.1f fence=%llu segment=%u pass=%llu "
                    "size=%ux%u colors=%u depth=%u draws=%llu begin_stage=%llu end_stage=%llu program=%s geometry=%s\n",
                    static_cast<unsigned long long>(captured_frame_), index, us, double(begin - first) * tick_ns / 1e3,
                    static_cast<unsigned long long>(label.fence), label.segment,
                    static_cast<unsigned long long>(label.pass), label.width, label.height, label.colors,
                    unsigned(label.depth), static_cast<unsigned long long>(label.draws),
                    static_cast<unsigned long long>(label.begin_stage),
                    static_cast<unsigned long long>(end_stages[completion(index) -
                        (stamps + timestamp_slots * index + 1)]),
                    label.program[0] ? label.program.data() : "-",
                    label.geometry[0] ? label.geometry.data() : "-");
            } else if (label.kind == 'X') {
                call_us += us;
                armigo::diag::write(armigo::diag::Level::info,
                    "nvn_gpuprof frame=%llu index=%zu us=%.1f kind=call start_us=%.1f fence=%llu segment=%u "
                    "after_pass=%llu\n",
                    static_cast<unsigned long long>(captured_frame_), index, us, double(begin - first) * tick_ns / 1e3,
                    static_cast<unsigned long long>(label.fence), label.segment,
                    static_cast<unsigned long long>(label.pass));
            } else {
                compute_us += us;
                armigo::diag::write(armigo::diag::Level::info,
                    "nvn_gpuprof frame=%llu index=%zu us=%.1f kind=compute start_us=%.1f fence=%llu segment=%u "
                    "after_pass=%llu dispatches=%llu fills=%llu program=%s\n",
                    static_cast<unsigned long long>(captured_frame_), index, us, double(begin - first) * tick_ns / 1e3,
                    static_cast<unsigned long long>(label.fence), label.segment,
                    static_cast<unsigned long long>(label.pass), static_cast<unsigned long long>(label.dispatches),
                    static_cast<unsigned long long>(label.fills), label.program[0] ? label.program.data() : "-");
            }
        }
        armigo::diag::write(armigo::diag::Level::info,
            "nvn_gpuprof_frame frame=%llu render_ms=%.2f compute_ms=%.2f call_ms=%.2f encoders=%zu timed=%zu "
            "render_passes=%zu dropped=%zu span_ms=%.2f\n",
            static_cast<unsigned long long>(captured_frame_), render_us / 1e3, compute_us / 1e3, call_us / 1e3, count + dropped_,
            timed, passes, dropped_,
            last > first ? double(last - first) * tick_ns / 1e6 : 0.0);
    }

    id<MTLDevice> device_ = nil;
    id<MTL4CounterHeap> heap_ = nil;
    bool failed_ = false;
    std::uint64_t every_ = 0, frame_ = 0, captured_frame_ = 0, target_fence_ = 0;
    State state_ = State::idle;
    std::vector<Label> labels_;
    std::size_t dropped_ = 0;
};

// The allocations the parts of one guest submission made resident so far,
// shared by its parts so a later part does not add them again. The parts hold
// it; it keeps the allocations alive. The sweep keeps their entries while any
// part of the submission is in flight (first_sequence), and a sweep that
// removed anything since `removals` makes the table start over.
struct ResidencyCarry {
    std::weak_ptr<const CommandSubmission> submission;
    ResidentTable residents;
    NSMutableArray<id>* retained = [NSMutableArray array];
    std::uint64_t removals = 0, first_sequence = 0;
};

// Native frames own this small result after command/history storage retires.
struct NativeProducerResult {
    std::mutex mutex;
    bool retired = false;
    std::string error;
    std::string check() {
        std::lock_guard lock(mutex);
        return retired ? error : "Native producer fence has not retired";
    }
    void finish(std::string detail) {
        std::lock_guard lock(mutex);
        error = std::move(detail);
        retired = true;
    }
};

struct PendingSubmission : std::enable_shared_from_this<PendingSubmission> {
    std::uint64_t scheduling_period_ns = 0;
    // The guest-memory pool table shaders of this submission read, once a
    // stage needs it (its submission slot's buffer), and the guest addresses
    // a storing stage may have written, as sorted disjoint ranges. Until the
    // learned set has settled that is every pool; once stores are predicted
    // it is the learned pools. guest_writes: a storing stage ran.
    // guest_writes_predicted: this submission was encoded from that set.
    id<MTLBuffer> guest_table = nil;
    std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> guest_ranges;
    bool guest_writes = false;
    bool guest_writes_predicted = false;
    // The two status words task stages of this submission record a launch
    // beyond Metal's mesh grid in (in its upload arena, zeroed), once a task
    // stage needs them.
    const std::uint32_t* task_status = nullptr;
    std::uint64_t task_status_gpu = 0;
    bool preserve_encode_spin = false;
    // The walk indices of the records this submission runs (a part of a
    // guest submission split at an unmet event wait runs only its own).
    std::size_t first_record = 0, end_record = SIZE_MAX;
    // Its submission slot (allocator, query and guest-table storage), held
    // from encoding until retirement, and whether it retired.
    std::size_t slot = 0;
    std::atomic<bool> retired{false};
    // Its place in encoding order: completion effects apply in that order,
    // which differs from fence order once a held submission is encoded after
    // later ones (under fence_mutex_ once set).
    std::uint64_t sequence = 0;
    // This submission evaluates conditional renders from the report table.
    // Writers are stamped with `sequence`, because completion publishes
    // effects in encode order.
    bool replay_conditionals = false;
    // Set when the encode thread replays this submission's Metal encoding;
    // commands is then nil on the worker.
    EncodeThread* encode_thread = nullptr;
    // Objects the encode thread creates for this submission (its command
    // buffers, clear passes). Only the encode thread touches it until the
    // final commit.
    NSMutableArray<id>* encode_retained = [NSMutableArray array];
    // Phases and command ownership are worker-only until segments.sealed.
    struct Phases {
        std::uint64_t preamble = 0, walk = 0, finish = 0, commit = 0, pool = 0;
        std::uint64_t encode_end = 0;
    } phases;
    // Resource GPU-write generation once this submission was encoded: its
    // retirement completes every write noted up to it.
    std::uint64_t resource_write_generation = 0;
    // Only fence_mutex_ protects segment completion and execution_error.
    SubmissionSegments segments;
    std::uint64_t gpu_interval_ns = 0, gpu_first_ns = 0, gpu_last_ns = 0;
    std::uint64_t timed_segments = 0;
    // GPU start and end (host seconds) of each segment by its order in the
    // submission, zero until its feedback arrives; segments_committed is
    // worker-only and numbers the next one.
    std::vector<std::array<double, 2>> segment_times;
    std::uint32_t segments_committed = 0;
    // ARMIGO_NVN_DISPATCH_PROFILE: the stage each segment holds alone (by
    // segment order; empty for other work), worker-only until retirement.
    std::vector<std::string> segment_labels;
    std::uint64_t encode_begin_ns = 0, first_commit_ns = 0, final_commit_ns = 0;
    std::size_t committed_effects = 0;
    bool commands_open = false;
    std::shared_ptr<const CommandSubmission> captured;
    std::vector<UniformUpdate> uniform_updates;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> other_buffer_writes;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> completion_writes;
    WriteRanges completion_ranges;
    // The span of completion_writes, so a read outside it needs no scan.
    std::uintptr_t completion_low = UINTPTR_MAX, completion_high = 0;
    // Per-submission work counters, printed every 60 frames when requested.
    struct Counters {
        std::uint64_t draws = 0, passes = 0, overlay_calls = 0, overlay_bytes = 0;
        std::uint64_t buffer_barriers = 0;
        std::uint64_t metal_binds = 0, pipeline_hashes = 0, pipeline_hits = 0;
        std::uint64_t binds_vertex = 0, binds_buffer = 0, binds_texture = 0, binds_sampler = 0;
        std::uint64_t window_uploads = 0, window_reuses = 0;
        std::uint64_t report_fast = 0, report_ordered = 0;
        std::uint64_t arg_table_rebuilds = 0, uniform_updates = 0;
        std::uint64_t fill_calls = 0, fill_bytes = 0, version_creations = 0;
        // Stage argument encodes and reuses, and what invalidated the stages:
        // the registration or write clock, or a uniform update.
        std::uint64_t stage_binds = 0, stage_reuses = 0, stage_uniform_binds = 0;
        std::uint64_t syncs_registration = 0, syncs_writes = 0, uniform_invalidations = 0;
        // Texture uses that waited for guest-memory stores to be published.
        std::uint64_t guest_waits = 0;
        std::uint64_t dispatches = 0;
    };
    Counters counters;
    UploadArena* upload_arena = nullptr;
    std::uint64_t fence = 0;
    // Host time at commit, including queue latency before GPU execution.
    std::uint64_t commit_time = 0;
    ReplayError execution_error;
    // An encoding problem that dropped some work but let the rest of the
    // submission run (a render pipeline that failed to compile): the effects
    // still apply, and the error is reported without failing later work.
    ReplayError encode_warning;
    id<MTL4CommandAllocator> allocator = nil;
    id<MTL4CommandBuffer> commands = nil;
    SharedResidency* shared_residency = nullptr;
    // The work counters at the last segment commit: the timeline labels each
    // segment with its own draws, passes, dispatches and window uploads.
    std::uint64_t segment_draws = 0, segment_passes = 0, segment_dispatches = 0, segment_uploads = 0;
    // Allocations first used since the last segment commit, which adds them
    // to the shared set; `retained` keeps them alive.
    std::vector<__unsafe_unretained id<MTLAllocation>> resident_adds;
    // Allocations of encode calls (clears), which add to the set they are
    // handed: made at the first such call, used by the command buffers it
    // was handed to (the last one in residency_commands).
    id<MTLResidencySet> residency = nil;
    __unsafe_unretained id<MTL4CommandBuffer> residency_commands = nil;
    id<MTLBuffer> visibility = nil;
    NSMutableArray<id>* retained = [NSMutableArray array];
    std::shared_ptr<ResidencyCarry> carry;
    bool count_residency = false;
    std::unordered_set<void*> command_residency_sets;
    // The timestamp heaps it took from the pool, one per 1024 reports.
    std::vector<id<MTL4CounterHeap>> timestamp_heaps;
    // The sequence of the first part of its guest submission.
    std::uint64_t first_sequence = 0;
    ResidentTable retains;
    std::vector<CompletionEffect> effects;
    std::vector<SyncState*> fenced_syncs;
    std::size_t notified_textures = 0, notified_buffers = 0;
    std::size_t timestamp_count = 0;
    std::vector<TextureState> texture_writes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> buffer_writes;
    // Pool-table versions this submission bound. They stay unmodified until
    // retirement drops the last reference.
    std::vector<std::shared_ptr<PoolTableVersion>> pool_versions;
    // Pass-dump records for this submission. The worker appends them while
    // encoding; retirement publishes them after releasing fence_mutex_.
    std::vector<PassDumpLine> pass_dumps;
};

// ---------- Encode thread ----------
// The replay worker (the translation thread) keeps every decision, every
// object it creates and every CPU write; what it hands on are the Metal calls
// of a submission's command buffers: begin and commit, residency, encoders
// and their state, argument-table entries, draws, dispatches and copies. They
// go as records into a single-producer / single-consumer ring and the encode
// thread replays them in order, so Metal's encoding cost overlaps the
// translation of the next commands. Command buffer order is the record order.
// Fences keep their meaning: the worker numbers segments and fences, the
// encode thread commits them, and retirement waits for completion, which
// follows the commit. Objects a record names stay owned by the submission's
// retained array or a backend cache, except the pass descriptors, commit
// options, copy operands and calls a record retains itself. On by default;
// ARMIGO_NVN_ENCODE_THREAD=0 encodes on the worker.
enum class EncodeOp : std::uint16_t {
    stop = EncodeRing::wrap + 1, begin_commands, end_commands, commit, upload_commit, call,
    render_begin, transfer_begin, end_encoder, encoder_barrier,
    render_pipeline, depth_stencil, stencil_reference, cull, winding, fill, depth_clip,
    depth_bias, blend_color, viewports, scissors, visibility,
    table_address, table_vertex, table_texture, table_sampler,
    draw, draw_indexed, compute_pipeline_compile, compute_table, dispatch, dispatch_indirect,
    copy_buffer, copy_buffer_texture, copy_texture_buffer, copy_texture, timestamp,
    copy_memory, predicate, draw_indirect_args, draw_indexed_indirect_args, render_timestamp, predicate_draw,
    profile_stamp,
    mesh_tables, draw_mesh, residency_set,
};

struct CommitRecord { void* completion; PendingSubmission* owner; std::uint64_t segment, publication; };
struct ResidentRecord { PendingSubmission* owner; void* allocation; };
struct RenderBeginRecord { void* descriptor; void* vertex_table; void* fragment_table; bool relaxed; };
struct BarrierRecord { std::uint64_t after, before; };
struct StencilRecord { std::uint32_t front, back; };
struct TableRecord { void* table; std::uint64_t value, stride; };
struct DrawRecord { std::uint32_t primitive, start, count, instances, base_instance; };
struct MeshTablesRecord { void* object_table; void* mesh_table; };
// A mesh draw: `groups` task (or mesh) threadgroups, or the GPU address of a
// {x, y, z} threadgroup count when `indirect` is nonzero.
struct DrawMeshRecord {
    std::uint32_t groups;
    std::uint32_t object[3], mesh[3];
    std::uint64_t indirect;
};
struct DrawIndexedRecord {
    std::uint32_t primitive, count, type, instances, base_instance;
    std::int32_t base_vertex;
    std::uint64_t address, length;
};
struct DispatchRecord { MTLSize grid, threads; };
struct DispatchIndirectRecord { std::uint64_t address; MTLSize threads; };
struct CopyRecord {
    void* source = nullptr;
    void* destination = nullptr;
    std::uint64_t source_offset = 0, destination_offset = 0, size = 0, row = 0, image = 0;
    std::uint32_t source_slice = 0, source_level = 0, destination_slice = 0, destination_level = 0;
    MTLOrigin source_origin{}, destination_origin{};
    MTLSize extent{};
};
struct TimestampRecord { void* heap; std::uint64_t index; };
struct CopyMemoryRecord { void* destination; const void* source; std::uint64_t size; };
struct PredicateRecord { void* pipeline; void* table; std::uint64_t job; };
struct PredicateDrawRecord { void* pipeline; void* table; std::uint64_t job; void* vertex_table; };
struct DrawIndirectArgsRecord { std::uint32_t primitive; std::uint64_t arguments; };
struct DrawIndexedIndirectArgsRecord {
    std::uint32_t primitive, type;
    std::uint64_t address, length, arguments;
};

// What a queued call encodes into: the current command buffer of its
// submission. A failure detail ends the submission with an execution error.
struct EncodeCallContext {
    id<MTL4CommandBuffer> commands = nil;
    id<MTLResidencySet> residency = nil;
    NSMutableArray<id>* retained = nil;
    std::string error;
};
using EncodeCall = void (^)(EncodeCallContext& context);

class ReplayImpl;

class EncodeThread {
public:
    explicit EncodeThread(ReplayImpl& backend) : backend_(backend) {
        thread_ = std::thread([this] { run(); });
    }
    ~EncodeThread() {
        put(EncodeOp::stop, 0);
        ring_.publish();
        if (thread_.joinable()) thread_.join();
    }
    EncodeThread(const EncodeThread&) = delete;
    EncodeThread& operator=(const EncodeThread&) = delete;

    // Stands for the open encoder on the worker.
    id token() const { return token_; }
    // Records reach the encode thread at the next publish().
    void* reserve(EncodeOp op, std::uint32_t argument, std::size_t payload) {
        return ring_.put(static_cast<std::uint16_t>(op), argument, payload);
    }
    void put(EncodeOp op, std::uint32_t argument) { reserve(op, argument, 0); }
    template <class T> void put(EncodeOp op, std::uint32_t argument, const T& value) {
        std::memcpy(reserve(op, argument, sizeof value), &value, sizeof value);
    }
    void upload_commit(std::function<void()> commit) {
        auto* owned = new std::function<void()>(std::move(commit));
        put(EncodeOp::upload_commit, 0, static_cast<void*>(owned));
        ring_.publish();
    }
    void publish() { ring_.publish(); }
    // Returns once the encode thread has replayed every record put so far.
    void drain() { ring_.drain(); }
    std::uint64_t cpu_ns() const { return cpu_ns_.load(std::memory_order_relaxed); }

private:
    void run();
    bool exec(const EncodeRing::Header& header, const void* payload);

    ReplayImpl& backend_;
    EncodeRing ring_;
    id token_ = [NSObject new];
    std::thread thread_;
    // Encode thread only.
    id<MTL4CommandBuffer> commands_ = nil;
    id encoder_ = nil;
    // The submission of the open command buffer.
    std::shared_ptr<PendingSubmission> owned_;
    PendingSubmission* owner_ = nullptr;
    void* pool_ = nullptr;
    // The last record was a submission's final commit: the next records come
    // with the next submission, so the thread sleeps instead of spinning.
    bool idle_ = true;
    // How long the thread spins for the next record inside a submission. A
    // frame paced below 120 Hz leaves enough time to sleep between bursts. An
    // unpaced frame or a frame at 120 Hz or faster keeps the bounded spin so
    // semaphore wake-ups do not delay its tightly budgeted commits. The thread
    // still sleeps between submissions after the final commit.
    static constexpr std::uint64_t spin_unpaced_ns = 500'000;
    static constexpr std::uint64_t spin_paced_min_period_ns = 1'000'000'000 / 120 + 1;
    std::uint64_t spin_ns_ = spin_unpaced_ns;
    bool spin_reported_ = false;
    // The last pipeline record named a pipeline that failed to compile.
    bool skip_draws_ = false;
    bool skip_dispatches_ = false;
    std::uint64_t cpu_mark_ = 0;
    std::atomic<std::uint64_t> cpu_ns_{0};
};

// Adds an allocation to a submission's residency set once. A texture view
// shares its parent's storage, which must be resident for the GPU to read or
// write the view.
// Retains an object once per submission. A bind of the same sampler keeps
// alive the reference taken by the first one.
static void retain_once(PendingSubmission& pending, id object) {
    if (!object) return;
    if (!pending.retains.insert_unique((__bridge void*)object)) return;
    [pending.retained addObject:object];
}

// The shared set is committed before the segment that uses it, on the thread
// that commits.
static void add_resident(PendingSubmission& pending, id<MTLAllocation> allocation) {
    pending.resident_adds.push_back(allocation);
    [pending.carry->retained addObject:allocation];
}

static void use_residency_set(PendingSubmission& pending, id<MTL4CommandBuffer> commands,
                               id<MTLResidencySet> set) {
    [commands useResidencySet:set];
    if (pending.count_residency) pending.command_residency_sets.insert((__bridge void*)set);
}

// The residency set an encode call adds to, made at the first call and used
// by each command buffer a call encodes into.
static id<MTLResidencySet> call_residency(PendingSubmission& pending, id<MTL4CommandBuffer> commands) {
    if (!commands) return nil;
    if (!pending.residency) {
        auto* descriptor = [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = 8;
        NSError* error = nil;
        pending.residency = [commands.device newResidencySetWithDescriptor:descriptor error:&error];
        if (!pending.residency) return nil;
    }
    if (pending.residency_commands != commands) {
        use_residency_set(pending, commands, pending.residency);
        pending.residency_commands = commands;
    }
    return pending.residency;
}

// Whether an allocation is a texture, cached per class: the Metal objects a
// submission makes resident come from a handful of classes, and the runtime
// protocol walk is far slower than a class compare.
static bool is_texture_allocation(id allocation) {
    struct Entry { Class type = nil; bool texture = false; };
    static thread_local std::array<Entry, 8> classes{};
    const Class type = object_getClass(allocation);
    for (const auto& entry : classes)
        if (entry.type == type) return entry.texture;
    const bool texture = [allocation conformsToProtocol:@protocol(MTLTexture)];
    std::rotate(classes.rbegin(), classes.rbegin() + 1, classes.rend());
    classes.front() = {type, texture};
    return texture;
}

// The texture type declared by a sampled resource. A compatible view or zero
// fallback must be bound in this type.
static std::optional<MTLTextureType> declared_texture_type(
    const platform::CompiledShaderResource& resource) {
    if (!resource.texture_descriptor) return std::nullopt;
    const auto& declared = *resource.texture_descriptor;
    switch (declared.dimension) {
    case platform::CompiledShaderTextureDimension::d1:
        return declared.array ? MTLTextureType1DArray : MTLTextureType1D;
    case platform::CompiledShaderTextureDimension::d2:
        if (declared.multisample)
            return declared.array ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
        return declared.array ? MTLTextureType2DArray : MTLTextureType2D;
    case platform::CompiledShaderTextureDimension::cube:
        return declared.array ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
    case platform::CompiledShaderTextureDimension::d3:
        return MTLTextureType3D;
    case platform::CompiledShaderTextureDimension::buffer:
        return MTLTextureTypeTextureBuffer;
    default:
        return std::nullopt;
    }
}

static void make_resident(PendingSubmission& pending, __unsafe_unretained id<MTLAllocation> allocation) {
    if (!allocation) return;
    if (!pending.carry->residents.insert_unique((__bridge void*)allocation)) return;
    add_resident(pending, allocation);
    if (is_texture_allocation(allocation)) {
        for (id<MTLTexture> parent = ((id<MTLTexture>)allocation).parentTexture; parent;
             parent = parent.parentTexture) {
            if (!pending.carry->residents.insert_unique((__bridge void*)parent)) break;
            add_resident(pending, parent);
        }
    }
}

// Only descriptor inputs belong in the key. Unused attachments, attributes
// and streams remain zero, and every byte is initialized before lookup.
struct DrawPipelineKey {
    const platform::CompiledGraphicsStage* vertex;
    const platform::CompiledGraphicsStage* fragment;
    std::array<std::uint32_t, 8> colors;
    std::array<VertexAttributeStateBits, buffer_slots> attributes;
    std::array<VertexStreamStateBits, buffer_slots> streams;
    std::array<BlendStateBits, 8> blend;
    std::uint32_t blend_enable, channel_mask, samples, alpha_to_coverage;
    std::int32_t primitive;
    std::uint32_t rasterizer_discard;
};

constexpr std::uint32_t compute_stage_bit = 1u << NVN_SHADER_STAGE_COMPUTE;
constexpr std::uint32_t graphics_stage_bits = ((1u << stage_count) - 1) & ~compute_stage_bit;

// The attachment, blend and sample state of a pipeline key under `s`.
void fill_attachment_key(const ReplayState& s, const std::array<MTLPixelFormat, 8>& formats, NSUInteger samples,
                         DrawPipelineKey& key) {
    key.rasterizer_discard = s.rasterizer_discard;
    key.samples = std::max<NSUInteger>(samples, 1);
    key.alpha_to_coverage = s.multisample.alpha_to_coverage;
    key.blend_enable = s.color.blend_enable;
    key.channel_mask = s.channel_mask.mask;
    // A colour component outside an API 231 fragment stage's output map
    // keeps its old value on the guest GPU, but Metal writes every component
    // under the write mask, so the mask drops the unmapped ones. API 55
    // pipelines keep the NVN mask alone: their recorded pipeline lists name
    // it.
    if (key.fragment) {
        const auto api = selected_api_version();
        if (api && api->major == 231) key.channel_mask &= key.fragment->interface.color_output_mask;
    }
    for (std::size_t i = 0; i < key.colors.size(); ++i) {
        key.colors[i] = static_cast<std::uint32_t>(formats[i]);
        if (!key.colors[i]) continue;
        const auto& source = s.blend[i];
        auto& blend = key.blend[i];
        blend.target = source.target;
        blend.source_color = source.source_color;
        blend.destination_color = source.destination_color;
        blend.source_alpha = source.source_alpha;
        blend.destination_alpha = source.destination_alpha;
        blend.color_equation = source.color_equation;
        blend.alpha_equation = source.alpha_equation;
    }
}

// A mesh draw's pipeline: its task (optional), mesh and fragment stages and
// the attachment state. The mesh stage takes the key's vertex slot.
constexpr std::int32_t mesh_draw_primitive = -2;
struct MeshPipelineKey {
    const platform::CompiledGraphicsStage* task;
    DrawPipelineKey draw;
};

const char* fill_mesh_pipeline_key(const ReplayState& s, const std::array<MTLPixelFormat, 8>& formats,
                                   NSUInteger samples, MeshPipelineKey& key) {
    if (s.failed_stages & graphics_stage_bits) return "program stage failed to translate";
    const auto stage_of = [&](std::size_t slot) -> const platform::CompiledGraphicsStage* {
        const auto& program = s.stage_programs[slot];
        return program ? program->stages[slot].get() : nullptr;
    };
    const auto* task = stage_of(NVN_SHADER_STAGE_TESS_CONTROL);
    const auto* mesh = stage_of(NVN_SHADER_STAGE_TESS_EVALUATION);
    if (s.geometry_mode == 0) return "mesh draw requires a mesh geometry processing mode";
    if (!mesh || !mesh->mesh) return "mesh draw has no translated mesh stage";
    if (task && !task->task) return "mesh draw's task stage has no task launch";
    if (mesh->mesh->launched_by_task != (task != nullptr))
        return task ? "mesh draw's mesh stage was translated to run without a task stage"
                    : "mesh draw's mesh stage was translated to run under a task stage";
    // The mesh stage reads its payload words from the task's payload, which
    // the pipeline sizes by the task stage.
    if (task && mesh->mesh->payload_bytes > task->task->payload_bytes)
        return "mesh draw's mesh stage reads more payload than its task stage passes";
    if (s.stage_programs[NVN_SHADER_STAGE_VERTEX] || s.stage_programs[NVN_SHADER_STAGE_GEOMETRY])
        return "mesh draw has a vertex or geometry stage bound";
    if (s.color.logic_op != NVN_LOGIC_OP_COPY || s.color.alpha_test != NVN_ALPHA_FUNC_ALWAYS)
        return "logic operations and fixed-function alpha tests require shader support";
    if (s.multisample.coverage_modulation || s.multisample.coverage_to_color_enable ||
        s.multisample.sample_locations_grid)
        return "programmable multisample coverage state is unsupported";
    std::memset(&key, 0, sizeof key);
    key.task = task;
    key.draw.vertex = mesh;
    key.draw.fragment = stage_of(NVN_SHADER_STAGE_FRAGMENT);
    key.draw.primitive = mesh_draw_primitive;
    fill_attachment_key(s, formats, samples, key.draw);
    return nullptr;
}

// The pipeline key of a draw under `s`, or why the state is not supported.
// Shared by the draw path and the look-ahead, which must agree exactly.
const char* fill_pipeline_key(const ReplayState& s, std::int32_t primitive,
                              const std::array<MTLPixelFormat, 8>& formats, NSUInteger samples,
                              DrawPipelineKey& key) {
    if (s.failed_stages & graphics_stage_bits) return "program stage failed to translate";
    const auto& vertex_program = s.stage_programs[NVN_SHADER_STAGE_VERTEX];
    const auto& fragment_program = s.stage_programs[NVN_SHADER_STAGE_FRAGMENT];
    const auto* vertex_shader = vertex_program
        ? vertex_program->stages[NVN_SHADER_STAGE_VERTEX].get() : nullptr;
    const auto* fragment_shader = fragment_program
        ? fragment_program->stages[NVN_SHADER_STAGE_FRAGMENT].get() : nullptr;
    if (!vertex_shader) return "draw has no bound vertex program";
    for (const auto stage : {NVN_SHADER_STAGE_GEOMETRY, NVN_SHADER_STAGE_TESS_CONTROL,
                             NVN_SHADER_STAGE_TESS_EVALUATION})
        if (s.stage_programs[stage]) return "geometry and tessellation stages require lowering";
    if (s.color.logic_op != NVN_LOGIC_OP_COPY || s.color.alpha_test != NVN_ALPHA_FUNC_ALWAYS)
        return "logic operations and fixed-function alpha tests require shader support";
    if (s.multisample.coverage_modulation || s.multisample.coverage_to_color_enable ||
        s.multisample.sample_locations_grid)
        return "programmable multisample coverage state is unsupported";
    std::memset(&key, 0, sizeof key);
    key.vertex = vertex_shader;
    key.fragment = fragment_shader;
    key.primitive = primitive;
    fill_attachment_key(s, formats, samples, key);
    const auto attribute = [&](std::size_t destination, std::size_t source) {
        if (source >= s.attributes.size() || destination >= key.attributes.size()) return false;
        key.attributes[destination] = s.attributes[source];
        const auto stream = s.attributes[source].stream;
        if (stream < key.streams.size()) key.streams[stream] = s.streams[stream];
        return true;
    };
    // A fetching vertex stage reads the layout from its fetch table, so its
    // pipelines do not depend on it.
    if (vertex_shader->vertex_fetch_buffer_index) return nullptr;
    if (!vertex_shader->vertex_class_inputs.empty()) {
        for (const auto& input : vertex_shader->vertex_class_inputs)
            if (!attribute(input.location, input.location)) return "vertex input exceeds the attribute limit";
    } else {
        for (const auto& input : vertex_shader->vertex_attributes)
            if (!attribute(input.attribute_index, input.generic_slot)) return "vertex input exceeds the attribute limit";
    }
    return nullptr;
}

struct NativeDepthState { id<MTLDepthStencilState> state = nil; };
using DepthStateKey = std::array<std::uint32_t, 6>;
// Vertex stage, fragment stage and depth texture of the draw that opens a pass.
using OmittedDepthKey = std::tuple<const void*, const void*, const void*>;

class ReplayImpl;

struct Encoder {
    ReplayImpl& backend;
    PendingSubmission& pending;
    // The encode thread, or null when Metal calls are made here.
    EncodeThread* queue;
    ReplayState state;
    id<MTL4RenderCommandEncoder> render = nil;
    id<MTL4ComputeCommandEncoder> transfer = nil;
    // The operation that would open the next transfer encoder (diagnostics).
    TransferSite transfer_site = TransferSite::other;
    struct SiteScope {
        Encoder& encoder;
        TransferSite previous;
        SiteScope(Encoder& owner, TransferSite site, bool keep_outer = false)
            : encoder(owner), previous(owner.transfer_site) {
            if (!keep_outer || previous == TransferSite::other) owner.transfer_site = site;
        }
        ~SiteScope() { encoder.transfer_site = previous; }
        SiteScope(const SiteScope&) = delete;
        SiteScope& operator=(const SiteScope&) = delete;
    };
    // Guest ranges the open transfer encoder's operations accessed since its
    // last barrier. An operation that states its guest accesses runs without
    // a barrier when none conflicts with these (a write against any access, a
    // read against a write); any other operation takes the barrier and leaves
    // the encoder untracked until the next one.
    struct GuestAccess { std::uint64_t begin, end; bool write; };
    std::vector<GuestAccess> transfer_accesses;
    bool transfer_untracked = false;
    // The compute stage's guest accesses, gathered by bind_stage. A bound
    // texture over guest bytes (an in-place or buffer texture) leaves them
    // incomplete.
    std::vector<GuestAccess> compute_accesses;
    bool compute_untracked = false;
    id<MTL4ArgumentTable> vertex_table = nil;
    id<MTL4ArgumentTable> fragment_table = nil;
    // A mesh draw's task (object) and mesh stage tables, and whether
    // open_render is preparing a mesh draw.
    id<MTL4ArgumentTable> object_table = nil;
    id<MTL4ArgumentTable> mesh_table = nil;
    bool mesh_draw = false;
    id<MTL4ArgumentTable> compute_table = nil;
    // Game barrier policy: the last encoder was a render pass without storage
    // writes and nothing came after it; a barrier was asked for since the last
    // pass began; the last render pass ordered only later fragment work.
    bool after_render_pass = false, barrier_requested = false, relaxed_producer = false;
    // The open render pass began with the relaxed barrier.
    bool pass_relaxed = false;
    // A storage image a stage writes: dispatches only, through a texture
    // made shader-writable, published as a GPU write of its texture.
    bool note_image_write(std::int32_t texture_id, __strong id<MTLTexture>& texture, bool compute);
    // Copies storage replaced for an image write. A texture that already
    // shares the previous texture's buffer keeps those bytes, with the same
    // barrier around the point the copy would have run.
    bool copy_replaced_image(id<MTLTexture> previous, id<MTLTexture> upgraded);
    // An atomic storage image. Storage that can move to R32 is replaced by
    // prepare_images before any stage writes an argument slot. Replacing
    // it here is a fallback: slots already written keep the old storage. A bind
    // that uses the stand-in records no write of the real texture. `texture`
    // becomes the texture the stage binds.
    void report_type_mismatch(const platform::CompiledGraphicsStage& shader,
                              const platform::CompiledShaderResource& resource, std::int32_t texture_id,
                              id<MTLTexture> texture, MTLTextureType want);
    bool note_image_atomic(std::size_t stage_index, const platform::CompiledShaderResource& resource,
                           std::uint32_t element, std::int32_t texture_id,
                           __strong id<MTLTexture>& texture, bool compute);
    // Replaces atomic-image storage for every stage this draw or dispatch
    // binds, before any of those stages writes an argument slot.
    bool prepare_images(
        std::initializer_list<std::pair<std::size_t, const platform::CompiledGraphicsStage*>> stages);
    // Drops descriptor and argument-slot caches of storage that just moved,
    // and marks every stage's arguments dirty so the next bind writes the new
    // texture.
    void forget_replaced_image(std::uint64_t serial, id<MTLTexture> previous);
    void note_image_atomic_replacement(std::uint64_t serial, id<MTLTexture> previous, bool late);
    [[nodiscard]] std::uint64_t bound_c2_handle(std::size_t stage_index, std::uint64_t offset) const;
    [[nodiscard]] std::int32_t bound_texture_id_at(std::size_t stage_index, std::size_t slot, bool image) const;
    std::uint64_t read_stage_cbuf_word(std::size_t stage_index, std::uint32_t slot, std::uint64_t offset);
    std::uint32_t read_stage_guest_word(std::uint64_t address);
    std::uint64_t read_stage_handle(std::size_t stage_index, const platform::CompiledGraphicsStage& shader,
                                    const platform::CompiledShaderResource& resource, std::uint32_t element,
                                    std::vector<std::optional<std::uint64_t>>& handle_program_words);
    // One warning per stage and slot in this submission.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> image_atomic_warnings;
    std::uint8_t feedback_written = 0, feedback_direct_reads = 0, feedback_draw_writes = 0;
    bool feedback_depth_written = false, feedback_depth_direct_reads = false;
    bool feedback_draw_depth_write = false;
    std::uint64_t feedback_epoch = 0;
    bool feedback_reopen = false, feedback_rebind = false;
    struct FeedbackUse { std::uint64_t original = 0; id<MTLTexture> texture = nil; bool image = false; };
    std::vector<FeedbackUse> feedback_uses;
    id<MTLTexture> feedback_texture(id<MTLTexture> texture, bool image,
                                  std::optional<std::uint32_t> pinned_level = {});
    // Under ARMIGO_NVN_DISPATCH_PROFILE: commits the work encoded so far as a
    // segment holding `label` (empty: other work).
    bool profile_cut(std::string label);
    // The draw being opened does work before the fragment stage that a pass
    // after only a fragment barrier cannot take: a vertex, geometry, task or
    // mesh stage that writes storage or guest memory, or a mesh draw.
    bool pre_fragment_hazard() const;
    // An API 231 compute stage's driver data holds the dispatch's workgroup
    // counts; the last driver data uploaded for the compute stage, which an
    // indirect dispatch copies its counts into.
    std::array<std::uint32_t, 3> dispatch_groups{};
    UploadSlice compute_driver_slice;
    std::int32_t render_primitive = -1;
    MTL4RenderPassDescriptor* render_descriptor = nil;
    struct DeferredClear {
        graphics::ClearPlan plan;
        id<MTLTexture> texture;
        std::uint64_t backing_begin = 0, backing_size = 0;
    };
    std::vector<DeferredClear> deferred_clears;
    bool flush_clears(std::size_t count = std::numeric_limits<std::size_t>::max());
    bool encode_planned_clear(const graphics::ClearPlan& plan, ClearBindings bindings,
                              const char* prefix);
    bool apply_clears(MTL4RenderPassDescriptor* pass);
    struct ArgumentSlot {
        std::uint64_t value = 0;
        std::uint32_t stride = 0;
        bool present = false;
    };
    struct DriverBlock { std::uint64_t generation = 0, gpu = 0; };
    struct WindowUpload {
        std::uint64_t gpu = 0, source = 0, source_size = 0, generation = 0;
        std::uint32_t bytes = 0;
        std::uint8_t zero_padding_bytes = 0;
        std::byte* cpu = nullptr;
        // GPU writes recorded when the snapshot was taken: a later one over
        // the source makes the snapshot stale, as does a later guest-memory
        // stage whose pools hold it.
        std::size_t writes = 0;
        std::uint64_t guest_writes = 0;
    };
    static_assert(sizeof(WindowUpload) == 64);
    // A stage owns its table shadow, reflected read extents and write list.
    // All resolved addresses and uploads live only for this submission.
    struct StageBindings {
        const platform::CompiledGraphicsStage* shader = nullptr;
        std::array<ArgumentSlot, 32> buffers{};
        std::array<ArgumentSlot, texture_slots> textures{};
        std::array<ArgumentSlot, sampler_slots> samplers{};
        // Driver data (slot 0), the bound-texture table (slot 2) and the push
        // block (3).
        std::array<DriverBlock, 4> driver{};
        std::array<std::uint64_t, 4> driver_generation{1, 1, 1, 1};
        std::array<WindowUpload, buffer_slots + 3> windows{};
        std::array<std::uint32_t, buffer_slots + 3> window_extents{};
        std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
        bool writes_buffers = false;
        // Pool-table cache generation last bound for this stage. A newer
        // generation rebinds; an older version is never bound after it.
        std::uint64_t pool_generation = 0;
        // The shader reads hardware constant buffer 16 (push block or slot 13).
        bool reads_push_bank = false;
        std::uint32_t uniform_slots = UINT32_MAX;
        std::size_t uniform_effects = 0, uniform_writes = 0;
        // The words of hardware constant buffers the shader's texture handles
        // and storage descriptors are read from.
        struct ContentRead { std::uint32_t slot, offset, length; };
        std::vector<ContentRead> content_reads;
        // Cached with the resolved bindings, so a selected dump frame can
        // describe stages whose argument tables were reused.
        std::vector<std::string> dump_reads;
        struct DumpTexture { id<MTLTexture> texture; std::uint64_t serial; std::string kind; };
        std::vector<DumpTexture> dump_textures;
        std::array<std::uint64_t, 32> dump_buffer_sizes{};
        // When the stage's handle programs last ran: inline updates, buffer
        // writes and storing guest-memory stages so far in the submission.
        // Any of them may change a word a handle program reads through a
        // pointer, which the content reads above do not cover.
        std::uint64_t handle_updates = 0, handle_writes = 0, handle_stores = 0;
    };
    std::array<StageBindings, stage_count> bindings;
    // Argument-table entries. An encoder reads a table's entries when it
    // encodes a draw or dispatch, so with the encode thread they are records
    // in order with the draws. The encoder's table members hold every table
    // for the whole submission, so the per-draw helpers take tables
    // unretained.
    void table_address(__unsafe_unretained id<MTL4ArgumentTable> table, std::uint32_t index, std::uint64_t address) {
        if (queue) queue->put(EncodeOp::table_address, index, TableRecord{(__bridge void*)table, address, 0});
        else [table setAddress:address atIndex:index];
    }
    void table_vertex(__unsafe_unretained id<MTL4ArgumentTable> table, std::uint32_t index, std::uint64_t address,
                      std::uint32_t stride) {
        if (queue) queue->put(EncodeOp::table_vertex, index, TableRecord{(__bridge void*)table, address, stride});
        else [table setAddress:address attributeStride:stride atIndex:index];
    }
    void table_texture(__unsafe_unretained id<MTL4ArgumentTable> table, std::uint32_t index, std::uint64_t identifier) {
        if (queue) queue->put(EncodeOp::table_texture, index, TableRecord{(__bridge void*)table, identifier, 0});
        else [table setTexture:MTLResourceID{identifier} atIndex:index];
    }
    void table_sampler(__unsafe_unretained id<MTL4ArgumentTable> table, std::uint32_t index, std::uint64_t identifier) {
        if (queue) queue->put(EncodeOp::table_sampler, index, TableRecord{(__bridge void*)table, identifier, 0});
        else [table setSamplerState:MTLResourceID{identifier} atIndex:index];
    }
    // Each returns whether the entry changed. An unchanged GPU address or
    // resource ID names an allocation this submission already made resident:
    // the submission retains it, so no other allocation can take its address.
    bool write_buffer_slot(__unsafe_unretained id<MTL4ArgumentTable> table, std::size_t stage,
                           std::uint32_t index, std::uint64_t address) {
        if (index >= 32 || stage >= stage_count) return true;
        auto &slot = bindings[stage].buffers[index];
        if (slot.present && slot.value == address) return false;
        table_address(table, index, address);
        ++pending.counters.metal_binds;
        ++pending.counters.binds_buffer;
        slot = {address, 0, 1};
        return true;
    }
    bool write_vertex_slot(__unsafe_unretained id<MTL4ArgumentTable> table, std::size_t stage,
                           std::uint32_t index, std::uint64_t address,
                           std::uint32_t stride) {
        if (index >= 32 || stage >= stage_count) return true;
        auto &slot = bindings[stage].buffers[index];
        if (slot.present && slot.value == address && slot.stride == stride) return false;
        table_vertex(table, index, address, stride);
        ++pending.counters.metal_binds;
        ++pending.counters.binds_vertex;
        slot = {address, stride, 1};
        return true;
    }
    bool write_texture_slot(__unsafe_unretained id<MTL4ArgumentTable> table, std::size_t stage,
                            std::uint32_t index, std::uint64_t identifier) {
        if (index >= texture_slots) return true;
        auto &slot = bindings[stage].textures[index];
        if (slot.present && slot.value == identifier) return false;
        table_texture(table, index, identifier);
        ++pending.counters.metal_binds;
        ++pending.counters.binds_texture;
        slot = {identifier, 0, 1};
        return true;
    }
    void write_sampler_slot(__unsafe_unretained id<MTL4ArgumentTable> table, std::size_t stage,
                            std::uint32_t index, std::uint64_t identifier) {
        if (index >= sampler_slots) return;
        auto &slot = bindings[stage].samplers[index];
        if (slot.present && slot.value == identifier) return;
        table_sampler(table, index, identifier);
        ++pending.counters.metal_binds;
        ++pending.counters.binds_sampler;
        slot = {identifier, 0, 1};
    }
    std::string pipeline_error_;
    DrawInvalidation dirty;
    std::int32_t last_pipeline_primitive = -1;
    NativePipeline* last_pipeline = nullptr;
    void invalidate_args(std::size_t stage) {
        if (stage < stage_count) dirty.arguments[stage] = 1;
    }
    // A new uniform range changes resource bindings only when the shader
    // reads a texture handle or storage descriptor from that hardware slot.
    // Other stages retain their conservative full rebuild, including the
    // compute driver's uniform descriptor table.
    void invalidate_uniform_binding(std::size_t stage, std::uint32_t index) {
        bindings[stage].uniform_slots |= 1u << (index + 3);
        if (!uniform_rebind_enabled() ||
            (stage != NVN_SHADER_STAGE_VERTEX && stage != NVN_SHADER_STAGE_FRAGMENT) ||
            std::ranges::any_of(bindings[stage].content_reads, [index](const auto& read) {
                return read.slot == index + 3;
            })) {
            invalidate_args(stage);
        } else {
            dirty.uniforms[stage] = true;
        }
    }
    void invalidate_args_all() {
        dirty.arguments.fill(true);
        for (auto& stage : bindings) stage.uniform_slots = UINT32_MAX;
    }
    // ARMIGO_NVN_UNIFORM_REBIND=0: every update rebinds whole stages, as
    // before uniform-only rebinds. Measurement and bisection only.
    static bool uniform_rebind_enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("ARMIGO_NVN_UNIFORM_REBIND");
            return !(value && *value == '0');
        }();
        return enabled;
    }
    void invalidate_uniforms_all() {
        if (!uniform_rebind_enabled()) return invalidate_args_all();
        for (std::size_t stage = 0; stage < stage_count; ++stage) {
            dirty.uniforms[stage] = true;
            bindings[stage].uniform_slots = UINT32_MAX;
        }
    }
    // A CPU copy into upload memory the GPU reads only after this
    // submission's later records: with the encode thread it runs there, in
    // order with them.
    void copy_memory(void* destination, const void* source, std::size_t size) {
        if (!size) return;
        if (queue) queue->put(EncodeOp::copy_memory, 0, CopyMemoryRecord{destination, source, size});
        else std::memcpy(destination, source, size);
    }
    // Invalidates every stage when a registration or a tracked write moved
    // since the last draw, counting which clock moved.
    bool sync_draw_resources();
    std::array<MTLPixelFormat, 8> attachment_formats{};
    RenderColorAliases color_aliases;
    std::uint8_t attached_colors = 0xff;
    bool pass_depth_only = true;
    // Root textures of the open pass's attachments, taken once when its
    // descriptor is built: a sampled texture is compared against them.
    std::array<__unsafe_unretained id<MTLTexture>, 10> attachment_roots{};
    unsigned attachment_root_count = 0;
    bool depth_omitted = false;
    bool attachmentless_pass = false;
    OmittedDepthKey depth_omitted_key{};
    id<MTLTexture> depth_barrier_allocation = nil;
    bool depth_barrier_reads = false;
    void note_depth_barrier_read(__unsafe_unretained id<MTLTexture> texture);
    void note_attachment_sample(__unsafe_unretained id<MTLTexture> texture);
    // A BC4 target viewed as RGBA16UI renders one texel per compressed block
    // into a block-sized alias, copied from the storage before the pass and
    // back into it after the pass.
    struct BlockAlias {
        id<MTLTexture> alias = nil, storage = nil;
        id<MTLBuffer> staging = nil;
        NSUInteger level = 0, slice = 0, plane = 0, width = 0, height = 0, blocks_x = 0, blocks_y = 0, pitch = 0;
    };
    std::array<BlockAlias, 8> block_aliases{};
    bool block_alias(const nvn::TextureState& state, const nvn::TextureViewState* view, BlockAlias& out);
    bool copy_block_aliases(bool to_alias);
    NSUInteger attachment_width = 0, attachment_height = 0, attachment_samples = 0;
    const NativePipeline* encoded_pipeline = nullptr;
    std::uint32_t fixed_valid = 0;
    struct FixedValues {
        std::array<std::uint32_t, 6> depth{};
        std::array<std::uint32_t, 2> stencil{};
        MTLCullMode cull{};
        std::uint32_t winding{}, fill{};
        bool depth_clamp{};
        std::array<float, 3> bias{};
        std::array<float, 4> blend{};
        std::pair<std::array<MTLViewport, 16>, std::size_t> viewport{};
        std::pair<std::array<MTLScissorRect, 16>, std::size_t> scissor{};
    } fixed;

    template <class T> bool fixed_changed(unsigned index, T& cached, const T& value) {
        const bool changed = !(fixed_valid & (1u << index)) ||
            std::memcmp(&cached, &value, sizeof value) != 0;
        if (changed) {
            cached = value;
            fixed_valid |= 1u << index;
        }
        return changed;
    }
    bool failed = false;
    DrawWriteOrder buffer_write_order;
    ReplayError error;
    std::size_t visibility_cursor = 0;
    std::size_t counter_begin = 0;
    // Samples-passed reports of this submission made after a reset in it, by
    // effect index: their value is the sum of visibility slots [begin, end).
    // The CPU span the pending effects write (reports and event signals), so
    // a read outside it needs no scan; extended as effects are added and
    // reset where they are cleared.
    struct EffectSpan {
        std::size_t counted = 0;
        std::uintptr_t low = UINTPTR_MAX, high = 0;
        WriteRanges writes;
        void clear() {
            counted = 0;
            low = UINTPTR_MAX;
            high = 0;
            writes.clear();
        }
    } effect_span;
    struct GpuCounter { std::size_t effect; std::uint32_t begin, end; };
    std::vector<GpuCounter> gpu_counters;
    bool counter_reset = false;
    std::size_t reset_slot = 0;
    // The open conditional block whose draws the GPU predicates: the CPU
    // address of its argument records (null when no block is open).
    struct PredicateBlock {
        std::byte* arguments = nullptr;
        std::uint64_t arguments_gpu = 0;
        std::uint32_t* record_count = nullptr;
        std::uint32_t used = 0;
        std::int32_t mode = 0;
        std::uint64_t address = 0;
        // The block's predicate job until it runs: first in the block's
        // render pass, or as the kernel before that pass or before transfer
        // work that could change the counters in between. 0 once it ran.
        std::uint64_t job = 0;
    } predicate;
    bool dispatch_predicate(std::uint64_t job);
    id<MTLRenderPipelineState> predicate_draw_pipeline(MTL4RenderPassDescriptor* pass);
    void encode_predicate_draw(id<MTLRenderPipelineState> pipeline);
    bool predicate_open() const { return predicate.arguments != nullptr; }
    bool gpu_predicate(const RenderEnableConditional& command, bool& handled);
    bool read_predicate(std::int32_t mode, std::uint64_t address);
    void log_conditional(std::uint64_t address, bool if_equal, std::uint64_t left, bool left_known,
                         std::uint64_t right, bool right_known, bool drawn);
    // Resolves an open GPU-predicated block on the CPU (a drain), for work
    // the GPU predicate cannot cover.
    bool resolve_predicate() {
        return resolve_carried_predicate() && (!predicate_open() || synchronize_gpu());
    }
    // A block carried over from the previous submission: its counters are in
    // memory once that submission retired. Resolved before the first work the
    // condition affects; a new condition replaces it unresolved.
    bool resolve_carried_predicate();
    std::uint64_t predicate_arguments(const void* record, std::size_t size);
    std::size_t pass_visibility = std::numeric_limits<std::size_t>::max();
    // The open pass was created with the visibility-result buffer: draws count
    // samples into per-draw slots, so counter operations need not end it.
    bool pass_visibility_buffer = false;
    // In-stream uniform updates encoded so far, by aligned guest word: they
    // reach guest memory only when the GPU runs the submission, but handles
    // the shaders read from uniform buffers are resolved while encoding.
    // Only queried words fully supplied by this submission's captures are
    // cached. Ordinary backing and prior submissions still require ordering.
    std::unordered_map<std::uint64_t, std::uint32_t> uniform_words;
    void refresh_uniform_words(std::uint64_t address, std::uint64_t size, const std::byte* bytes);

    Encoder(ReplayImpl& owner, PendingSubmission& submission, const ReplayState& initial, bool canonical,
            nvn::ConditionalReports* report_table = nullptr)
        : backend(owner), pending(submission), queue(submission.encode_thread), state(initial),
          cpu_addressed_buffers(canonical), reports(report_table) {
        dirty.arguments.fill(true);
        refresh_attachment_cpu_bounds();
        trace_submission_id = trace_pass_submission();
        if (trace_submission_id) timeline_event("pass_trace_submission", pending.fence, trace_submission_id);
        trace_pass_event("submission", "begin");
    }

    // Metal calls on the submission's command buffer and its open encoder:
    // direct, or records for the encode thread. With the encode thread,
    // render and transfer hold its token while an encoder is open.
    id<MTL4RenderCommandEncoder> begin_render(MTL4RenderPassDescriptor* pass,
                                              id<MTL4ArgumentTable> vertex,
                                              id<MTL4ArgumentTable> fragment);
    id<MTL4ComputeCommandEncoder> begin_transfer();
    void end_encoder(id encoder, MTLStages after, bool relaxed = false);
    void encoder_barrier(id encoder, MTLStages after, MTLStages before,
                         MTL4VisibilityOptions extra_visibility = MTL4VisibilityOptionNone);
    bool set_render_pipeline(const NativePipeline& pipeline);
    void set_depth_stencil(id<MTLDepthStencilState> depth_state);
    void set_stencil_reference(std::uint32_t front, std::uint32_t back);
    void set_cull(MTLCullMode mode);
    void set_winding(MTLWinding winding);
    void set_fill(MTLTriangleFillMode mode);
    void set_depth_clip(MTLDepthClipMode mode);
    void set_depth_bias(float units, float slope, float clamp);
    void set_blend_color(const std::array<float, 4>& color);
    void set_viewports(const MTLViewport* viewports, std::size_t count);
    void set_scissors(const MTLScissorRect* scissors, std::size_t count);
    void set_visibility(MTLVisibilityResultMode mode, std::size_t offset);
    void encode_draw(MTLPrimitiveType primitive, std::uint32_t start, std::uint32_t count,
                     std::uint32_t instances, std::uint32_t base_instance);
    void encode_draw_indexed(MTLPrimitiveType primitive, std::uint32_t count, MTLIndexType type,
                             std::uint64_t address, std::uint64_t length, std::uint32_t instances,
                             std::int32_t base_vertex, std::uint32_t base_instance);
    bool set_compute_pipeline(const NativeComputePipeline& pipeline);
    void set_compute_table(id<MTL4ArgumentTable> table);
    void encode_dispatch(MTLSize grid, MTLSize threads);
    void encode_dispatch_indirect(std::uint64_t address, MTLSize threads);
    void copy_buffer_range(id<MTLBuffer> source, std::uint64_t source_offset,
                           id<MTLBuffer> destination, std::uint64_t destination_offset,
                           std::uint64_t size);
    void copy_into_texture(id<MTLBuffer> source, std::uint64_t offset, std::uint64_t row,
                           std::uint64_t image, MTLSize extent, id<MTLTexture> destination,
                           std::uint32_t slice, std::uint32_t level, MTLOrigin origin);
    void copy_from_texture(id<MTLTexture> source, std::uint32_t slice, std::uint32_t level,
                           MTLOrigin origin, MTLSize extent, id<MTLBuffer> destination,
                           std::uint64_t offset, std::uint64_t row, std::uint64_t image);
    void copy_texture_range(id<MTLTexture> source, std::uint32_t source_slice,
                            std::uint32_t source_level, MTLOrigin source_origin, MTLSize extent,
                            id<MTLTexture> destination, std::uint32_t destination_slice,
                            std::uint32_t destination_level, MTLOrigin destination_origin);
    void write_timestamp(id<MTL4CounterHeap> heap, std::size_t index);
    void write_render_timestamp(id<MTL4CounterHeap> heap, std::size_t index);
    // Runs call on the command buffer: now (a failure fails with code), or on
    // the encode thread in order (a failure is an execution error).
    bool encode_call(EncodeCall call, ReplayErrorCode code);

    bool visit(const CommandRecordView& record);
    bool visit_command(const CommandRecordView& record);
    bool fail(ReplayErrorCode code, std::string detail);
    struct MeshBindDiagnostic {
        std::size_t ordinal = 0;
        std::string identity;
    };
    struct MeshBindHistory {
        std::array<MeshBindDiagnostic, 8> binds;
        std::size_t total = 0;
    };
    std::optional<MeshBindHistory> mesh_bind_history;
    std::string mesh_stage_identity() const;
    void remember_mesh_bind(const BindProgram& command);
    void diagnose_mesh_failure(const DrawMeshTasks& command);
    bool program_capture_enabled = program_capture::enabled();
    std::unique_ptr<program_capture::Event> begin_program_capture(const CommandRecordView& record) const noexcept;
    void finish_program_capture(std::unique_ptr<program_capture::Event> event, const CommandRecordView& record,
                                bool accepted, std::uint64_t previous_draws) const noexcept;
    Op current_op_ = Op::end;
    std::uint64_t trace_submission_id = 0;
    // ARMIGO_NVN_GPUPROF slots of the open render and transfer encoders.
    int profile_render = -1, profile_transfer = -1;
    int profile_begin(char kind, MTL4RenderPassDescriptor* pass, id encoder);
    void profile_end(int& slot, char kind, id encoder);
    // Before an encode call (slot -1): labels it and stamps its start,
    // returning its slot; after it (its slot): stamps its end.
    int profile_call(int slot);
    std::uint64_t trace_encoder_sequence = 0, trace_active_encoder = 0;
    std::uint64_t trace_conditional = 0;
    std::size_t finalized_visibility = 0;
    std::size_t pass_buffer_writes_begin = 0;
    bool pass_storage_writes = false;
    // Draws encoded before the open render pass, so its dump can report how
    // many draws that pass itself encoded.
    std::uint64_t pass_draws_at_open = 0;
    struct ComputeDumpImage {
        std::uint32_t slot = 0;
        std::uint64_t serial = 0;
        id<MTLTexture> texture = nil;
    };
    // Storage images the open compute dispatch writes, recorded only while a
    // pass dump of this frame is enabled.
    std::vector<ComputeDumpImage> compute_dump_images;
    std::set<std::string> render_dump_reads;
    std::vector<std::string> render_dump_draws;
    std::vector<StageBindings::DumpTexture> render_dump_textures;
    std::vector<PassDumpJob> render_dump_buffers;
    void trace_pass_event(const char* event, const char* reason,
                          std::uint64_t a = 0, std::uint64_t b = 0,
                          std::uint64_t c = 0, std::uint64_t d = 0) const;
    bool close_render(PassBreak why, bool keep_clears = false);
    bool close_transfer();
    // Samples the attachments a pass just stored, or the storage images a
    // dispatch just wrote. Both no-op unless the pass dump selected this frame.
    std::vector<std::string> dump_binding_metadata(std::size_t stage) const;
    void dump_render_pass(PassBreak why);
    void dump_compute_dispatch();
    bool barrier(PassBreak why);
    bool synchronize_gpu();
    bool synchronize_gpu_commit();
    bool restart_segment();
    bool cut_submission();
    void drain_failed_prefix();
    std::uint64_t cut_draws = 0;
    bool order_buffer_writes();
    struct UniformVersion {
        std::uint64_t begin, end, valid_end, dirty_begin, dirty_end;
        UploadSlice slice;
        std::uintptr_t source_cpu = 0;
        std::uint64_t source_gpu = 0;
    };
    // Prior update bytes remain owned until their submission fence retires.
    std::vector<std::shared_ptr<PendingSubmission>> previous_submissions;
    std::uint64_t retired_fence = 0;
    // The newest of those that ran a guest-memory stage that stores.
    std::uint64_t prior_guest_writer_fence = 0;
    // The submission's leading event waits that the scheduler saw met while
    // it held the submission: they are passed without reading the event again.
    std::size_t passed_leading_waits = 0;
    // Queued draws share their immutable submission's descriptor boundary.
    std::uint64_t draw_sampler_registrations() const {
        // Handles capture descriptor indices, not descriptor contents. Freeze
        // the submit-visible population before asynchronous replay can lag
        // behind another registration or another use of the command handle.
        return pending.captured ? submission_sampler_registrations(*pending.captured)
            : state.sampler_registrations ? state.sampler_registrations : UINT64_MAX;
    }
    bool order_cpu_read(std::uint64_t address, std::uint64_t size, bool overlay,
                        const std::byte* cpu, bool current = true);
    bool current_buffer_write_overlap(std::uintptr_t cpu, std::uint64_t size);
    WriteRanges current_cpu_writes;
    std::size_t cpu_writes_indexed = 0, cpu_writes_completed = 0;
    // A storing stage ran since the last GPU wait. Until the next one, an
    // address in the submission's guest-write ranges may have been written.
    bool guest_writes_pending = false;
    // Guest-memory stages bound in this submission.
    std::uint64_t guest_write_serial = 0;
    void note_guest_writes();
    bool guest_written(std::uint64_t address, std::uint64_t size) const;
    std::uint64_t cpu_writes_registration = 0;
    bool cpu_writes_unknown = false;
    // Preserve command order even when addresses repeat: a later outer update
    // supersedes older inner writes, while a shorter rewrite preserves the tail
    // of an earlier larger update. Address-only replacement loses both rules.
    std::vector<UniformUpdate> uniform_updates;
    // Moves on every recorded update to invalidate cached constant windows.
    std::uint64_t update_generation = 0;
    void record_uniform_update(std::uint64_t address, std::uint64_t size,
                               const std::byte* bytes, const BufferSpan& destination);
    WriteRanges captured_guest_bytes, captured_cpu_words, captured_gpu_words;
    void discard_uniform_captures(std::uintptr_t cpu, std::uint64_t gpu, std::uint64_t size);
    void discard_gpu_uniform_captures(std::uint64_t address, std::uint64_t size);
    bool uniform_cpu_conflict(std::uintptr_t cpu, std::uint64_t size) const;
    void rebuild_capture_ranges();
    bool capture_retirement_needed = false;
    bool cpu_addressed_buffers = false;
    std::uintptr_t uniform_cpu_low = UINTPTR_MAX, uniform_cpu_high = 0;
    bool retire_captures_if_full();
    bool uniform_capture_alias(std::uint64_t address, std::uint64_t size, const BufferSpan& span) const;
    bool prepare_graphics_uniform_writes();
    std::uintptr_t attachment_cpu_low = UINTPTR_MAX, attachment_cpu_high = 0;
    void refresh_attachment_cpu_bounds() {
        attachment_cpu_low = UINTPTR_MAX;
        attachment_cpu_high = 0;
        const auto include = [&](const TextureState* target) {
            if (!target || !target->memory || !target->size) return;
            const auto begin = reinterpret_cast<std::uintptr_t>(target->memory);
            if (target->size > UINTPTR_MAX - begin) {
                attachment_cpu_low = 0;
                attachment_cpu_high = UINTPTR_MAX;
                return;
            }
            attachment_cpu_low = std::min(attachment_cpu_low, begin);
            attachment_cpu_high = std::max(attachment_cpu_high, begin + std::uintptr_t(target->size));
        };
        for (const auto* target : state.targets.color) include(target);
        include(state.targets.depth);
    }
    // Guest pages any recorded or unretired update writes: an overlay over
    // bytes no update reached has nothing to apply and skips the scan.
    nvn::FlatMap64<bool> update_pages;
    void note_update_pages(std::uint64_t address, std::uint64_t size) {
        if (!size) return;
        for (auto page = address >> 12, last = (address + size - 1) >> 12; page <= last; ++page)
            update_pages.emplace(page, true);
    }
    bool updates_touch(std::uint64_t address, std::uint64_t size) const {
        if (!size || !update_pages.size()) return false;
        for (auto page = address >> 12, last = (address + size - 1) >> 12; page <= last; ++page)
            if (update_pages.find(page)) return true;
        return false;
    }
    std::vector<UniformVersion> uniform_versions;
    // GPU waits of this submission so far; each writes the listed uniform
    // versions back to guest memory and ends them.
    std::uint64_t gpu_synchronizations = 0;
    // This log only appends during an encoder's lifetime. Keep it for suffix
    // queries, while the range union answers ordinary full-history overlaps.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> other_buffer_writes;
    WriteRanges buffer_write_ranges;
    std::size_t buffer_writes_indexed = 0;
    bool buffer_writes_overlap(std::uint64_t address, std::uint64_t size, std::size_t first = 0);
    void update_effect_ranges();
    bool fill_uniform(UniformVersion& version, std::uint64_t end);
    bool version_uniform(std::uint64_t address, std::uint64_t size, const std::byte* bytes,
                         const BufferSpan& destination);
    bool flush_uniforms();
    bool uniform_conflict(std::uint64_t address, std::uint64_t size);
    struct PhysicalMapping { std::uintptr_t cpu; std::uint64_t gpu, size; };
    // Sorted disjoint CPU intervals. GPU zero means multiple virtual aliases
    // have written the interval; every subsequent mapping then needs visibility.
    std::vector<PhysicalMapping> uniform_native_mappings;
    static void record_alias_mapping(std::vector<PhysicalMapping>& mappings,
                                     std::uintptr_t cpu, std::uint64_t address, std::uint64_t size);
    static bool mapping_alias(const std::vector<PhysicalMapping>& mappings,
                              std::uintptr_t cpu, std::uint64_t address, std::uint64_t size);
    // Actual pending copies retain their guest-to-CPU identity independently
    // of captured overlays. A drain retires both the writes and this index.
    std::vector<PhysicalMapping> uniform_copy_mappings;
    std::size_t uniform_copy_count = 0;
    bool uniform_copy_unknown = false;
    void record_uniform_copy(std::uint64_t address, std::uint64_t size, const BufferSpan& destination);
    bool needs_alias_visibility(std::uintptr_t cpu, std::uint64_t gpu, std::uint64_t size) const;
    bool prepare_uniform_access(std::uint64_t address, std::uint64_t size, std::uint64_t second = 0);
    std::uint64_t uniform_address(const BoundBuffer& binding, std::uint64_t gpu,
                                  std::uint64_t extent);
    void overlay_uniform(std::uint64_t address, std::byte* bytes, std::size_t size,
                         std::uint32_t* covered = nullptr, bool current_only = false) const;
    bool enqueue_counter(const ReportCounter& command, const BufferSpan& destination, std::byte* destination_cpu);
    // Replay policy only. A timestamp counter is 0, samples-passed stays
    // unknown until its fence completes, and any other kind stays unknown.
    void note_report_writes(std::size_t stage);
    void note_report_counter(std::int32_t type, std::uint64_t address) {
        if (!reports) return;
        using Kind = nvn::ConditionalReports::Kind;
        auto kind = Kind::other;
        if (type == NVN_COUNTER_TYPE_TIMESTAMP || type == NVN_COUNTER_TYPE_TIMESTAMP_TOP)
            kind = Kind::timestamp;
        else if (type == NVN_COUNTER_TYPE_SAMPLES_PASSED)
            kind = Kind::samples_passed;
        reports->note_report(address, kind);
    }
    nvn::ConditionalReports* reports = nullptr;
    bool order_completion_read(const BufferSpan& span, std::uint64_t size);
    bool order_completion_read(id<MTLBuffer> __unsafe_unretained buffer, std::uint64_t offset,
                               const std::byte* cpu, std::uint64_t size,
                               const char* wait_tag = nullptr, bool* ordered = nullptr);
    // Resolved sampled textures are owned by this walk. Rebinding an unchanged
    // ID avoids the resource-table lock; a registration of that ID or a change
    // to its texture's entry (its stamp) or tracked writes force resolution
    // again. Resources that give no stamp use the registration generation. A
    // bounded table drops its references with the encoder.
    struct ResolvedTexture {
        std::int32_t texture_id = -1;
        std::uint64_t registration = 0, writes = 0;
        ReplayResources::DescriptorStamp stamp;
        id<MTLTexture> texture = nil;
    };
    std::array<ResolvedTexture, 256> resolved_textures;
    struct ReportPage {
        std::uint64_t page = UINT64_MAX, generation = 0;
        bool aliases = true;
    };
    std::array<ReportPage, 64> report_pages{};
    bool report_destination_isolated(std::uint64_t address, std::uint64_t size);
    bool order_texture_backing(std::uint64_t begin, std::uint64_t size, bool& ordered);
    id<MTLTexture> texture_for_use(const TextureState& state, const TextureViewState* view);
    id<MTLTexture> sampled_texture(std::int32_t texture_id);
    BufferSpan buffer_at(std::uint64_t address, std::uint64_t size);
    BufferAddress buffer_address(std::uint64_t address, std::uint64_t size);
    bool open_transfer() { return open_transfer_with({}, false); }
    bool open_transfer_with(std::span<const GuestAccess> accesses, bool declared);
    void note_compute_texture(__unsafe_unretained id<MTLTexture> texture);
    bool open_render(std::int32_t primitive);
    bool bind_stage(std::size_t stage, const platform::CompiledGraphicsStage& shader,
                    __unsafe_unretained id<MTL4ArgumentTable> table, bool vertex,
                    bool uniforms_only = false);
    bool stage_has_pool(const platform::CompiledGraphicsStage* shader) const;
    bool refresh_pools(std::size_t stage, const platform::CompiledGraphicsStage& shader);
    bool bind_pool_resource(std::size_t stage, const platform::CompiledShaderResource& resource,
                            __unsafe_unretained id<MTL4ArgumentTable> table, bool fragment);
    void retain_pool_version(const std::shared_ptr<PoolTableVersion>& version);
    // Versions whose residency set is already on the open command buffer.
    // A new segment starts this empty; the strong refs live on the submission.
    std::vector<const PoolTableVersion*> pool_residency_attached;
    void invalidate_uniforms(std::uint64_t address, std::uint64_t size, std::uintptr_t cpu);
    bool bind_draw_support(std::uint32_t base_instance);
    bool bind_vertex_fetch(const platform::CompiledGraphicsStage& vertex);
    UploadSlice upload(std::size_t size, const void* data = nullptr);
    // Hardware constant buffers as the shaders address them: 0 is driver
    // data, 1 the stage's own constants, 2 the bound-texture handle table and
    // 3 + n the NVN uniform buffer slot n.
    BoundBuffer hardware_buffer(std::size_t stage, std::uint32_t slot) const;
    std::array<std::byte, compute_driver_data_bytes> driver_constants(std::size_t stage, std::uint32_t slot) const;
    // The bound API 231 compute program's driver data layout, if any.
    const nvn::ComputeDriverLayout* compute_layout() const {
        const auto& program = state.stage_programs[NVN_SHADER_STAGE_COMPUTE];
        return program && program->compute_layout ? &*program->compute_layout : nullptr;
    }
    // An API 231 compute program needs its driver data layout.
    bool check_compute_layout() {
        const auto api = selected_api_version();
        if (!api || api->major != 231 || compute_layout()) return true;
        return fail(ReplayErrorCode::unsupported, "API 231 compute program has no driver data layout");
    }
    // Starts every change of what an API 231 compute stage's driver data holds.
    void invalidate_compute_driver() {
        ++bindings[NVN_SHADER_STAGE_COMPUTE].driver_generation[0];
        bindings[NVN_SHADER_STAGE_COMPUTE].windows[0] = {};
    }
    // The base instance of the draw being encoded, which graphics driver data
    // holds at graphics_base_instance_offset. A change rebinds the stages
    // whose driver data reads reach that word.
    std::uint32_t draw_base_instance = 0;
    void note_base_instance(std::uint32_t base_instance);
    bool draw(const Draw& draw);
    // DrawMeshTasks(first, count), or its indirect form reading {count, first}
    // from GPU memory at `indirect`.
    bool draw_mesh(std::uint32_t first, std::uint32_t count, std::uint64_t indirect);
    // Threadgroup counts copied from indirect mesh records, by record address:
    // valid until the replay reaches record mesh_groups_end, the next one that
    // may write a record.
    std::unordered_map<std::uint64_t, std::uint64_t> mesh_groups;
    std::size_t mesh_groups_end = 0;
    bool take_mesh_records(std::uint64_t indirect);
    // Records visited so far, and whether the look-ahead already ran for this
    // submission.
    std::size_t records_seen = 0;
    bool pipelines_predicted = false;
    void predict_pipelines();
    bool draw_indexed(const DrawIndexed& draw);
    bool draw_indirect(const DrawIndirect& draw);
    bool draw_indexed_indirect(const DrawIndexedIndirect& draw);
    bool dispatch(const Dispatch& dispatch);
    bool dispatch_indirect(const DispatchIndirect& dispatch);
    bool copy_buffer(const CopyBuffer& copy);
    bool copy_buffer_texture(const CopyBufferToTexture& copy, bool upload);
    bool copy_buffer_to_texture(const CopyBufferToTexture& copy);
    bool copy_texture_to_buffer(const CopyBufferToTexture& copy);
    bool copy_texture_to_texture(const CopyTextureToTexture& copy);
    bool scale_texture_copy(const CopyTextureToTexture& copy, id<MTLTexture> source,
                            id<MTLTexture> destination);
    bool clear_bound_color(const ClearColor& clear);
    bool clear_bound_depth(const ClearDepthStencil& clear);
    bool clear_texture(const ClearTexture& clear);
    bool encode_clear_plan(const TextureState& texture, const TextureViewState* view,
                           const NVNcopyRegion& region, std::uint8_t color_mask,
                           const std::array<std::uint32_t, 4>& color, bool depth,
                           float depth_value, bool stencil, std::uint32_t stencil_value,
                           std::uint32_t stencil_mask);
    void note_draw_writes();
    void note_writes();
    void note_shader_writes(std::size_t stage);
    // A GPU write to a texture that shares the guest's bytes (linear,
    // buffer-backed storage) writes guest memory: it is also recorded as a
    // buffer write, so constant windows copied from that memory are ordered
    // after it.
    void note_texture_write(const TextureState& texture);
    bool finish();
};

bool Encoder::pre_fragment_hazard() const {
    if (mesh_draw) return true;
    for (const auto stage : {NVN_SHADER_STAGE_VERTEX, NVN_SHADER_STAGE_GEOMETRY, NVN_SHADER_STAGE_TESS_CONTROL,
                             NVN_SHADER_STAGE_TESS_EVALUATION}) {
        const auto& program = state.stage_programs[stage];
        const auto* shader = program ? program->stages[stage].get() : nullptr;
        if (shader && ((shader->guest_memory && shader->guest_memory->writes) ||
                       std::ranges::any_of(shader->storage_buffers, [](const auto& binding) {
                return binding.access != platform::CompiledShaderResourceAccess::Read;
            })))
            return true;
    }
    return false;
}

id<MTL4RenderCommandEncoder> Encoder::begin_render(MTL4RenderPassDescriptor* pass,
                                                   id<MTL4ArgumentTable> vertex,
                                                   id<MTL4ArgumentTable> fragment) {
    pass_buffer_writes_begin = other_buffer_writes.size();
    pass_storage_writes = false;
    if (trace_submission_id) {
        trace_active_encoder = ++trace_encoder_sequence;
        trace_pass_event("begin", current_op_ == Op::downsample ? "resolve" : "draw",
                         pass.colorAttachments[0].texture.width, pass.colorAttachments[0].texture.height,
                         pass.depthAttachment.texture != nil, pass.stencilAttachment.texture != nil);
    }
    // Under the game policy a pass that directly follows a render pass waits
    // only for its fragment work, so this pass's vertex work overlaps it.
    const bool relaxed = after_render_pass && !barrier_requested && game_barriers() && !pre_fragment_hazard();
    after_render_pass = barrier_requested = relaxed_producer = false;
    pass_relaxed = relaxed;
    if (queue) {
        // The worker keeps changing its descriptor after the pass begins.
        queue->put(EncodeOp::render_begin, 0,
                   RenderBeginRecord{(__bridge_retained void*)[pass copy], (__bridge void*)vertex,
                                     (__bridge void*)fragment, relaxed});
        profile_render = profile_begin('R', pass, nil);
        queue->publish();
        return (id<MTL4RenderCommandEncoder>)queue->token();
    }
    id<MTL4RenderCommandEncoder> encoder = [pending.commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) return nil;
    [encoder barrierAfterQueueStages:relaxed ? MTLStageFragment : MTLStageAll
                        beforeStages:relaxed ? MTLStageFragment : MTLStageAll
                   visibilityOptions:MTL4VisibilityOptionDevice];
    if (vertex) [encoder setArgumentTable:vertex atStages:MTLRenderStageVertex];
    if (fragment) [encoder setArgumentTable:fragment atStages:MTLRenderStageFragment];
    profile_render = profile_begin('R', pass, encoder);
    return encoder;
}

id<MTL4ComputeCommandEncoder> Encoder::begin_transfer() {
    after_render_pass = relaxed_producer = false;
    if (queue) {
        queue->put(EncodeOp::transfer_begin, 0);
        profile_transfer = profile_begin('C', nil, nil);
        queue->publish();
        return (id<MTL4ComputeCommandEncoder>)queue->token();
    }
    id<MTL4ComputeCommandEncoder> encoder = [pending.commands computeCommandEncoder];
    if (!encoder) return nil;
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit | MTLStageDispatch
                   visibilityOptions:MTL4VisibilityOptionDevice];
    profile_transfer = profile_begin('C', nil, encoder);
    return encoder;
}

// A relaxed end orders only later fragment work after this encoder; every
// encoder but a relaxed render pass orders itself after all earlier work.
void Encoder::end_encoder(id encoder, MTLStages after, bool relaxed) {
    if (queue) {
        queue->put(EncodeOp::end_encoder, relaxed, std::uint64_t(after));
        queue->publish();
        return;
    }
    [(id<MTL4CommandEncoder>)encoder barrierAfterStages:after
                                     beforeQueueStages:relaxed ? MTLStageFragment : MTLStageAll
                                       visibilityOptions:MTL4VisibilityOptionDevice];
    [(id<MTL4CommandEncoder>)encoder endEncoding];
}

void Encoder::encoder_barrier(id encoder, MTLStages after, MTLStages before,
                               MTL4VisibilityOptions extra_visibility) {
    if (queue) queue->put(EncodeOp::encoder_barrier, std::uint32_t(extra_visibility), BarrierRecord{after, before});
    else [(id<MTL4CommandEncoder>)encoder barrierAfterEncoderStages:after beforeEncoderStages:before
                                                   visibilityOptions:MTL4VisibilityOptionDevice | extra_visibility];
}

bool Encoder::set_render_pipeline(const NativePipeline& pipeline) {
    // The encode thread waits for the compile when it reaches this record.
    if (queue) {
        queue->put(EncodeOp::render_pipeline, 0, static_cast<const void*>(pipeline.compile.get()));
        return true;
    }
    id<MTLRenderPipelineState> state = pipeline.compile->wait();
    if (!state) return fail(ReplayErrorCode::unsupported, pipeline.compile->error);
    [render setRenderPipelineState:state];
    return true;
}

void Encoder::set_depth_stencil(id<MTLDepthStencilState> depth_state) {
    if (queue) queue->put(EncodeOp::depth_stencil, 0, (__bridge void*)depth_state);
    else [render setDepthStencilState:depth_state];
}

void Encoder::set_stencil_reference(std::uint32_t front, std::uint32_t back) {
    if (queue) queue->put(EncodeOp::stencil_reference, 0, StencilRecord{front, back});
    else [render setStencilFrontReferenceValue:front backReferenceValue:back];
}

void Encoder::set_cull(MTLCullMode mode) {
    if (queue) queue->put(EncodeOp::cull, std::uint32_t(mode));
    else [render setCullMode:mode];
}

void Encoder::set_winding(MTLWinding winding) {
    if (queue) queue->put(EncodeOp::winding, std::uint32_t(winding));
    else [render setFrontFacingWinding:winding];
}

void Encoder::set_fill(MTLTriangleFillMode mode) {
    if (queue) queue->put(EncodeOp::fill, std::uint32_t(mode));
    else [render setTriangleFillMode:mode];
}

void Encoder::set_depth_clip(MTLDepthClipMode mode) {
    if (queue) queue->put(EncodeOp::depth_clip, std::uint32_t(mode));
    else [render setDepthClipMode:mode];
}

void Encoder::set_depth_bias(float units, float slope, float clamp) {
    if (queue) queue->put(EncodeOp::depth_bias, 0, std::array{units, slope, clamp});
    else [render setDepthBias:units slopeScale:slope clamp:clamp];
}

void Encoder::set_blend_color(const std::array<float, 4>& color) {
    if (queue) queue->put(EncodeOp::blend_color, 0, color);
    else [render setBlendColorRed:color[0] green:color[1] blue:color[2] alpha:color[3]];
}

void Encoder::set_viewports(const MTLViewport* viewports, std::size_t count) {
    if (queue) std::memcpy(queue->reserve(EncodeOp::viewports, std::uint32_t(count), count * sizeof *viewports),
                           viewports, count * sizeof *viewports);
    else [render setViewports:viewports count:count];
}

void Encoder::set_scissors(const MTLScissorRect* scissors, std::size_t count) {
    if (queue) std::memcpy(queue->reserve(EncodeOp::scissors, std::uint32_t(count), count * sizeof *scissors),
                           scissors, count * sizeof *scissors);
    else [render setScissorRects:scissors count:count];
}

void Encoder::set_visibility(MTLVisibilityResultMode mode, std::size_t offset) {
    if (queue) queue->put(EncodeOp::visibility, std::uint32_t(mode), std::uint64_t(offset));
    else [render setVisibilityResultMode:mode offset:offset];
}

void Encoder::encode_draw(MTLPrimitiveType primitive, std::uint32_t start, std::uint32_t count,
                          std::uint32_t instances, std::uint32_t base_instance) {
    // A drain while preparing the draw resolved its block on the CPU.
    if (!state.render_enabled) return;
    if (predicate_open()) {
        const MTLDrawPrimitivesIndirectArguments arguments{count, instances, start, base_instance};
        const auto address = predicate_arguments(&arguments, sizeof arguments);
        if (!address) return;
        if (queue) {
            queue->put(EncodeOp::draw_indirect_args, 0, DrawIndirectArgsRecord{std::uint32_t(primitive), address});
            queue->publish();
        } else {
            [render drawPrimitives:primitive indirectBuffer:address];
        }
        return;
    }
    if (queue) {
        queue->put(EncodeOp::draw, 0,
                   DrawRecord{std::uint32_t(primitive), start, count, instances, base_instance});
        queue->publish();
        return;
    }
    [render drawPrimitives:primitive vertexStart:start vertexCount:count
             instanceCount:instances baseInstance:base_instance];
}

void Encoder::encode_draw_indexed(MTLPrimitiveType primitive, std::uint32_t count, MTLIndexType type,
                                  std::uint64_t address, std::uint64_t length, std::uint32_t instances,
                                  std::int32_t base_vertex, std::uint32_t base_instance) {
    if (!state.render_enabled) return;
    if (predicate_open()) {
        const MTLDrawIndexedPrimitivesIndirectArguments arguments{
            count, instances, 0, base_vertex, base_instance};
        const auto arguments_address = predicate_arguments(&arguments, sizeof arguments);
        if (!arguments_address) return;
        if (queue) {
            queue->put(EncodeOp::draw_indexed_indirect_args, 0,
                       DrawIndexedIndirectArgsRecord{std::uint32_t(primitive), std::uint32_t(type), address, length,
                                                     arguments_address});
            queue->publish();
        } else {
            [render drawIndexedPrimitives:primitive indexType:type indexBuffer:address indexBufferLength:length
                           indirectBuffer:arguments_address];
        }
        return;
    }
    if (queue) {
        queue->put(EncodeOp::draw_indexed, 0,
                   DrawIndexedRecord{std::uint32_t(primitive), count, std::uint32_t(type), instances,
                                     base_instance, base_vertex, address, length});
        queue->publish();
        return;
    }
    [render drawIndexedPrimitives:primitive indexCount:count indexType:type indexBuffer:address
                indexBufferLength:length instanceCount:instances baseVertex:base_vertex
                     baseInstance:base_instance];
}

bool Encoder::set_compute_pipeline(const NativeComputePipeline& pipeline) {
    // The encode thread waits for the compile when it reaches this record.
    if (queue) {
        queue->put(EncodeOp::compute_pipeline_compile, 0, static_cast<const void*>(pipeline.compile.get()));
        return true;
    }
    id<MTLComputePipelineState> state = pipeline.compile->wait();
    if (!state) return fail(ReplayErrorCode::unsupported, pipeline.compile->error);
    [transfer setComputePipelineState:state];
    return true;
}

void Encoder::set_compute_table(id<MTL4ArgumentTable> table) {
    if (queue) queue->put(EncodeOp::compute_table, 0, (__bridge void*)table);
    else [transfer setArgumentTable:table];
}

void Encoder::encode_dispatch(MTLSize grid, MTLSize threads) {
    if (queue) {
        queue->put(EncodeOp::dispatch, 0, DispatchRecord{grid, threads});
        queue->publish();
        return;
    }
    [transfer dispatchThreadgroups:grid threadsPerThreadgroup:threads];
}

void Encoder::encode_dispatch_indirect(std::uint64_t address, MTLSize threads) {
    if (queue) {
        queue->put(EncodeOp::dispatch_indirect, 0, DispatchIndirectRecord{address, threads});
        queue->publish();
        return;
    }
    [transfer dispatchThreadgroupsWithIndirectBuffer:address threadsPerThreadgroup:threads];
}

// Copy records hold their operands until replayed.
void Encoder::copy_buffer_range(id<MTLBuffer> source, std::uint64_t source_offset,
                                id<MTLBuffer> destination, std::uint64_t destination_offset,
                                std::uint64_t size) {
    if (queue) {
        CopyRecord record;
        record.source = (__bridge_retained void*)source;
        record.destination = (__bridge_retained void*)destination;
        record.source_offset = source_offset;
        record.destination_offset = destination_offset;
        record.size = size;
        queue->put(EncodeOp::copy_buffer, 0, record);
        queue->publish();
        return;
    }
    [transfer copyFromBuffer:source sourceOffset:source_offset toBuffer:destination
           destinationOffset:destination_offset size:size];
}

void Encoder::copy_into_texture(id<MTLBuffer> source, std::uint64_t offset, std::uint64_t row,
                                std::uint64_t image, MTLSize extent, id<MTLTexture> destination,
                                std::uint32_t slice, std::uint32_t level, MTLOrigin origin) {
    if (queue) {
        CopyRecord record;
        record.source = (__bridge_retained void*)source;
        record.destination = (__bridge_retained void*)destination;
        record.source_offset = offset;
        record.row = row;
        record.image = image;
        record.extent = extent;
        record.destination_slice = slice;
        record.destination_level = level;
        record.destination_origin = origin;
        queue->put(EncodeOp::copy_buffer_texture, 0, record);
        queue->publish();
        return;
    }
    [transfer copyFromBuffer:source sourceOffset:offset sourceBytesPerRow:row
         sourceBytesPerImage:image sourceSize:extent toTexture:destination
            destinationSlice:slice destinationLevel:level destinationOrigin:origin];
}

void Encoder::copy_from_texture(id<MTLTexture> source, std::uint32_t slice, std::uint32_t level,
                                MTLOrigin origin, MTLSize extent, id<MTLBuffer> destination,
                                std::uint64_t offset, std::uint64_t row, std::uint64_t image) {
    if (queue) {
        CopyRecord record;
        record.source = (__bridge_retained void*)source;
        record.destination = (__bridge_retained void*)destination;
        record.source_slice = slice;
        record.source_level = level;
        record.source_origin = origin;
        record.extent = extent;
        record.destination_offset = offset;
        record.row = row;
        record.image = image;
        queue->put(EncodeOp::copy_texture_buffer, 0, record);
        queue->publish();
        return;
    }
    [transfer copyFromTexture:source sourceSlice:slice sourceLevel:level sourceOrigin:origin
                   sourceSize:extent toBuffer:destination destinationOffset:offset
       destinationBytesPerRow:row destinationBytesPerImage:image];
}

void Encoder::copy_texture_range(id<MTLTexture> source, std::uint32_t source_slice,
                                 std::uint32_t source_level, MTLOrigin source_origin, MTLSize extent,
                                 id<MTLTexture> destination, std::uint32_t destination_slice,
                                 std::uint32_t destination_level, MTLOrigin destination_origin) {
    if (queue) {
        CopyRecord record;
        record.source = (__bridge_retained void*)source;
        record.destination = (__bridge_retained void*)destination;
        record.source_slice = source_slice;
        record.source_level = source_level;
        record.source_origin = source_origin;
        record.extent = extent;
        record.destination_slice = destination_slice;
        record.destination_level = destination_level;
        record.destination_origin = destination_origin;
        queue->put(EncodeOp::copy_texture, 0, record);
        queue->publish();
        return;
    }
    [transfer copyFromTexture:source sourceSlice:source_slice sourceLevel:source_level
                 sourceOrigin:source_origin sourceSize:extent toTexture:destination
             destinationSlice:destination_slice destinationLevel:destination_level
            destinationOrigin:destination_origin];
}

void Encoder::write_timestamp(id<MTL4CounterHeap> heap, std::size_t index) {
    if (queue) {
        queue->put(EncodeOp::timestamp, 0, TimestampRecord{(__bridge void*)heap, index});
        queue->publish();
        return;
    }
    [pending.commands writeTimestampIntoHeap:heap atIndex:index];
}

// A timestamp inside the open render pass, once the draws before it finished
// their fragment work (relaxed: it may be taken later, at an encoder edge).
void Encoder::write_render_timestamp(id<MTL4CounterHeap> heap, std::size_t index) {
    if (queue) {
        queue->put(EncodeOp::render_timestamp, 0, TimestampRecord{(__bridge void*)heap, index});
        queue->publish();
        return;
    }
    [render writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageFragment
                                 intoHeap:heap atIndex:index];
}

bool Encoder::encode_call(EncodeCall call, ReplayErrorCode code) {
    // The call's encoders order themselves after all earlier work; a pass
    // after them takes the full barrier.
    after_render_pass = false;
    // A profiled frame times the call between two command-buffer timestamps.
    const int slot = profile_call(-1);
    if (queue) {
        queue->put(EncodeOp::call, 0, (__bridge_retained void*)[call copy]);
        profile_call(slot);
        queue->publish();
        return true;
    }
    EncodeCallContext context{pending.commands, call_residency(pending, pending.commands), pending.retained, {}};
    call(context);
    profile_call(slot);
    return context.error.empty() || fail(code, std::move(context.error));
}

// ARMIGO_NVN_ATT_DONTCARE=1: measurement only. Every attachment load becomes
// DontCare, so a run shows what the loads cost; the picture is wrong.
bool measurement_dontcare_loads() {
    static const bool enabled = [] {
        const char* value = std::getenv("ARMIGO_NVN_ATT_DONTCARE");
        return value && *value && std::string_view(value) != "0";
    }();
    return enabled;
}

thread_local Encoder* active_encoder = nullptr;
struct EncodingScope {
    Encoder* previous;
    explicit EncodingScope(Encoder& encoder) : previous(std::exchange(active_encoder, &encoder)) {}
    ~EncodingScope() { active_encoder = previous; }
};

// ---- Pipeline manifest rows and warm-up -----------------------------------

// A render pipeline's manifest row: its stages by identity and its state as
// the key's bytes after the two stage pointers. Empty when a stage has no
// identity (not loaded from a package), which nothing can warm up.
std::optional<PsoRow> pipeline_row(const DrawPipelineKey& key) {
    if (!key.vertex || key.vertex->identity.empty() || (key.fragment && key.fragment->identity.empty()))
        return std::nullopt;
    constexpr auto state_offset = offsetof(DrawPipelineKey, colors);
    return PsoRow{'R', {key.vertex->identity, key.fragment ? key.fragment->identity : "-",
                        "v1:" + pso_hex_encode(reinterpret_cast<const std::byte*>(&key) + state_offset,
                                               sizeof key - state_offset)}};
}

// A program image can have several control/constant variants, and a mesh
// can be tailored to its fragment inputs. Name the emitted source as well.
std::string mesh_stage_reference(const platform::CompiledGraphicsStage& stage) {
    auto text = stage.identity;
    if (!stage.msl_source_digest.empty()) text += "@" + stage.msl_source_digest;
    if (!stage.native_stage_identity.empty()) text += "#" + stage.native_stage_identity;
    return text;
}

struct RecordedStageReference {
    std::string identity, source_digest, native_identity;
};

std::optional<RecordedStageReference> recorded_stage_reference(const std::string& text) {
    const auto hash = [](const std::string& digest) {
        return digest.size() == 64 && digest.find_first_not_of("0123456789abcdef") == std::string::npos;
    };
    const auto native = text.find('#');
    const auto base = text.substr(0, native);
    const auto native_identity = native == std::string::npos ? std::string{} : text.substr(native + 1);
    if (native != std::string::npos && !hash(native_identity)) return std::nullopt;
    const auto at = base.find('@');
    if (at == std::string::npos) return RecordedStageReference{base, {}, native_identity};
    const auto digest = base.substr(at + 1);
    if (at == 0 || !hash(digest))
        return std::nullopt;
    return RecordedStageReference{base.substr(0, at), digest, native_identity};
}

std::optional<PsoRow> mesh_pipeline_row(const MeshPipelineKey& key) {
    auto draw = pipeline_row(key.draw);
    if (!draw || (key.task && key.task->identity.empty())) return std::nullopt;
    return PsoRow{'M', {key.task ? mesh_stage_reference(*key.task) : "-", mesh_stage_reference(*key.draw.vertex),
                        key.draw.fragment ? mesh_stage_reference(*key.draw.fragment) : "-", draw->fields[2]}};
}

// A clear pipeline's manifest row, and the configuration a row names (none
// for a malformed row).
PsoRow clear_row(const ClearPipelineConfig& config) {
    return {'K', {config.vertex, config.fragment, std::to_string(config.color), std::to_string(config.mask),
                  std::to_string(config.depth), std::to_string(config.stencil), std::to_string(config.samples)}};
}

std::optional<ClearPipelineConfig> clear_config(const PsoRow& row) {
    if (row.kind != 'K' || row.fields.size() != 7) return std::nullopt;
    std::array<std::uint64_t, 5> numbers{};
    for (std::size_t index = 0; index != numbers.size(); ++index) {
        const auto& field = row.fields[index + 2];
        const auto [end, error] = std::from_chars(field.data(), field.data() + field.size(), numbers[index]);
        if (error != std::errc{} || end != field.data() + field.size() || field.empty()) return std::nullopt;
    }
    return ClearPipelineConfig{row.fields[0], row.fields[1], static_cast<MTLPixelFormat>(numbers[0]),
                               static_cast<MTLColorWriteMask>(numbers[1]), static_cast<MTLPixelFormat>(numbers[2]),
                               static_cast<MTLPixelFormat>(numbers[3]), static_cast<std::uint32_t>(numbers[4])};
}

// The fragment output kind a color format selects through the fragment
// stage's function constants: 0 float, 1 unsigned, 2 signed integer.
std::uint32_t fragment_output_scalar(std::uint32_t format) {
    switch (format) {
        case MTLPixelFormatR8Uint:
        case MTLPixelFormatR16Uint:
        case MTLPixelFormatR32Uint:
        case MTLPixelFormatRG8Uint:
        case MTLPixelFormatRG16Uint:
        case MTLPixelFormatRG32Uint:
        case MTLPixelFormatRGBA8Uint:
        case MTLPixelFormatRGBA16Uint:
        case MTLPixelFormatRGBA32Uint: return 1;
        case MTLPixelFormatR8Sint:
        case MTLPixelFormatR16Sint:
        case MTLPixelFormatR32Sint:
        case MTLPixelFormatRG8Sint:
        case MTLPixelFormatRG16Sint:
        case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA8Sint:
        case MTLPixelFormatRGBA16Sint:
        case MTLPixelFormatRGBA32Sint: return 2;
        default: return 0;
    }
}

// The identity of a pipeline's open form: its stages, the fragment output
// kinds its color formats select, and every state byte except the color
// formats, blending and write masks. Every attachment of the open form is
// unspecialized, so the formats reach it only as those kinds (an attachment
// without a format selects the float kind), and the primitive only as its
// topology class.
std::string open_key(const DrawPipelineKey& key) {
    DrawPipelineKey open = key;
    const bool kinds = key.fragment && key.fragment->fragment_output_types_from_render_targets;
    for (auto& color : open.colors) color = kinds ? fragment_output_scalar(color) : 0u;
    open.blend = {};
    open.blend_enable = 0;
    open.channel_mask = 0;
    open.primitive = key.vertex->writes_point_size ? -1 : static_cast<std::int32_t>(topology_class(key.primitive));
    std::string text(reinterpret_cast<const char*>(&open), sizeof open);
    return text;
}

// Why a render row did not resolve: a stage the package lacks says the
// manifest is from another shader package; state bytes that do not decode say
// the key layout changed since the manifest was written.
enum class RowResolution { resolved, missing_stage, undecodable_state };

// A shader package answers a row by the identity it stored. A native section
// answers a run-time identity from the same program image. A generated compute
// list names the packaged stage identity instead, and still resolves that way.
enum class RecordedStageForm { runtime, runtime_or_packaged_compute };

#ifdef ARMIGO_HAS_NATIVE_PACKAGE
shader_input::Stage recorded_input_stage(armigo::shader_native::NativeStageKind stage) {
    switch (stage) {
    case armigo::shader_native::NativeStageKind::vertex: return shader_input::Stage::Vertex;
    case armigo::shader_native::NativeStageKind::fragment: return shader_input::Stage::Fragment;
    case armigo::shader_native::NativeStageKind::task: return shader_input::Stage::TessellationControl;
    case armigo::shader_native::NativeStageKind::mesh: return shader_input::Stage::TessellationEvaluation;
    case armigo::shader_native::NativeStageKind::compute: return shader_input::Stage::Compute;
    }
    return shader_input::Stage::Compute;
}

// make_source_stage hashes source text only when it is in memory. An omitted
// source.metal names the stand-in "packaged" so the call accepts it, then the
// packaged library replaces the source.
std::shared_ptr<const platform::CompiledGraphicsStage> stage_from_native_files(
    shader_input::Stage stage, const armigo::shader_native::NativeStageFiles& files, std::string native_identity) {
    auto made = platform::make_source_stage(stage, files.program, {files.reflection.data(), files.reflection.size()},
                                             files.source, files.source.empty() ? std::string("packaged") : std::string{});
    const auto* built = std::get_if<std::shared_ptr<const platform::CompiledGraphicsStage>>(&made);
    if (!built || !*built) return nullptr;
    platform::CompiledGraphicsStage copy = **built;
    copy.native_stage_identity = std::move(native_identity);
    copy.metallib = files.library;
    copy.msl_source.clear();
    copy.msl_source_file.clear();
    return std::make_shared<const platform::CompiledGraphicsStage>(std::move(copy));
}

// The native section of the warm-up's package, opened at its first use.
struct NativeRowSource {
    std::once_flag once;
    std::unique_ptr<armigo::shader_native::NativePackage> package;
};

std::shared_ptr<const platform::CompiledGraphicsStage> resolve_recorded_stage(
    platform::MetalShaderPackage& package, NativeRowSource& native, const std::string& identity,
    RecordedStageForm form) {
    const auto reference = recorded_stage_reference(identity);
    if (!reference) return nullptr;
    const auto matches = [&](const auto& stage) {
        return stage && (reference->source_digest.empty() || stage->msl_source_digest == reference->source_digest) &&
            (reference->native_identity.empty() || stage->native_stage_identity == reference->native_identity);
    };
    if (const auto stage = package.stage_by_identity(reference->identity)) return matches(stage) ? stage : nullptr;
    std::call_once(native.once, [&] {
        auto opened = armigo::shader_native::NativePackage::open(package.directory());
        if (auto* opened_package = std::get_if<armigo::shader_native::NativePackage>(&opened))
            native.package = std::make_unique<armigo::shader_native::NativePackage>(std::move(*opened_package));
    });
    const auto& native_package = native.package;
    if (!native_package) return nullptr;
    if (const auto parsed = armigo::shader_native::parse_runtime_stage_identity(reference->identity)) {
        auto stage_identity = native_package->identity_for_program(parsed->program_digest, parsed->stage);
        if (!reference->native_identity.empty()) stage_identity = reference->native_identity;
        else if (!reference->source_digest.empty()) {
            stage_identity.reset();
            const auto depth = parsed->stage == armigo::shader_native::NativeStageKind::mesh ||
                parsed->stage == armigo::shader_native::NativeStageKind::vertex
                ? (parsed->vertex_depth_zero_to_one ? armigo::shader_native::NativeDepth::zero_to_one
                                                    : armigo::shader_native::NativeDepth::minus_one_to_one)
                : armigo::shader_native::NativeDepth::stage;
            for (const auto& candidate : native_package->section().entries) {
                if (candidate.stage != parsed->stage || candidate.program_digest != parsed->program_digest) continue;
                for (const auto& variant : candidate.variants)
                    if (variant.depth == depth && variant.source_digest == reference->source_digest) {
                        stage_identity = candidate.stage_identity;
                        break;
                    }
                if (stage_identity) break;
            }
        }
        if (!stage_identity) return nullptr;
        // The packer translates with pixel rate off, no vertex fetch, and a
        // fragment position that matches the guest.
        armigo::shader_native::NativeLookup lookup;
        lookup.stage = parsed->stage;
        lookup.pixel_rate_disabled = true;
        lookup.vertex_fetch = false;
        lookup.fragment_position_matches = true;
        lookup.vertex_depth_zero_to_one = parsed->vertex_depth_zero_to_one;
        auto found = native_package->find(*stage_identity, lookup);
        const auto* files = std::get_if<armigo::shader_native::NativeStageFiles>(&found);
        if (!files) return nullptr;
        const auto stage = stage_from_native_files(recorded_input_stage(parsed->stage), *files, *stage_identity);
        // The built identity carries the depth or compute suffix. A row is a
        // hit only when that text is the one it named.
        if (!matches(stage) || stage->identity != reference->identity) return nullptr;
        return stage;
    }
    if (form != RecordedStageForm::runtime_or_packaged_compute) return nullptr;
    armigo::shader_native::NativeLookup lookup;
    lookup.stage = armigo::shader_native::NativeStageKind::compute;
    lookup.pixel_rate_disabled = true;
    auto found = native_package->find(reference->identity, lookup);
    const auto* files = std::get_if<armigo::shader_native::NativeStageFiles>(&found);
    if (!files) return nullptr;
    const auto stage = stage_from_native_files(shader_input::Stage::Compute, *files, reference->identity);
    return matches(stage) ? stage : nullptr;
}

#else
// Without the native package reader a row resolves from the shader package only.
struct NativeRowSource {};

std::shared_ptr<const platform::CompiledGraphicsStage> resolve_recorded_stage(
    platform::MetalShaderPackage& package, NativeRowSource&, const std::string& identity, RecordedStageForm) {
    const auto reference = recorded_stage_reference(identity);
    if (!reference) return nullptr;
    const auto stage = package.stage_by_identity(reference->identity);
    return stage && (reference->source_digest.empty() || stage->msl_source_digest == reference->source_digest)
        && (reference->native_identity.empty() || stage->native_stage_identity == reference->native_identity)
        ? stage : nullptr;
}
#endif

// The key a row names, with its stages loaded from the shader package or,
// when a run-time identity misses there, from the native section.
RowResolution resolve_pipeline_row(const PsoRow& row, platform::MetalShaderPackage& package,
                                   NativeRowSource& native, DrawPipelineKey& key,
                                   std::shared_ptr<const platform::CompiledGraphicsStage>& vertex,
                                   std::shared_ptr<const platform::CompiledGraphicsStage>& fragment) {
    constexpr auto state_offset = offsetof(DrawPipelineKey, colors);
    if ((row.kind != 'R' && row.kind != 'O') || row.fields.size() != 3 || row.fields[2].rfind("v1:", 0) != 0)
        return RowResolution::undecodable_state;
    vertex = resolve_recorded_stage(package, native, row.fields[0], RecordedStageForm::runtime);
    fragment = row.fields[1] == "-"
                   ? nullptr
                   : resolve_recorded_stage(package, native, row.fields[1], RecordedStageForm::runtime);
    if (!vertex || (row.fields[1] != "-" && !fragment)) return RowResolution::missing_stage;
    std::memset(&key, 0, sizeof key);
    if (!pso_hex_decode(std::string_view(row.fields[2]).substr(3),
                        reinterpret_cast<std::byte*>(&key) + state_offset, sizeof key - state_offset))
        return RowResolution::undecodable_state;
    key.vertex = vertex.get();
    key.fragment = fragment.get();
    // A row recorded before the stage fetched its own vertices names the
    // layout the draw key of a fetching stage leaves out.
    if (vertex->vertex_fetch_buffer_index) {
        key.attributes = {};
        key.streams = {};
    }
    return RowResolution::resolved;
}

RowResolution resolve_mesh_pipeline_row(const PsoRow& row, platform::MetalShaderPackage& package,
                                        NativeRowSource& native, MeshPipelineKey& key,
                                        std::shared_ptr<const platform::CompiledGraphicsStage>& task,
                                        std::shared_ptr<const platform::CompiledGraphicsStage>& mesh,
                                        std::shared_ptr<const platform::CompiledGraphicsStage>& fragment) {
    if (row.kind != 'M' || row.fields.size() != 4) return RowResolution::undecodable_state;
    const auto resolution = resolve_pipeline_row({'R', {row.fields[1], row.fields[2], row.fields[3]}},
                                                 package, native, key.draw, mesh, fragment);
    if (resolution != RowResolution::resolved) return resolution;
    task = row.fields[0] == "-" ? nullptr
        : resolve_recorded_stage(package, native, row.fields[0], RecordedStageForm::runtime);
    if (row.fields[0] != "-" && !task) return RowResolution::missing_stage;
    if (!mesh->mesh || mesh->stage != shader_input::Stage::TessellationEvaluation ||
        (task && (!task->task || task->stage != shader_input::Stage::TessellationControl)) ||
        (fragment && fragment->stage != shader_input::Stage::Fragment) ||
        mesh->mesh->launched_by_task != (task != nullptr) ||
        (task && mesh->mesh->payload_bytes > task->task->payload_bytes) ||
        key.draw.primitive != mesh_draw_primitive)
        return RowResolution::undecodable_state;
    key.task = task.get();
    key.draw.attributes = {};
    key.draw.streams = {};
    return RowResolution::resolved;
}

// Builds the manifest's shaders and pipelines in row order on background
// threads at low priority, from launch, so the first use of each finds it
// built (or compiling). The replay takes an entry when it first needs that
// pipeline; one it needs before the warm-up reached it is built the usual way
// and the warm-up skips it.
class PipelineWarmup {
public:
    struct Hooks {
        std::function<id<MTLLibrary>(const std::shared_ptr<const platform::CompiledGraphicsStage>&)> library;
        std::function<std::unique_ptr<NativePipeline>(
            const DrawPipelineKey&, std::shared_ptr<const platform::CompiledGraphicsStage>,
            std::shared_ptr<const platform::CompiledGraphicsStage>)> render;
        std::function<std::unique_ptr<NativePipeline>(
            const MeshPipelineKey&, std::shared_ptr<const platform::CompiledGraphicsStage>,
            std::shared_ptr<const platform::CompiledGraphicsStage>,
            std::shared_ptr<const platform::CompiledGraphicsStage>)> mesh;
        std::function<std::optional<NativeComputePipeline>(
            std::shared_ptr<const platform::CompiledGraphicsStage>)> compute;
        std::function<bool(const ClearPipelineConfig&)> clear;
        // An archive build's open-form row; unset when not capturing, which
        // skips those rows.
        std::function<bool(const DrawPipelineKey&, std::shared_ptr<const platform::CompiledGraphicsStage>,
                           std::shared_ptr<const platform::CompiledGraphicsStage>)> open;
        std::function<void()> done;
    };
    struct Counts {
        std::uint64_t rows = 0, built = 0, unresolved = 0, undecodable = 0, failed = 0, taken_render = 0,
                      taken_compute = 0;
        bool finished = false;
    };

    PipelineWarmup(std::shared_ptr<platform::MetalShaderPackage> package, std::vector<PsoRow> rows,
                   unsigned threads, Hooks hooks)
        : package_(std::move(package)), rows_(std::move(rows)), hooks_(std::move(hooks)),
          started_(timeline_now_ns()) {
        counts_rows_ = rows_.size();
        thread_count_ = std::clamp(threads, 1u, 16u);
    }
    // Starts the threads. Separate from construction so the owner can finish
    // assigning the object before a thread's done hook can reach it.
    void start() {
        active_ = thread_count_;
        for (unsigned i = 0; i != thread_count_; ++i) threads_.emplace_back([this] { run(); });
    }
    ~PipelineWarmup() {
        stop_ = true;
        for (auto& thread : threads_) thread.join();
    }

    // The built pipeline for a row's text, once; null when it was not (yet)
    // built. Either way the replay owns that pipeline from now on.
    // Whether the calling thread is one of the warm-up's.
    static bool on_warmup_thread() { return warmup_thread_; }

    std::unique_ptr<NativePipeline> take_render(const std::string& key) {
        const std::lock_guard lock(mutex_);
        if (const auto found = render_.find(key); found != render_.end()) {
            auto pipeline = std::move(found->second);
            render_.erase(found);
            ++taken_render_;
            return pipeline;
        }
        claimed_.insert(key);
        return nullptr;
    }
    std::optional<NativeComputePipeline> take_compute(const std::string& identity) {
        const std::lock_guard lock(mutex_);
        if (const auto found = compute_.find(identity); found != compute_.end()) {
            auto pipeline = std::move(found->second);
            compute_.erase(found);
            ++taken_compute_;
            return pipeline;
        }
        claimed_.insert(identity);
        return std::nullopt;
    }
    // Returns once the last warm-up thread has finished, including the done hook.
    void wait_done() {
        std::unique_lock lock(done_mutex_);
        done_changed_.wait(lock, [&] { return done_; });
    }
    Counts counts() const {
        Counts result;
        result.rows = counts_rows_;
        result.built = built_;
        result.unresolved = unresolved_;
        result.undecodable = undecodable_;
        result.failed = failed_;
        result.finished = finished_;
        const std::lock_guard lock(mutex_);
        result.taken_render = taken_render_;
        result.taken_compute = taken_compute_;
        return result;
    }

private:
    bool claimed(const std::string& key) const {
        const std::lock_guard lock(mutex_);
        return claimed_.count(key) != 0;
    }

    void run() {
        warmup_thread_ = true;
        pthread_setname_np("nvn-pso-warm");
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
        for (;;) {
            const auto index = next_.fetch_add(1);
            if (index >= rows_.size() || stop_) break;
            @autoreleasepool { build(rows_[index]); }
        }
        if (active_.fetch_sub(1) != 1) return;
        // An early stop leaves rows unbuilt; only a full pass counts as finished.
        finished_ = !stop_;
        const auto elapsed = timeline_now_ns() - started_;
        timeline_event("pso_warm_done", built_, elapsed);
        armigo::diag::write(armigo::diag::Level::info, 
                     "Armigo NVN: pipeline warm-up finished: %llu rows, %llu built, %llu not in this "
                     "package, %llu with state that does not decode (key layout changed), %llu failed, "
                     "%.1f s\n",
                     (unsigned long long)counts_rows_, (unsigned long long)built_,
                     (unsigned long long)unresolved_, (unsigned long long)undecodable_,
                     (unsigned long long)failed_, elapsed / 1e9);
        if (!stop_ && hooks_.done) hooks_.done();
        {
            const std::lock_guard lock(done_mutex_);
            done_ = true;
        }
        done_changed_.notify_all();
    }

    void build(const PsoRow& row) {
        std::shared_ptr<const platform::CompiledGraphicsStage> vertex, fragment;
        if (row.kind == 'S') {
            const auto stage = resolve_recorded_stage(*package_, native_rows_, row.fields[0],
                                                      RecordedStageForm::runtime);
            if (!stage) { ++unresolved_; return; }
            if (hooks_.library(stage)) ++built_; else ++failed_;
        } else if (row.kind == 'R') {
            DrawPipelineKey key;
            const auto resolution = resolve_pipeline_row(row, *package_, native_rows_, key, vertex, fragment);
            if (resolution != RowResolution::resolved) {
                ++(resolution == RowResolution::missing_stage ? unresolved_ : undecodable_);
                return;
            }
            const auto canonical = pipeline_row(key);
            const auto text = canonical ? canonical->text() : row.text();
            {
                // Rows that differ only in a layout a fetching stage ignores
                // share one pipeline.
                const std::lock_guard lock(mutex_);
                if (claimed_.count(text) || render_.count(text)) return;
            }
            auto pipeline = hooks_.render(key, vertex, fragment);
            if (!pipeline) { ++failed_; return; }
            const auto compile = pipeline->compile;
            {
                const std::lock_guard lock(mutex_);
                // The replay needed it meanwhile and built its own, or another
                // row of the same pipeline came first.
                if (claimed_.count(text) || !render_.emplace(text, std::move(pipeline)).second) return;
            }
            ++built_;
            // One compile in flight per thread keeps the warm-up from
            // flooding the compiler ahead of what the replay needs.
            compile->wait_silent();
        } else if (row.kind == 'M') {
            std::shared_ptr<const platform::CompiledGraphicsStage> task;
            MeshPipelineKey key;
            const auto resolution = resolve_mesh_pipeline_row(row, *package_, native_rows_, key,
                                                               task, vertex, fragment);
            if (resolution != RowResolution::resolved) {
                ++(resolution == RowResolution::missing_stage ? unresolved_ : undecodable_);
                return;
            }
            const auto canonical = mesh_pipeline_row(key);
            const auto text = canonical ? canonical->text() : row.text();
            {
                const std::lock_guard lock(mutex_);
                if (claimed_.count(text) || render_.count(text)) return;
            }
            auto pipeline = hooks_.mesh ? hooks_.mesh(key, task, vertex, fragment) : nullptr;
            if (!pipeline) { ++failed_; return; }
            // Publish only a successful ready mesh pipeline. A draw that
            // outruns preparation uses its archive lookup instead; an
            // incomplete archive can still require a draw-time compile.
            if (!pipeline->compile->wait_silent()) { ++failed_; return; }
            {
                const std::lock_guard lock(mutex_);
                if (claimed_.count(text) || !render_.emplace(text, std::move(pipeline)).second) return;
            }
            ++built_;
        } else if (row.kind == 'O') {
            if (!hooks_.open) return;
            DrawPipelineKey key;
            const auto resolution = resolve_pipeline_row(row, *package_, native_rows_, key, vertex, fragment);
            if (resolution != RowResolution::resolved) {
                ++(resolution == RowResolution::missing_stage ? unresolved_ : undecodable_);
                return;
            }
            if (hooks_.open(key, std::move(vertex), std::move(fragment))) ++built_; else ++failed_;
        } else if (row.kind == 'C') {
            const auto stage = resolve_recorded_stage(*package_, native_rows_, row.fields[0],
                                                      RecordedStageForm::runtime_or_packaged_compute);
            if (!stage) { ++unresolved_; return; }
            if (claimed(row.fields[0])) return;
            auto pipeline = hooks_.compute(stage);
            if (!pipeline) { ++failed_; return; }
            const auto compile = pipeline->compile;
            {
                const std::lock_guard lock(mutex_);
                if (claimed_.count(row.fields[0])) return;
                compute_.emplace(row.fields[0], std::move(*pipeline));
            }
            ++built_;
            compile->wait_silent();
        } else if (row.kind == 'K') {
            const auto config = clear_config(row);
            if (!config) { ++undecodable_; return; }
            if (hooks_.clear && hooks_.clear(*config)) ++built_; else ++failed_;
            // Clear compiles at warm-up are part of its time, not first-use compiles.
            (void)take_clear_compiles();
        }
    }

    std::shared_ptr<platform::MetalShaderPackage> package_;
    NativeRowSource native_rows_;
    const std::vector<PsoRow> rows_;
    Hooks hooks_;
    const std::uint64_t started_;
    std::uint64_t counts_rows_ = 0;
    std::atomic<std::size_t> next_{0};
    std::atomic<unsigned> active_{0};
    unsigned thread_count_ = 1;
    std::atomic<std::uint64_t> built_{0}, unresolved_{0}, undecodable_{0}, failed_{0};
    std::atomic<bool> stop_{false}, finished_{false};
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<NativePipeline>> render_;
    std::unordered_map<std::string, NativeComputePipeline> compute_;
    std::unordered_set<std::string> claimed_;
    std::uint64_t taken_render_ = 0, taken_compute_ = 0;
    static inline thread_local bool warmup_thread_ = false;
    std::mutex done_mutex_;
    std::condition_variable done_changed_;
    bool done_ = false;
    std::vector<std::thread> threads_;
};

// The replay whose pipeline archive the process-exit handler flushes.
std::mutex& archive_exit_mutex() {
    static std::mutex mutex;
    return mutex;
}
class ReplayImpl;
ReplayImpl*& archive_exit_target() {
    static ReplayImpl* target = nullptr;
    return target;
}

// Fence publication precedes diagnostics. A shared tail ticket lets the
// callback finish without touching a potentially retired backend, while
// shutdown separately waits for every logging tail and retained-log owner.
struct RetirementTails {
    std::mutex mutex;
    std::condition_variable changed;
    std::uint64_t active = 0;
    bool drain(std::optional<std::chrono::steady_clock::time_point> deadline) {
        std::unique_lock lock(mutex);
        if (deadline) return changed.wait_until(lock, *deadline, [&] { return active == 0; });
        changed.wait(lock, [&] { return active == 0; });
        return true;
    }
};
struct RetirementTailTicket {
    explicit RetirementTailTicket(std::shared_ptr<RetirementTails> tails) : tails(std::move(tails)) {
        const std::lock_guard lock(this->tails->mutex);
        ++this->tails->active;
    }
    RetirementTailTicket(const RetirementTailTicket&) = delete;
    RetirementTailTicket& operator=(const RetirementTailTicket&) = delete;
    ~RetirementTailTicket() {
        const std::lock_guard lock(tails->mutex);
        --tails->active;
        tails->changed.notify_all();
    }
    std::shared_ptr<RetirementTails> tails;
};

// Pools guest-memory stores have written, learned from each submission's
// write log. Until that set has stayed the same across two logs, a storing
// stage counts as a write anywhere in the table. After that, it counts as a
// write of the learned pools only. Past the cap, stores count as a write
// anywhere again.
struct LearnedGuestWrites {
    static constexpr std::size_t default_cap = 4096;
    static constexpr int stable_after = 2;
    static constexpr std::size_t ring_size = 256;
    struct Read {
        // Encoding sequence of the submission whose prediction this read
        // relied on. Held queues encode out of guest-fence order, so the
        // learning cutoff and the wrap check use this sequence.
        std::uint64_t sequence = 0;
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
    };
    mutable std::mutex mutex;
    std::size_t cap = default_cap;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> snapshot =
        std::make_shared<const std::vector<std::pair<std::uint64_t, std::uint64_t>>>();
    int stable = 0;
    bool over = false;
    bool announced = false;
    bool over_logged = false;
    std::uint64_t logs = 0;
    std::uint64_t misses = 0;
    std::uint64_t predicted_submissions = 0;
    int info_lines = 0;
    // Reads that skipped a wait because a submission was encoded from the
    // learned set and that set excluded the range: a CPU read, a texture
    // whose backing feeds from those pools, or a uniform capture used after
    // the storing stage. A later log that names a new pool checks reads at
    // or after that submission's encoding sequence.
    std::array<Read, ring_size> ring{};
    std::uint64_t ring_n = 0;
};

bool guest_ranges_overlap(const std::vector<std::pair<std::uint64_t, std::uint64_t>>* ranges,
                          std::uint64_t address, std::uint64_t size);

class ReplayImpl final : public MetalReplayBackend {
public:
    ReplayImpl(id<MTLDevice> device, id<MTL4CommandQueue> queue, std::shared_ptr<ReplayResources> resources,
               ReplaySchedulingPolicy scheduling)
        : device_(device), pool_tables_(device), queue_(queue), resources_(std::move(resources)),
          scheduling_(scheduling) {
        submission_owner_ = resources_->submission_owner();
        if (!submission_owner_)
            submission_owner_ = std::make_shared<Metal4SubmissionOwner>(device_, queue_);
        // Read per backend: a process-lifetime cache would keep a value set
        // by an earlier test. Unset and any value other than 0 predict.
        if (const char* value = std::getenv("ARMIGO_NVN_PREDICT_GUEST_WRITES"))
            predict_guest_writes_ = *value != '0';
        if (const char* value = std::getenv("ARMIGO_NVN_MESH_DIAGNOSTICS"))
            mesh_diagnostics_enabled_ = std::string_view(value) == "1";
        NSError* error = nil;
        compiler_ = [device_ newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        if (!device_ || !queue_) {
            last_error_ = {ReplayErrorCode::allocation,
                           "Metal replay requires a device and command queue"};
        } else if (!compiler_) {
            const char* detail = error ? error.localizedDescription.UTF8String :
                                         "unknown compiler allocation failure";
            last_error_ = {ReplayErrorCode::allocation,
                           std::string("Metal replay compiler creation failed: ") + detail};
        }
        // Missing pool bytes are unknown, so a conditional on them draws.
        reports_.set_reader([this](std::uint64_t address, std::uint64_t& value) {
            const auto span = resources_->buffer_at(address, sizeof value);
            const auto* contents = span.buffer ? static_cast<const std::byte*>(span.buffer.contents) : nullptr;
            if (!contents || span.offset > span.buffer.length || sizeof value > span.buffer.length - span.offset)
                return false;
            std::memcpy(&value, contents + span.offset, sizeof value);
            return true;
        });
        resources_->set_completion_hook([this] {
            if (active_encoder && &active_encoder->backend == this)
                return active_encoder->synchronize_gpu();
            // Every encoded submission has run: work a held queue has not
            // encoded yet cannot have touched the resource.
            std::unique_lock lock(fence_mutex_);
            const auto latest = std::ranges::max_element(inflight_, {}, [](const auto& entry) {
                return entry.second->sequence;
            });
            if (latest != inflight_.end()) {
                const auto submission = latest->second;
                completion_.wait(lock, [&] { return submission->retired.load(std::memory_order_acquire); });
            }
            lock.unlock();
            return last_error().code != ReplayErrorCode::execution;
        });
        state_.channel_mask.mask = 0xffffffffu;
        state_.color.logic_op = NVN_LOGIC_OP_COPY;
        state_.color.alpha_test = NVN_ALPHA_FUNC_ALWAYS;
        state_.multisample.enable = 1;
        state_.polygon.cull = NVN_FACE_NONE;
        state_.polygon.front_face = NVN_FRONT_FACE_CCW;
        state_.polygon.mode = NVN_POLYGON_MODE_FILL;
        for (auto& blend : state_.blend) {
            blend.source_color = NVN_BLEND_FUNC_ONE;
            blend.destination_color = NVN_BLEND_FUNC_ZERO;
            blend.source_alpha = NVN_BLEND_FUNC_ONE;
            blend.destination_alpha = NVN_BLEND_FUNC_ZERO;
            blend.color_equation = NVN_BLEND_EQUATION_ADD;
            blend.alpha_equation = NVN_BLEND_EQUATION_ADD;
        }
        state_.depth_stencil.depth_write = 1;
        state_.depth_stencil.depth_function = NVN_COMPARE_FUNC_LESS;
        state_.depth_stencil.front_function = NVN_COMPARE_FUNC_ALWAYS;
        state_.depth_stencil.back_function = NVN_COMPARE_FUNC_ALWAYS;
        state_.depth_stencil.front_fail = NVN_STENCIL_OP_KEEP;
        state_.depth_stencil.front_depth_fail = NVN_STENCIL_OP_KEEP;
        state_.depth_stencil.front_pass = NVN_STENCIL_OP_KEEP;
        state_.depth_stencil.back_fail = NVN_STENCIL_OP_KEEP;
        state_.depth_stencil.back_depth_fail = NVN_STENCIL_OP_KEEP;
        state_.depth_stencil.back_pass = NVN_STENCIL_OP_KEEP;
        initial_state_ = state_;
        const auto setting = [](const char* name, std::uint64_t fallback) {
            const char* value = std::getenv(name);
            if (!value || !*value) return fallback;
            char* end = nullptr;
            const auto parsed = std::strtoll(value, &end, 10);
            return end && !*end && parsed >= 0 ? std::uint64_t(parsed) : fallback;
        };
        segment_policy_.draws = setting("ARMIGO_NVN_CUT", segment_policy_.draws);
        segment_policy_.inflight = setting("ARMIGO_NVN_CUT_INFLIGHT", segment_policy_.inflight);
        if (setting("ARMIGO_NVN_ENCODE_THREAD", 1)) encode_thread_ = std::make_unique<EncodeThread>(*this);
        resources_->set_upload_publication([this](std::function<void()> commit) {
            if (encode_thread_) encode_thread_->upload_commit(std::move(commit));
            else commit();
        });
        if (const char* profile = std::getenv("ARMIGO_NVN_GPUPROF"); profile && !std::strncmp(profile, "every:", 6))
            gpu_profile_ = std::make_unique<GpuProfile>(device_);
        if (PassDumpConfig dump_config = read_pass_dump_config(); dump_config.enabled) {
            pass_dump_ = std::make_unique<PassDump>(device_, std::move(dump_config));
            pass_dump_enabled_ = true;
        }
        worker_ = std::thread([this] { run_worker(); });
        set_event_signal_observer(&ReplayImpl::cpu_event_signaled, this);
    }

    std::shared_ptr<Metal4SubmissionOwner> submission_owner() const override { return submission_owner_; }

    bool enqueue_frame_capture(std::uint64_t after_fence, std::function<void(bool)> capture) override {
        std::lock_guard lock(mutex_);
        if (stopping_ || after_fence >= next_fence_) return false;
        QueuedSubmission item;
        item.capture = std::move(capture);
        if (after_fence) {
            if (active_cpu_fence_ == after_fence) item.capture_after = active_cpu_ready_;
            else for (const auto& queued : queued_)
                if (queued.fence == after_fence) { item.capture_after = queued.cpu_ready; break; }
        }
        queued_.push_back(std::move(item));
        schedule_generation_.fetch_add(1, std::memory_order_release);
        queued_ready_.notify_all();
        return true;
    }

    void track_native_submission(const std::shared_ptr<Metal4NativeSubmission>& submission) override {
        std::lock_guard lock(native_mutex_);
        if (!accepting_submissions_.load(std::memory_order_acquire)) {
            submission->cancel();
            return;
        }
        if (native_submissions_.size() >= 128)
            std::erase_if(native_submissions_, [](const auto& entry) { return entry.expired(); });
        native_submissions_.push_back(submission);
    }

    ~ReplayImpl() override {
        clear_event_signal_observer(this);
        if (archive_loader_.joinable()) archive_loader_.join();
        if (internal_preparer_.joinable()) internal_preparer_.join();
        warm_.reset();
        {
            // The exit handler must not reach a destroyed replay.
            const std::lock_guard lock(archive_exit_mutex());
            if (archive_exit_target() == this) archive_exit_target() = nullptr;
        }
        (void)stop_submission_workers();
        if (mesh_diagnostics_enabled_) {
            const std::lock_guard lock(mesh_diagnostics_mutex_);
            armigo::diag::opt_in("nvn_mesh_diag_summary seen=%llu emitted=%zu duplicates=%llu records_truncated=%llu\n",
                static_cast<unsigned long long>(mesh_diagnostics_seen_), mesh_diagnostics_keys_.size(),
                static_cast<unsigned long long>(mesh_diagnostics_duplicates_),
                static_cast<unsigned long long>(mesh_diagnostics_truncated_));
        }
        reports_.forget();
    }

    bool stop_submission_workers(
        std::optional<std::chrono::steady_clock::time_point> deadline = std::nullopt) override {
        std::unique_lock serial(shutdown_mutex_, std::defer_lock);
        if (deadline ? !serial.try_lock_until(*deadline) : (serial.lock(), false)) return false;
        std::uint64_t fence;
        {
            std::lock_guard lock(mutex_);
            accepting_submissions_.store(false, std::memory_order_release);
            stopping_ = true;
            fence = next_fence_ - 1;
        }
        // A worker may be recycling an upload allocator whose GPU batch is
        // behind an unfilled native frame. Release those positions before
        // joining it; captures racing shutdown are canceled by tracking.
        std::vector<std::weak_ptr<Metal4NativeSubmission>> native;
        { std::lock_guard lock(native_mutex_); native.swap(native_submissions_); }
        for (const auto& entry : native) if (const auto submission = entry.lock()) submission->cancel();
        queued_ready_.notify_all();
        {
            std::unique_lock lock(mutex_);
            if (deadline) {
                if (!worker_exited_.wait_until(lock, *deadline, [&] { return worker_stopped_; })) return false;
            } else {
                worker_exited_.wait(lock, [&] { return worker_stopped_; });
            }
        }
        if (worker_.joinable()) worker_.join();
        resources_->set_upload_publication({});
        if (scheduling_.observe_shutdown_workers_joined)
            scheduling_.observe_shutdown_workers_joined(scheduling_.context);
        // Cleanup still drains an abandoned submission after public fence
        // waits report its fatal replay failure.
        {
            std::unique_lock lock(fence_mutex_);
            const auto retired = [&] { return fence_retired_locked(fence); };
            if (deadline) {
                if (!completion_.wait_until(lock, *deadline, retired)) return false;
            } else {
                completion_.wait(lock, retired);
            }
        }
        const auto owner_remaining = deadline ? std::max(std::chrono::nanoseconds::zero(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(*deadline - std::chrono::steady_clock::now()))
            : std::chrono::nanoseconds::max();
        if (!submission_owner_->wait(submission_owner_->issued(), owner_remaining)) return false;
        if (!retirement_tails_->drain(deadline)) return false;
        // GPU retirement proves every encode record completed. Keep the
        // backend/resources alive for late VI fence readers and archive flush.
        std::unique_ptr<EncodeThread> encode;
        {
            const std::lock_guard lock(mutex_);
            if (encode_thread_) stopped_encode_cpu_ns_ = encode_thread_->cpu_ns();
            encode = std::move(encode_thread_);
        }
        encode.reset();
        resources_->set_completion_hook({});
        return true;
    }

    // The manifest lists the stages in the order the title first selects them,
    // which is the order its first draws ask the package for them. Open-form
    // rows count only for an archive build, the one session that builds them.
    static void prefetch_manifest_stages(platform::MetalShaderPackage& package, const std::vector<PsoRow>& rows,
                                         bool archive_build) {
        std::vector<std::string> identities;
        for (const auto& row : rows) {
            if (row.fields.empty() || row.kind == 'K' || (row.kind == 'O' && !archive_build)) continue;
            if (row.kind == 'M') {
                for (std::size_t i = 0; i < 3; ++i)
                    if (row.fields[i] != "-") identities.push_back(row.fields[i]);
                continue;
            }
            identities.push_back(row.fields[0]);
            if ((row.kind == 'R' || row.kind == 'O') && row.fields.size() > 1 && row.fields[1] != "-")
                identities.push_back(row.fields[1]);
        }
        package.prefetch_first(identities);
    }

    void configure_pipeline_warmup(PipelineWarmupConfig config) override {
        if (!compiler_ || warm_ || recorder_) return;
        prepare_internal_pipelines();
        // A seed that is named but unreadable is reported; the user manifest
        // legitimately does not exist before the first run.
        const auto read_rows = [](const std::string& path, bool required) {
            std::vector<PsoRow> result;
            if (path.empty()) return result;
            std::ifstream in(path, std::ios::binary);
            if (!in) {
                if (required)
                    armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: cannot read the pipeline warm-up manifest %s; "
                                         "no seed rows are used\n", path.c_str());
                return result;
            }
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            return parse_pso_manifest(text);
        };
        const auto seed_rows = read_rows(config.seed_manifest, true);
        const auto user_rows = read_rows(config.user_manifest, false);
        auto rows = merge_pso_manifests(seed_rows, user_rows);
        recorder_ = std::make_unique<PsoRecorder>(config.record_manifest, config.user_manifest, rows);
        if (!config.archive_root.empty() && (config.package || !config.archive_identity.empty()))
            open_archive(config.package ? "package " + config.package->chain_id() + "\n" +
                                              config.package->content_id()
                                        : "identity " + config.archive_identity,
                         config.archive_root, config.archive_capture, config.archive_bundle_root);
        if (config.package) prefetch_manifest_stages(*config.package, rows, config.archive_open_rows);
        if (!config.warm || !config.package || rows.empty()) return;
        PipelineWarmup::Hooks hooks;
        hooks.library = [this](const std::shared_ptr<const platform::CompiledGraphicsStage>& stage) {
            std::string detail;
            return stage_library(stage, detail);
        };
        hooks.render = [this](const DrawPipelineKey& key,
                              std::shared_ptr<const platform::CompiledGraphicsStage> vertex,
                              std::shared_ptr<const platform::CompiledGraphicsStage> fragment) {
            std::string detail;
            return build_pipeline(key, std::move(vertex), std::move(fragment), detail);
        };
        hooks.compute = [this](std::shared_ptr<const platform::CompiledGraphicsStage> stage) {
            std::string detail;
            return build_compute_pipeline(std::move(stage), detail);
        };
        hooks.mesh = [this](const MeshPipelineKey& key,
                           std::shared_ptr<const platform::CompiledGraphicsStage> task,
                           std::shared_ptr<const platform::CompiledGraphicsStage> mesh,
                           std::shared_ptr<const platform::CompiledGraphicsStage> fragment) {
            std::string detail;
            return build_mesh_pipeline(key, std::move(task), std::move(mesh), std::move(fragment), detail);
        };
        hooks.clear = [this](const ClearPipelineConfig& config) { return prepare_clear_pipeline(device_, config); };
        if (config.archive_capture && config.archive_open_rows) {
            hooks.open = [this](const DrawPipelineKey& key,
                                std::shared_ptr<const platform::CompiledGraphicsStage> vertex,
                                std::shared_ptr<const platform::CompiledGraphicsStage> fragment) {
                std::string detail;
                const std::string names = vertex->identity + " " + (fragment ? fragment->identity : "-");
                const bool captured =
                    build_pipeline(key, std::move(vertex), std::move(fragment), detail, true) != nullptr;
                if (!captured)
                    armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: open-form row not archived (%s): %s\n", names.c_str(),
                                 detail.c_str());
                return captured;
            };
        }
        hooks.done = [this] { flush_archive(false); };
        const auto row_count = rows.size();
        warm_ = std::make_unique<PipelineWarmup>(config.package, std::move(rows), config.warm_threads,
                                                 std::move(hooks));
        warm_->start();
        armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: pipeline warm-up started (%zu manifest rows: %zu seed, %zu user)\n",
                     row_count, seed_rows.size(), user_rows.size());
    }

    // Pipelines built this session go to a Metal 4 binary archive, so a later
    // launch looks them up instead of compiling. The archive belongs to one
    // shader package, OS build and GPU; anything else finds an empty directory.
    void open_archive(const std::string& source, const std::string& root, bool capture,
                      const std::string& bundle_root = {}) {
        char build[64]{};
        std::size_t size = sizeof build - 1;
        sysctlbyname("kern.osversion", build, &size, nullptr, 0);
        const std::string identity = "armigo-pso-archive-v2\n" + source + "\n" + build + "\n" +
            (device_.name ? device_.name.UTF8String : "") + "\n" +
            (device_.architecture.name ? device_.architecture.name.UTF8String : "");
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(identity.data(), static_cast<CC_LONG>(identity.size()), digest);
        const auto name = "archive-" + pso_hex_encode(digest, 12);
        archive_directory_ = std::filesystem::path(root) / name;
        std::error_code error;
        if (!capture && !bundle_root.empty() && std::filesystem::is_directory(std::filesystem::path(bundle_root) / name, error))
            archive_directory_ = std::filesystem::path(bundle_root) / name;
        if (!capture) {
            // Lookup only: the archives an offline build (or a capturing
            // session) wrote for this package, OS build and GPU. Without one,
            // every pipeline compiles and counts as an archive miss.
            archive_lookup_.store(true, std::memory_order_release);
            if (!std::filesystem::is_directory(archive_directory_, error)) {
                std::size_t others = 0;
                for (const auto& entry : std::filesystem::directory_iterator(root, error))
                    if (entry.is_directory() && entry.path().filename().string().rfind("archive-", 0) == 0) ++others;
                armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: no pipeline archive for this package, OS build and GPU in %s "
                                     "(%zu other archive directories); pipelines compile at first use\n",
                             root.c_str(), others);
                archive_directory_.clear();
                return;
            }
            // Verified and opened in the background: checking a large
            // archive's digest must not delay the first frame. The first
            // pipeline lookup waits for it.
            archives_pending_.store(true, std::memory_order_release);
            archive_loader_ = std::thread([this, directory = archive_directory_] {
                const auto begin = timeline_now_ns();
                auto opened = open_archives(device_, directory);
                for (const auto& path : opened.rejected)
                    armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: ignoring pipeline archive %s\n", path.c_str());
                if (opened.archives.count)
                    armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: pipeline archive: %zu files loaded from %s in %.2f s\n",
                                 opened.loaded.size(), directory.c_str(), (timeline_now_ns() - begin) / 1e9);
                else
                    armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: no usable pipeline archive in %s; pipelines compile at "
                                         "first use\n", directory.c_str());
                {
                    const std::lock_guard lock(archive_load_mutex_);
                    if (opened.archives.count) lookup_archives_ = opened.archives;
                    archives_loaded_.store(true, std::memory_order_release);
                }
                archive_load_changed_.notify_all();
            });
            return;
        }
        std::filesystem::create_directories(archive_directory_, error);
        if (error) {
            armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: pipeline archive disabled: cannot create %s: %s\n",
                         archive_directory_.c_str(), error.message().c_str());
            archive_directory_.clear();
            return;
        }
        auto opened = open_archives(device_, archive_directory_);
        for (const auto& path : opened.rejected)
            armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: ignoring pipeline archive %s\n", path.c_str());
        loaded_archives_ = std::move(opened.loaded);
        NSMutableArray* archives = opened.archives;
        auto* serializer_descriptor = [MTL4PipelineDataSetSerializerDescriptor new];
        serializer_descriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
        serializer_ = [device_ newPipelineDataSetSerializerWithDescriptor:serializer_descriptor];
        if (!serializer_) {
            armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: pipeline archive disabled: no pipeline data set serializer\n");
            archive_directory_.clear();
            return;
        }
        auto* compiler_descriptor = [MTL4CompilerDescriptor new];
        compiler_descriptor.pipelineDataSetSerializer = serializer_;
        NSError* compiler_error = nil;
        if (id<MTL4Compiler> compiler = [device_ newCompilerWithDescriptor:compiler_descriptor error:&compiler_error])
            compiler_ = compiler;
        else {
            armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: pipeline archive disabled: serializing compiler failed: %s\n",
                         compiler_error ? compiler_error.localizedDescription.UTF8String : "unknown Metal error");
            serializer_ = nil;
            archive_directory_.clear();
            return;
        }
        if (archives.count) {
            task_options_ = [MTL4CompilerTaskOptions new];
            task_options_.lookupArchives = archives;
        }
        {
            const std::lock_guard lock(archive_exit_mutex());
            archive_exit_target() = this;
        }
        static std::once_flag registered;
        std::call_once(registered, [] {
            std::atexit([] {
                const std::lock_guard lock(archive_exit_mutex());
                if (auto* self = archive_exit_target()) self->flush_archive(true);
            });
        });
    }

    // Writes the pipelines captured since the last flush as a new archive file.
    // A session whose warm-up finished has rebuilt everything the older files
    // hold that the manifests name, so once it flushes at exit they are
    // dropped; otherwise they stay. It waits for capturing compiles in flight.
    void flush_archive(bool at_exit) {
        if (!serializer_ || archive_directory_.empty()) return;
        const std::unique_lock capture_lock(capture_mutex_);
        const auto captured = captures_since_flush_.exchange(0);
        const std::lock_guard lock(archive_mutex_);
        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        const auto tag = "pipelines-" + std::to_string(now);
        const auto file = archive_directory_ / (tag + ".partial");
        NSError* error = nil;
        const auto begin = timeline_now_ns();
        BOOL written = [serializer_ serializeAsArchiveAndFlushToURL:
            [NSURL fileURLWithPath:@(file.c_str())] error:&error];
        if (written) {
            const auto sealed = seal_archive(file, tag);
            written = !sealed.empty();
            if (written) last_archive_ = sealed.string();
        }
        timeline_event("pso_archive_flush", written, timeline_now_ns() - begin);
        // A failure here is mostly "Nothing to serialize": nothing was compiled since the last flush.
        if (!written && captured) {
            armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: pipeline archive not written: %s\n",
                         error ? error.localizedDescription.UTF8String : "the archive could not be sealed");
            archive_write_failed_ = true;
        }
        if (written) session_archives_ = true;
        const bool complete = warm_ && warm_->counts().finished;
        // A session that lost an archive file keeps the older files: they may
        // hold pipelines the lost file would have.
        if (at_exit && complete && session_archives_ && !archive_write_failed_ && !archive_consolidated_) {
            archive_consolidated_ = true;
            std::error_code ignored;
            for (const auto& old : loaded_archives_) std::filesystem::remove(old, ignored);
            loaded_archives_.clear();
        }
    }

    PipelineWarmupReport finish_pipeline_warmup() override {
        PipelineWarmupReport report;
        if (warm_) {
            warm_->wait_done();
            const auto counts = warm_->counts();
            report.rows = counts.rows;
            report.built = counts.built;
            report.unresolved = counts.unresolved;
            report.undecodable = counts.undecodable;
            report.failed = counts.failed;
            report.finished = counts.finished;
        } else {
            // A capturing session without a warm-up writes what it compiled.
            if (serializer_) flush_archive(false);
            report.finished = true;
        }
        report.archive_hits = archive_hits_.load();
        report.archive_open_hits = archive_open_hits_.load();
        report.archive_misses = archive_misses_.load() + archive_warmup_misses_.load();
        {
            const std::lock_guard keys(archive_keys_mutex_);
            report.unarchived = unarchived_keys_;
            report.warmup_missed = warmup_missed_keys_;
        }
        const std::lock_guard lock(archive_mutex_);
        report.archive = last_archive_;
        report.archive_incomplete = archive_write_failed_;
        return report;
    }

    // The archives pipelines are taken from (nil for none), once loaded.
    NSArray* loaded_archives() {
        if (!archives_pending_.load(std::memory_order_acquire)) return nil;
        if (!archives_loaded_.load(std::memory_order_acquire)) {
            std::unique_lock lock(archive_load_mutex_);
            archive_load_changed_.wait(lock, [&] { return archives_loaded_.load(std::memory_order_acquire); });
        }
        return lookup_archives_;
    }

    // A pipeline the loaded archives lack is compiled; the first ones are
    // named. The warm-up's misses (from launch, in the background) and the
    // replay's (when a draw needs the pipeline) are counted apart.
    void note_archive_miss(unsigned kind, const std::string& first, const std::string& second) {
        const bool warming = PipelineWarmup::on_warmup_thread();
        auto& counter = warming ? archive_warmup_misses_ : archive_misses_;
        const auto count = counter.fetch_add(1) + 1;
        if (warming && count <= 65536) {
            const std::lock_guard lock(archive_keys_mutex_);
            warmup_missed_keys_.push_back(first + " " + second);
        }
        timeline_event("pso_archive_miss", kind + (warming ? 2 : 0), 0);
        const char* where = warming ? "warm-up" : "draw";
        if (count <= 16)
            armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: pipeline not in the archive, compiled (%s): %s %s %s (miss %llu)\n",
                         where, kind ? "compute" : "render", first.c_str(), second.c_str(),
                         static_cast<unsigned long long>(count));
        else if ((count & (count - 1)) == 0)
            armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: %llu pipelines not in the archive so far (%s)\n",
                         static_cast<unsigned long long>(count), where);
    }

    void set_scheduling_period(std::uint64_t period_ns, bool preserve_encode_spin) override {
        const std::lock_guard lock(mutex_);
        scheduling_period_ns_ = period_ns;
        preserve_encode_spin_ = preserve_encode_spin;
    }

    void update_encode_budget(std::uint64_t period_ns) {
        periodic_graphics_tick(period_ns, ReplayThread::encode, scheduling_);
    }

    void observe_encode_spin(const PendingSubmission& pending, std::uint64_t spin_ns) {
        if (scheduling_.observe_encode_spin)
            scheduling_.observe_encode_spin(scheduling_.context, pending.fence,
                                            pending.scheduling_period_ns, spin_ns);
    }

    // The guest thread that acquires window textures paces the frame: it gets
    // the same declared time budget as the graphics threads, sampled from its
    // own CPU per acquire. Without a known period it keeps its last
    // declaration rather than dropping to standard scheduling, which would put
    // it below the interactive guest threads. ARMIGO_GUEST_RT=0 leaves it on
    // ordinary scheduling.
    void acquire_tick(std::uint64_t period_ns) override {
        static const bool enabled = [] {
            const char* value = std::getenv("ARMIGO_GUEST_RT");
            return !(value && *value == '0');
        }();
        if (enabled && period_ns) periodic_graphics_tick(period_ns, ReplayThread::guest_present, scheduling_);
    }

    void finish_frame(std::uint64_t period_ns, bool preserve_encode_spin) override {
        std::lock_guard lock(mutex_);
        if (stopping_) return;
        scheduling_period_ns_ = period_ns;
        preserve_encode_spin_ = preserve_encode_spin;
        queued_.push_back({0, 0, {}, period_ns, preserve_encode_spin, true, next_fence_ - 1});
        queued_ready_.notify_one();
    }

    // Pin the lists and snapshot deferred bindings on the guest thread. The
    // replay worker reads ordinary records in place through fence retirement.
    std::uint64_t submit(std::span<const NVNcommandHandle> handles) override {
        return submit_from(0, handles);
    }

    std::uint64_t submit_from(std::uintptr_t queue, std::span<const NVNcommandHandle> handles) override {
        if (!accepting_submissions_.load(std::memory_order_acquire)) {
            std::lock_guard lock(mutex_);
            last_error_ = {ReplayErrorCode::invalid_command, "Replay submission admission is closed."};
            return 0;
        }
        const auto timeline_begin = timeline_now_ns();
        const auto cpu_start = thread_cpu_ns();
        if (!replay_reported_.exchange(true, std::memory_order_relaxed))
            armigo::diag::write(armigo::diag::Level::info, core_replay()
                ? "Armigo NVN: the core replay runs this title's submissions\n"
                : "Armigo NVN: the standard replay runs this title's submissions\n");
        CommandWalkStatus status;
        auto captured = prepare_commands(handles, status);
        if (!captured) {
            std::lock_guard lock(mutex_);
            last_error_ = {ReplayErrorCode::invalid_command, std::string(command_walk_status_name(status))};
            return 0;
        }
        std::lock_guard lock(mutex_);
        if (stopping_ || !accepting_submissions_.load(std::memory_order_acquire)) {
            last_error_ = {ReplayErrorCode::invalid_command, "Replay submission admission is closed."};
            return 0;
        }
        // The split's unmet waits are read and their parts' holds registered
        // under the hold lock: a signal landing after the read is judged
        // against the registered hold by the value it left.
        std::unique_lock<std::mutex> held;
        // Declared after the lock so it ends first: the split counter drops
        // before the hold lock is released, also when an exception leaves.
        struct SplitScope {
            std::atomic<std::size_t>* counter = nullptr;
            ~SplitScope() { if (counter) counter->fetch_sub(1, std::memory_order_seq_cst); }
        } split_scope;
        SubmissionSplit split;
        if (submission_has_event_waits(*captured)) {
            held = std::unique_lock(hold_mutex_);
            splitting_.fetch_add(1, std::memory_order_seq_cst);
            split_scope.counter = &splitting_;
            split = split_at_unmet_waits(*captured);
            if (scheduling_.observe_split && !split.starts.empty()) scheduling_.observe_split(scheduling_.context);
        }
        std::uint64_t fence = 0;
        if (split.starts.empty()) {
            fence = next_fence_++;
            register_native_producer(queue, fence);
            fence_commands(*captured, fence);
            queued_.push_back({fence, queue, std::move(captured), scheduling_period_ns_, preserve_encode_spin_});
        } else {
            // Each part is queued as a submission of its own; the guest's
            // fence is the last part's, and a sync the submission fences
            // takes the fence of the part that records it.
            if (!split_reported_) {
                split_reported_ = true;
                armigo::diag::write(armigo::diag::Level::info,
                    "Armigo NVN: a submission waits for an event other work has not signalled yet; its queue "
                    "is held at the wait and the rest of the submission runs once the event is signalled\n");
            }
            std::size_t first = 0, sync = 0;
            for (std::size_t part = 0; part <= split.starts.size(); ++part) {
                const auto end = part < split.starts.size() ? split.starts[part] : SIZE_MAX;
                fence = next_fence_++;
                register_native_producer(queue, fence);
                for (; sync < split.syncs.size() && split.syncs[sync].first < end; ++sync) {
                    split.syncs[sync].second->fence = fence;
                    split.syncs[sync].second->fenced = 1;
                }
                QueuedSubmission item{fence, queue, captured, scheduling_period_ns_, preserve_encode_spin_};
                item.first_record = first;
                item.end_record = end;
                // A part that begins at an unmet wait is held from now on, so
                // every value a signal leaves before its queue reaches it is
                // seen.
                if (part) {
                    QueueHold hold{fence, leading_waits(*captured, first), 0};
                    if (!advance_hold(hold)) {
                        holds_.emplace(fence, std::move(hold));
                        hold_count_.store(holds_.size(), std::memory_order_release);
                    } else {
                        item.passed_waits = hold.waits.size();
                    }
                }
                queued_.push_back(std::move(item));
                first = end;
            }
        }
        if (held) {
            std::exchange(split_scope.counter, nullptr)->fetch_sub(1, std::memory_order_seq_cst);
            held.unlock();
        }
        schedule_generation_.fetch_add(1, std::memory_order_release);
        queued_ready_.notify_one();
        submit_cpu_ns_ += thread_cpu_ns() - cpu_start;
        timeline_event("submit", fence, timeline_now_ns() - timeline_begin);
        return fence;
    }

    std::optional<std::uint64_t> queue_wait_fence(std::uintptr_t queue, std::uint64_t fence) override {
        if (fence_completed(fence)) return 0;
        std::lock_guard lock(mutex_);
        if (stopping_ || fence >= next_fence_) return std::nullopt;
        QueuedSubmission item{};
        item.fence = next_fence_++;
        register_native_producer(queue, item.fence);
        item.queue = queue;
        item.wait_on = fence;
        const auto tail = item.fence;
        queued_.push_back(std::move(item));
        schedule_generation_.fetch_add(1, std::memory_order_release);
        queued_ready_.notify_one();
        return tail;
    }

    // A queued wait passed: its fence completes in encoding order, after
    // everything encoded before it (its queue's earlier work among it).
    void complete_queue_wait(std::uint64_t fence) {
        auto pending = std::make_shared<PendingSubmission>();
        pending->fence = fence;
        pending->slot = max_submission_slots;
        std::unique_lock lock(fence_mutex_);
        pending->sequence = ++encode_sequence_;
        inflight_[fence] = pending;
        finished_[fence] = true;
        retire_finished();
        publish_retired(lock);
    }

    void forget_queue(std::uintptr_t queue) override {
        std::lock_guard lock(mutex_);
        if (stopping_) return;
        QueuedSubmission item{};
        item.queue = queue;
        item.forget_queue = true;
        queued_.push_back(std::move(item));
        schedule_generation_.fetch_add(1, std::memory_order_release);
        queued_ready_.notify_one();
    }

    // Drops a finalized queue's replay state (worker only): a queue
    // initialized later starts from the defaults.
    void forget_queue_state(std::uintptr_t queue) {
        {
            std::lock_guard lock(native_results_mutex_);
            native_latest_.erase(queue);
            collect_native_results_locked();
        }
        queue_states_.erase(queue);
        if (state_queue_ == queue) {
            state_ = initial_state_;
            state_queue_ = no_state_queue;
        }
    }

    void run_worker() {
        pthread_setname_np("nvn-replay");
        // This worker feeds the GPU queue and has the same latency needs as
        // the encode thread, independently of the submitting thread's QoS.
        const int qos_status = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        if (qos_status)
            armigo::diag::write(armigo::diag::Level::error,  "nvn_replay: replay worker QoS declaration failed: %d\n", qos_status);
        for (;;) {
            QueuedSubmission item;
            std::size_t passed_waits = 0;
            bool abandoned = false;
            {
                std::unique_lock lock(mutex_);
                for (;;) {
                    queued_ready_.wait(lock, [&] { return stopping_ || !queued_.empty(); });
                    if (queued_.empty()) {
                        worker_stopped_ = true;
                        worker_exited_.notify_all();
                        return;
                    }
                    const auto generation = schedule_generation_.load(std::memory_order_acquire);
                    if (pick_locked(item, passed_waits)) break;
                    // Shutdown abandons held submissions so their fences complete.
                    if (stopping_) {
                        item = std::move(queued_.front());
                        queued_.pop_front();
                        abandoned = true;
                        break;
                    }
                    report_hold_stall();
                    // A release or a new submission bumps the generation; the
                    // timeout covers a release that lands before this wait.
                    queued_ready_.wait_for(lock, std::chrono::milliseconds(2), [&] {
                        return stopping_ || schedule_generation_.load(std::memory_order_acquire) != generation;
                    });
                }
                active_cpu_fence_ = item.fence;
                active_cpu_ready_ = item.cpu_ready;
            }
            struct CpuWalkDone {
                ReplayImpl& backend;
                std::shared_ptr<std::atomic<bool>> ready;
                ~CpuWalkDone() {
                    ready->store(true, std::memory_order_release);
                    {
                        std::lock_guard lock(backend.mutex_);
                        backend.active_cpu_fence_ = 0;
                        backend.active_cpu_ready_.reset();
                        backend.schedule_generation_.fetch_add(1, std::memory_order_release);
                    }
                    backend.queued_ready_.notify_all();
                }
            } cpu_walk_done{*this, item.cpu_ready};
            if (item.capture) {
                @autoreleasepool {
                    try { item.capture(!abandoned && accepting_submissions_.load(std::memory_order_acquire)); }
                    catch (...) {
                        armigo::diag::write(armigo::diag::Level::error, "nvn_replay: native frame capture failed\n");
                        try { item.capture(false); } catch (...) {}
                    }
                }
                continue;
            }
            if (abandoned) {
                {
                    std::lock_guard hold(hold_mutex_);
                    if (const auto found = holds_.find(item.fence); found != holds_.end()) {
                        holds_.erase(found);
                        hold_count_.store(holds_.size(), std::memory_order_release);
                    }
                }
                if (!item.frame_boundary && !item.forget_queue) finish_failed(item.fence, "Replay submission was abandoned during shutdown");
                continue;
            }
            if (item.forget_queue) {
                forget_queue_state(item.queue);
                continue;
            }
            if (item.wait_on) {
                complete_queue_wait(item.fence);
                continue;
            }
            if (item.frame_boundary) {
                pool_tables_.note_present();
                if (gpu_profile_) gpu_profile_->frame_boundary(last_encoded_fence_, completed_fence());
                if (gpu_frame_work_enabled_.load(std::memory_order_acquire)) {
                    std::lock_guard timing_lock(fence_mutex_);
                    gpu_frame_work_.frame_boundary(item.frame_end_fence);
                }
                periodic_graphics_tick(item.scheduling_period_ns, ReplayThread::worker, scheduling_);
                continue;
            }
            // Encoder and descriptor temporaries must retire with each replay
            // submission, even though the worker thread itself is long-lived.
            const auto cpu_start = thread_cpu_ns();
            @autoreleasepool {
                encode(item.fence, std::move(item.captured), item.scheduling_period_ns,
                       item.preserve_encode_spin, passed_waits, item.queue, item.first_record, item.end_record);
            }
            // Draining the submission's autoreleased objects is part of the
            // replay's per-frame cost and happens outside encode().
            if (encoding_submission_) {
                auto pending = std::move(encoding_submission_);
                const auto pool_end = timeline_now_ns();
                pending->phases.pool = pending->phases.encode_end
                    ? pool_end - pending->phases.encode_end : 0;
                statistics_.encode_cpu_ns += thread_cpu_ns() - cpu_start;
                publish_encoder();
                seal_submission(pending);
            } else {
                statistics_.encode_cpu_ns += thread_cpu_ns() - cpu_start;
                publish_encoder();
            }
        }
    }

    // Only short publication sections share the caller's mutex. The worker's
    // mutable statistics, errors and caches never belong to a feedback callback.
    void publish_encoder() {
        std::lock_guard lock(mutex_);
        published_statistics_ = statistics_;
        published_skips_ = skips_;
        if (encoder_error_) last_error_ = std::exchange(encoder_error_, {});
    }

    // Everything the retired submission's diagnostics need, captured under
    // fence_mutex_ so the lines themselves are written after the guest is woken.
    struct RetiredLog {
        std::shared_ptr<PendingSubmission> pending;
        std::uint64_t fence, lookups, misses;
    };

    // With ARMIGO_NVN_COUNT=1 the driver counts calls per entry point; every
    // 2000 submissions the log names the entry points the guest called since the
    // previous report, most called first.
    static void log_call_counts(std::uint64_t fence) {
        static std::mutex count_mutex;
        const std::lock_guard lock(count_mutex);
        static std::vector<nvn::CallCount> previous;
        std::vector<nvn::CallCount> current(previous.empty() ? 1024 : previous.size());
        current.resize(std::min(current.size(), nvn::call_counts(current)));
        if (previous.size() != current.size()) previous.assign(current.size(), {});
        std::vector<std::pair<std::uint64_t, std::string_view>> deltas;
        for (std::size_t i = 0; i < current.size(); ++i)
            if (current[i].calls > previous[i].calls)
                deltas.emplace_back(current[i].calls - previous[i].calls, current[i].name);
        previous = current;
        std::sort(deltas.begin(), deltas.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::string line = "nvn_calls fence=" + std::to_string(fence);
        for (const auto& [calls, name] : deltas) line += " " + std::string(name) + "=" + std::to_string(calls);
        armigo::diag::opt_in( "%s\n", line.c_str());
    }

    static void log_retired(const RetiredLog& entry) {
        const auto fence = entry.fence;
        static const bool counting = [] {
            const char* value = std::getenv("ARMIGO_NVN_COUNT");
            return value && std::string_view(value) == "1";
        }();
        if (counting && fence % 2000 == 0) log_call_counts(fence);
        const auto& c = entry.pending->counters;
        const auto& phases = entry.pending->phases;
        const auto& pending = *entry.pending;
        armigo::diag::frame(
                     "nvn_work frame=%llu draws=%llu passes=%llu report_fast=%llu report_ordered=%llu overlay_calls=%llu "
                     "buffer_barriers=%llu overlay_bytes=%llu metal_binds=%llu binds_vertex=%llu binds_buffer=%llu binds_texture=%llu binds_sampler=%llu window_uploads=%llu window_reuses=%llu pipeline_hashes=%llu "
                     "pipeline_hits=%llu uniform_updates=%llu fills=%llu fill_bytes=%llu "
                     "versions=%llu walk_us=%llu prep_us=%llu fin_us=%llu commit_us=%llu pool_us=%llu "
                     "segments=%llu gpu_interval_us=%llu timed_segments=%llu first_commit_us=%llu final_commit_us=%llu "
                     "guest_waits=%llu\n",
                     static_cast<unsigned long long>(fence),
                     static_cast<unsigned long long>(c.draws),
                     static_cast<unsigned long long>(c.passes),
                     static_cast<unsigned long long>(c.report_fast),
                     static_cast<unsigned long long>(c.report_ordered),
                     static_cast<unsigned long long>(c.overlay_calls),
                     static_cast<unsigned long long>(c.buffer_barriers),
                     static_cast<unsigned long long>(c.overlay_bytes),
                     static_cast<unsigned long long>(c.metal_binds),
                     static_cast<unsigned long long>(c.binds_vertex),
                     static_cast<unsigned long long>(c.binds_buffer),
                     static_cast<unsigned long long>(c.binds_texture),
                     static_cast<unsigned long long>(c.binds_sampler),
                     static_cast<unsigned long long>(c.window_uploads),
                     static_cast<unsigned long long>(c.window_reuses),
                     static_cast<unsigned long long>(c.pipeline_hashes),
                     static_cast<unsigned long long>(c.pipeline_hits),
                     static_cast<unsigned long long>(c.uniform_updates),
                     static_cast<unsigned long long>(c.fill_calls),
                     static_cast<unsigned long long>(c.fill_bytes),
                     static_cast<unsigned long long>(c.version_creations),
                     static_cast<unsigned long long>(phases.walk / 1000),
                     static_cast<unsigned long long>(phases.preamble / 1000),
                     static_cast<unsigned long long>(phases.finish / 1000),
                     static_cast<unsigned long long>(phases.commit / 1000),
                     static_cast<unsigned long long>(phases.pool / 1000),
                     static_cast<unsigned long long>(pending.segments.submitted),
                     static_cast<unsigned long long>(pending.gpu_interval_ns / 1000),
                     static_cast<unsigned long long>(pending.timed_segments),
                     static_cast<unsigned long long>(pending.first_commit_ns
                         ? (pending.first_commit_ns - pending.encode_begin_ns) / 1000 : 0),
                     static_cast<unsigned long long>(pending.final_commit_ns
                         ? (pending.final_commit_ns - pending.encode_begin_ns) / 1000 : 0),
                     static_cast<unsigned long long>(c.guest_waits));
        armigo::diag::frame( "nvn_textures frame=%llu lookups=%llu misses=%llu stage_binds=%llu "
                             "stage_reuses=%llu syncs_registration=%llu syncs_writes=%llu "
                             "uniform_invalidations=%llu stage_uniform_binds=%llu\n",
                     static_cast<unsigned long long>(fence),
                     static_cast<unsigned long long>(entry.lookups),
                     static_cast<unsigned long long>(entry.misses),
                     static_cast<unsigned long long>(c.stage_binds),
                     static_cast<unsigned long long>(c.stage_reuses),
                     static_cast<unsigned long long>(c.syncs_registration),
                     static_cast<unsigned long long>(c.syncs_writes),
                     static_cast<unsigned long long>(c.uniform_invalidations),
                     static_cast<unsigned long long>(c.stage_uniform_binds));
    }

    // Called with fence_mutex_ held; the caller then calls publish_retired().
    // Effects and resource retirement follow fence order even if feedback is
    // delivered out of order.
    void retire_finished() {
        // Effects follow encoding order: a finished submission retires once no
        // submission encoded before it is still running. A fence the worker
        // passed over (a held queue) is not encoded and holds nothing back;
        // once encoded, its effects follow those encoded before it.
        std::uint64_t limit = UINT64_MAX;
        for (const auto& [fence, pending] : inflight_)
            if (!finished_.contains(fence)) limit = std::min(limit, pending->sequence);
        const bool frame_work = gpu_frame_work_enabled_.load(std::memory_order_relaxed);
        // Finished submissions encoded before every running one, in encoding
        // order (a submission that failed before encoding has no effects).
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ready;
        for (const auto& [fence, done] : finished_) {
            const auto it = inflight_.find(fence);
            const auto sequence = it == inflight_.end() ? 0 : it->second->sequence;
            if (sequence < limit) ready.emplace_back(sequence, fence);
        }
        std::ranges::sort(ready);
        for (const auto& [sequence, fence] : ready) {
            finished_.erase(fence);
            const auto work = retire_one(fence);
            if (frame_work && applied_work_.size() < retired_work_limit) applied_work_[fence] = work;
            note_retired(fence);
        }
        // Every 64 retirements, allocations no running submission uses and
        // none used for a while leave the shared residency set.
        if (!ready.empty() && (residency_sweep_retirements_ += ready.size()) >= 64) {
            residency_sweep_retirements_ = 0;
            std::uint64_t oldest = encode_sequence_ + 1;
            for (const auto& [fence, pending] : inflight_)
                oldest = std::min(oldest, pending->first_sequence ? pending->first_sequence : pending->sequence);
            shared_residency_.sweep(oldest, encode_sequence_);
        }
        // The published fence advances only over contiguous retired fences;
        // frame work follows that order.
        for (auto range = retired_.find(completed_fence_ + 1); range != retired_.end();
             range = retired_.find(completed_fence_ + 1)) {
            const auto last = range->second;
            if (frame_work) {
                for (auto fence = range->first; fence <= last; ++fence) {
                    std::optional<std::uint64_t> work;
                    if (const auto it = applied_work_.find(fence); it != applied_work_.end()) {
                        work = it->second;
                        applied_work_.erase(it);
                    }
                    gpu_frame_work_.retire(fence, work);
                }
            }
            retired_.erase(range);
            completed_fence_ = last;
            published_completed_fence_.store(last, std::memory_order_release);
        }
        if (!frame_work) applied_work_.clear();
    }

    // Adds a retired fence to the ranges not yet published (fence_mutex_
    // held): one range per run between held fences, not an entry per fence.
    void note_retired(std::uint64_t fence) {
        auto next = retired_.lower_bound(fence);
        if (next != retired_.begin()) {
            if (auto previous = std::prev(next); previous->second + 1 == fence) {
                previous->second = fence;
                if (next != retired_.end() && next->first == fence + 1) {
                    previous->second = next->second;
                    retired_.erase(next);
                }
                return;
            }
        }
        if (next != retired_.end() && next->first == fence + 1) {
            const auto last = next->second;
            retired_.erase(next);
            retired_.emplace(fence, last);
            return;
        }
        retired_.emplace(fence, fence);
    }

    // Applies one finished fence's effects and retires its submission; returns
    // its GPU work for the frame-work collector.
    std::optional<std::uint64_t> retire_one(std::uint64_t fence) {
        std::optional<std::uint64_t> frame_work;
        std::string producer_error;
        if (const auto it = inflight_.find(fence); it != inflight_.end()) {
            if (it->second->execution_error) {
                producer_error = it->second->execution_error.detail;
                execution_error_ = it->second->execution_error;
                // Its report effects are skipped: their counters never land.
                if (it->second->replay_conditionals) reports_.set_failed(it->second->sequence);
            } else {
                apply_effects_locked(*it->second, true);
            }
            // Publish the signals before retiring their writer records. A
            // held queue must never see the older CPU value between them.
            std::size_t retired_writes = 0;
            {
                std::lock_guard lock(encoded_writes_mutex_);
                retired_writes = std::erase_if(encoded_writes_, [&](const auto& write) {
                    return write.fence == fence;
                });
            }
            if (retired_writes) {
                // A wait registered after a signal was published may still
                // have seen its pending ADD. Recheck the now-published word.
                bool released = false;
                {
                    std::lock_guard hold(hold_mutex_);
                    for (auto& [held_fence, held] : holds_)
                        released = advance_hold(held) || released;
                }
                if (released) {
                    schedule_generation_.fetch_add(1, std::memory_order_release);
                    queued_ready_.notify_all();
                }
            }
            report_guest_misses(*it->second);
            report_task_launches(*it->second);
            if (it->second->encode_warning) encode_warning_ = it->second->encode_warning;
            // Completing the writes ends the submission's pool leases: its
            // logged writes reach the resources first, while a release of
            // those pools still waits for them.
            publish_guest_writes(*it->second, false);
            resources_->complete_gpu_writes(it->second->resource_write_generation);
            if (!it->second->segment_labels.empty()) note_dispatch_profile(*it->second);
            const auto lookups = resources_->texture_lookup_counts();
            retired_log_.push_back({it->second, fence, lookups.lookups - reported_lookups_.lookups,
                                    lookups.misses - reported_lookups_.misses});
            reported_lookups_ = lookups;
            if (!it->second->execution_error && it->second->segments.submitted &&
                it->second->timed_segments == it->second->segments.submitted && it->second->gpu_first_ns)
                frame_work = it->second->gpu_interval_ns;
            if (it->second->slot < max_submission_slots) free_slots_ |= 1u << it->second->slot;
            // Its reports were read above: the heaps go back to the pool.
            for (id<MTL4CounterHeap> heap : it->second->timestamp_heaps) timestamp_pool_.push_back(heap);
            it->second->timestamp_heaps.clear();
            it->second->retired.store(true, std::memory_order_release);
            inflight_.erase(it);
        }
        finish_native_producer(fence, std::move(producer_error));
        return frame_work;
    }

    // Wakes the fence waiters, releases fence_mutex_, then writes the retired
    // submissions' diagnostics: the guest never waits for formatting. A woken
    // waiter may destroy the backend once the lock is free, so nothing after
    // the unlock touches it: the lines come from the submissions alone.
    void publish_retired(std::unique_lock<std::mutex>& lock) {
        // Register the tail before releasing fence_mutex_ or waking waiters.
        // Construct the ticket first so it outlives the retired payloads too.
        RetirementTailTicket ticket(retirement_tails_);
        auto retired = std::move(retired_log_);
        retired_log_.clear();
        const auto observe = scheduling_.observe_retired_log;
        auto* context = scheduling_.context;
        completion_.notify_all();
        lock.unlock();
        for (const auto& entry : retired) {
            if (observe) observe(context, entry.fence);
            log_retired(entry);
            // The lines and their pixel buffers outlive this backend. Nothing
            // here touches the replay after the lock was released.
            if (entry.pending && !entry.pending->pass_dumps.empty())
                publish_pass_dumps(std::move(entry.pending->pass_dumps));
        }
    }

    // Returns a slot a submission took but never put in flight.
    void release_slot(std::size_t slot) {
        std::lock_guard lock(fence_mutex_);
        free_slots_ |= 1u << slot;
    }

    // A submission that failed to encode completes in order with no GPU work.
    void finish_failed(std::uint64_t fence, std::string producer_error) {
        {
            std::lock_guard lock(native_results_mutex_);
            if (const auto found = native_results_.find(fence); found != native_results_.end())
                found->second.preflight_error = std::move(producer_error);
        }
        publish_encoder();
        std::unique_lock lock(fence_mutex_);
        finished_[fence] = true;
        retire_finished();
        publish_retired(lock);
    }

    void encode(std::uint64_t fence, std::shared_ptr<const CommandSubmission> captured,
                std::uint64_t scheduling_period_ns, bool preserve_encode_spin, std::size_t passed_waits,
                std::uintptr_t queue, std::size_t first_record, std::size_t end_record) {
        const auto phase_start = timeline_now_ns();
        // The worker owns encoding. Only a full ring waits: the submission takes
        // a slot no encoded, unretired submission holds (a held one takes none
        // until it is encoded), and no encoder or publication lock is held
        // while GPU completion makes a slot available.
        const auto ring_wait_begin = timeline_now_ns();
        std::size_t slot_index = 0;
        {
            std::unique_lock lock(fence_mutex_);
            if (!slots_sized_) {
                free_slots_ = (1u << submission_slot_count()) - 1;
                slots_sized_ = true;
            }
            completion_.wait(lock, [&] { return free_slots_ != 0; });
            slot_index = static_cast<std::size_t>(std::countr_zero(free_slots_));
            free_slots_ &= ~(1u << slot_index);
        }
        timeline_event("ring_wait", fence, timeline_now_ns() - ring_wait_begin);
        timeline_event("encode_begin", fence, 0);
        const auto encode_start = std::chrono::steady_clock::now();
        if (!compiler_) {
            const ReplayError failure{ReplayErrorCode::allocation, "Metal replay compiler is unavailable"};
            encoder_error_ = failure;
            release_slot(slot_index);
            finish_failed(fence, failure.detail);
            return;
        }
        auto pending = std::make_shared<PendingSubmission>();
        pending->fence = fence;
        pending->slot = slot_index;
        pending->scheduling_period_ns = scheduling_period_ns;
        pending->preserve_encode_spin = preserve_encode_spin;
        pending->encode_begin_ns = phase_start;
        pending->captured = std::move(captured);
        pending->first_record = first_record;
        pending->end_record = end_record;
        // Each NVN queue keeps its own draw state: another queue's submission
        // may run between two parts of a submission, or between two of a
        // queue's submissions while it is held. A queue starts with defaults.
        if (queue != state_queue_) {
            if (state_queue_ != no_state_queue) queue_states_[state_queue_] = std::move(state_);
            if (const auto found = queue_states_.find(queue); found != queue_states_.end()) {
                state_ = std::move(found->second);
                queue_states_.erase(found);
            } else {
                state_ = initial_state_;
            }
            state_queue_ = queue;
        }
        auto& slot = submission_storage_[slot_index];
        if (!slot.allocator) slot.allocator = [device_ newCommandAllocator];
        else [slot.allocator reset];
        pending->allocator = slot.allocator;
        pending->encode_thread = encode_thread_.get();
        if (!shared_residency_.ready()) shared_residency_.create(device_);
        pending->shared_residency = &shared_residency_;
        pending->count_residency = residency_count_log_ || scheduling_.observe_residency_sets;
        // Reuse query storage only after the slot's previous fence retired.
        if (!slot.visibility) {
            slot.visibility = [device_ newBufferWithLength:64 * 1024
                                                    options:MTLResourceStorageModeShared];
            ++statistics_.new_buffer_allocations;
        }
        pending->visibility = slot.visibility;
        // Without the encode thread the command buffer is created here; with
        // it, no record names the submission until every allocation exists.
        if (!pending->allocator || !shared_residency_.ready() || !pending->visibility ||
            (!encode_thread_ && !begin_commands(*pending))) {
            const ReplayError failure{ReplayErrorCode::allocation, "Metal replay submission allocation failed"};
            encoder_error_ = failure;
            release_slot(slot_index);
            finish_failed(fence, failure.detail);
            return;
        }
        std::memset(pending->visibility.contents, 0, pending->visibility.length);
        [pending->retained addObjectsFromArray:@[pending->allocator, pending->visibility]];
        if (encode_thread_) begin_commands(*pending);
        encoding_submission_ = pending;
        {
            std::lock_guard lock(fence_mutex_);
            // Validate the carry while publishing this part as in flight:
            // retirement must not sweep its allocations between those steps.
            pending->carry = residency_carry(pending->captured);
            pending->sequence = ++encode_sequence_;
            if (!pending->carry->first_sequence) pending->carry->first_sequence = pending->sequence;
            pending->first_sequence = pending->carry->first_sequence;
            inflight_[fence] = pending;
        }
        // Resident from a submission in flight: the shared set keeps it until
        // that submission retired.
        make_resident(*pending, pending->visibility);

        std::vector<std::shared_ptr<PendingSubmission>> released_submissions;
        resources_->begin_walk();
        struct EndResourceWalk {
            ReplayResources& resources;
            ~EndResourceWalk() { resources.end_walk(); }
        } end_resource_walk{*resources_};
        // Held queues are encoded out of guest-fence order. The report table
        // follows encode order, which is the order completion effects land.
        const bool replay_conditionals =
            nvn::title_policy().conditional_render == nvn::TitlePolicy::ConditionalRender::replay;
        pending->replay_conditionals = replay_conditionals;
        Encoder encoder(*this, *pending, state_, resources_->buffer_addresses_are_cpu(),
                        replay_conditionals ? &reports_ : nullptr);
        encoder.retired_fence = completed_fence();
        encoder.passed_leading_waits = passed_waits;
        // Releasing a retired submission frees its command buffers, allocators
        // and resources, which takes longer than the rest of the preamble. The
        // references are dropped when this function returns, after the commits.
        // A submission leaves this history when it retires, also out of fence
        // order while a held queue keeps the published fence back.
        for (auto it = previous_submissions_.begin(); it != previous_submissions_.end();) {
            if ((*it)->retired.load(std::memory_order_acquire)) {
                released_submissions.push_back(std::move(*it));
                it = previous_submissions_.erase(it);
            } else {
                ++it;
            }
        }
        encoder.previous_submissions.assign(previous_submissions_.begin(), previous_submissions_.end());
        statistics_.unretired_history = previous_submissions_.size();
        for (const auto& previous : encoder.previous_submissions) {
            if (previous->guest_writes) encoder.prior_guest_writer_fence = previous->fence;
            for (const auto& update : previous->uniform_updates)
                encoder.note_update_pages(update.address, update.size);
        }
        pending->phases.preamble = timeline_now_ns() - phase_start;
        const auto phase_walk = timeline_now_ns();
        EncodingScope scope(encoder);
        // A part of a split submission runs only its own records.
        struct RangeWalk {
            Encoder* encoder;
            std::size_t index, first, end;
        } range{&encoder, 0, pending->first_record, pending->end_record};
        const auto walk_status = walk_commands(*pending->captured,
            [](const CommandRecordView& record, void* context) {
                auto& range = *static_cast<RangeWalk*>(context);
                const auto at = range.index++;
                return at < range.first || at >= range.end || range.encoder->visit(record);
            }, &range);
        if (walk_status != CommandWalkStatus::complete && !encoder.failed) {
            encoder.failed = true;
            encoder.error = {ReplayErrorCode::invalid_command, std::string(command_walk_status_name(walk_status))};
        }
        pending->phases.walk = timeline_now_ns() - phase_walk;
        resident_hint_ = std::max(resident_hint_, pending->carry->residents.count);
        const auto phase_finish = timeline_now_ns();
        if (!encoder.failed && !encoder.finish()) {
            // finish records its exact error.
        }
        pending->resource_write_generation = resources_->gpu_write_generation();
        if (encoder.failed) {
            // A committed prefix owns this guest fence until all feedback has
            // arrived. Never recycle its allocator through finish_failed().
            const auto failure = encoder.error;
            {
                const std::lock_guard lock(fence_mutex_);
                if (!abandoned_error_) {
                    abandoned_error_ = failure;
                    abandoned_fence_ = fence;
                }
            }
            accepting_submissions_.store(false, std::memory_order_release);
            completion_.notify_all();
            encoder.drain_failed_prefix();
            encoder_error_ = failure;
            pending->phases.finish = timeline_now_ns() - phase_finish;
            pending->phases.encode_end = timeline_now_ns();
            return;
        }
        state_ = std::move(encoder.state);
        pending->uniform_updates = std::move(encoder.uniform_updates);
        pending->other_buffer_writes = std::move(encoder.other_buffer_writes);
        for (const auto& effect : pending->effects) {
            if (effect.kind == CompletionEffect::Kind::report) {
                const auto begin = reinterpret_cast<std::uintptr_t>(effect.destination_cpu);
                pending->completion_writes.emplace_back(begin, begin + sizeof(NVNcounterData));
            } else if (effect.kind == CompletionEffect::Kind::signal) {
                const auto begin = reinterpret_cast<std::uintptr_t>(effect.event);
                pending->completion_writes.emplace_back(begin, begin + sizeof(std::uint32_t));
            }
        }
        for (const auto& [begin, end] : pending->completion_writes) {
            pending->completion_low = std::min(pending->completion_low, begin);
            pending->completion_high = std::max(pending->completion_high, end);
            pending->completion_ranges.add(begin, end - begin);
        }
        previous_submissions_.push_back(pending);
        for (auto* sync : pending->fenced_syncs) {
            if (!sync) continue;
            sync->fence = pending->fence;
            sync->fenced = 1;
        }
        pending->phases.finish = timeline_now_ns() - phase_finish;
        ++statistics_.submissions;
        if (pending->replay_conditionals) reports_.commit(pending->sequence);
        commit_segment(*pending, SegmentKind::final);
        last_encoded_fence_ = std::max(last_encoded_fence_, pending->fence);
        note_encoded_signals(*pending);
        statistics_.encode_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - encode_start).count());
        pending->phases.encode_end = timeline_now_ns();
        timeline_event("encode_end", fence, timeline_now_ns() - ring_wait_begin);
    }

    void flush() override {
        if (!resources_->publish_uploads())
            armigo::diag::write(armigo::diag::Level::error, "nvn_replay: texture upload publication failed\n");
    }

    // Lock-free: guests poll syncs with a zero timeout in a loop, and taking
    // fence_mutex_ for each poll contended with fence retirement. The value is
    // stored after the fence's effects were applied (release), so a reader
    // that sees a fence also sees its effects.
    std::uint64_t completed_fence() const override {
        return published_completed_fence_.load(std::memory_order_acquire);
    }

    std::uint64_t gpu_frame_work_ns() const noexcept override {
        std::lock_guard lock(fence_mutex_);
        if (!gpu_frame_work_enabled_.load(std::memory_order_relaxed)) {
            // A completed submission is not necessarily a frame boundary.
            // The collector discards the first partial frame after this reset.
            gpu_frame_work_.reset(completed_fence_);
            gpu_frame_work_enabled_.store(true, std::memory_order_release);
        }
        return gpu_frame_work_.latest().value_or(0);
    }

    // Waits until every submission encoded before `current` has applied its
    // effects (see retire_finished); work a held queue has not encoded does
    // not count. Without held queues this is waiting for fence - 1.
    void wait_prior_effects(const PendingSubmission& current) {
        std::unique_lock lock(fence_mutex_);
        completion_.wait(lock, [&] {
            return std::ranges::none_of(inflight_, [&](const auto& entry) {
                return entry.second->sequence < current.sequence;
            });
        });
    }
    // Waits until `previous`, encoded before the caller's submission, applied
    // its effects.
    // Records the signals a submission left pending once it is committed, and
    // releases the holds its writes meet. A submission that already retired
    // landed its signals itself.
    void note_encoded_signals(const PendingSubmission& pending) {
        if (event_wait_drains().release) return;
        std::vector<std::pair<const volatile std::uint32_t*, std::uint32_t>> released;
        {
            std::lock_guard fences(fence_mutex_);
            if (pending.retired.load(std::memory_order_acquire)) return;
            std::lock_guard lock(encoded_writes_mutex_);
            for (const auto& effect : pending.effects) {
                if (effect.kind != CompletionEffect::Kind::signal) continue;
                auto found = std::ranges::find(encoded_writes_, effect.event, &EncodedEventWrite::event);
                if (found == encoded_writes_.end()) found = encoded_writes_.insert(encoded_writes_.end(), {effect.event});
                *found = {effect.event, pending.fence, effect.value, !effect.add, pending.sequence};
            }
            for (const auto& write : encoded_writes_)
                if (write.fence == pending.fence && write.exact) released.emplace_back(write.event, write.value);
        }
        for (const auto& [event, value] : released) evaluate_holds(event, value);
    }

    // Whether the newest encoded, unretired signal of `event` is a write that
    // leaves a value meeting the wait. Work encoded later runs after it on
    // the one Metal queue, whose next encoder takes a queue barrier, so such
    // a wait needs no completion.
    bool encoded_write_meets(const volatile std::uint32_t* event, std::int32_t mode, std::uint32_t value) const {
        const auto write = newest_encoded_write(event);
        return write && write->exact && wait_met_by({event, mode, value}, write->value);
    }


    // The residency carry of a guest submission's parts (worker only, with
    // fence_mutex_ held until the part is in flight): reuse a live carry.
    std::shared_ptr<ResidencyCarry> residency_carry(const std::shared_ptr<const CommandSubmission>& captured) {
        if (carries_.size() > 64)
            std::erase_if(carries_, [](const auto& entry) { return entry.second.expired(); });
        auto& slot = carries_[captured.get()];
        auto carry = slot.lock();
        if (!carry || carry->submission.lock() != captured) {
            carry = std::make_shared<ResidencyCarry>();
            carry->submission = captured;
            carry->residents.reserve(resident_hint_);
            carry->removals = shared_residency_.removals();
            slot = carry;
        } else if (carry->removals != shared_residency_.removals()) {
            carry->residents = ResidentTable{};
            carry->residents.reserve(resident_hint_);
            carry->removals = shared_residency_.removals();
        }
        return carry;
    }

    id<MTLTexture> border_coverage(NSUInteger width, NSUInteger height, NSUInteger levels) {
        BorderCoverageImage* selected = nullptr;
        for (auto& entry : border_coverage_) {
            if (entry.texture && entry.texture.width == width && entry.texture.height == height &&
                entry.texture.mipmapLevelCount == levels) {
                entry.use = ++border_use_;
                return entry.texture;
            }
            if (!selected || entry.use < selected->use) selected = &entry;
        }
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
            width:width height:height mipmapped:NO];
        descriptor.mipmapLevelCount = levels;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> texture = [device_ newTextureWithDescriptor:descriptor];
        if (!texture) return nil;
        const std::vector<std::uint8_t> row(width, 255);
        for (NSUInteger mip = 0; mip != levels; ++mip) {
            const auto w = std::max<NSUInteger>(1, width >> mip), h = std::max<NSUInteger>(1, height >> mip);
            for (NSUInteger y = 0; y != h; ++y)
                [texture replaceRegion:MTLRegionMake2D(0, y, w, 1) mipmapLevel:mip
                             withBytes:row.data() bytesPerRow:w];
        }
        *selected = {texture, ++border_use_};
        return texture;
    }

    id<MTLTexture> border_identity(id<MTLTexture> original) {
        const auto swizzle = original.swizzle;
        if (swizzle.red == MTLTextureSwizzleRed && swizzle.green == MTLTextureSwizzleGreen &&
            swizzle.blue == MTLTextureSwizzleBlue && swizzle.alpha == MTLTextureSwizzleAlpha) return original;
        BorderIdentityView* selected = nullptr;
        for (auto& entry : border_identity_) {
            if (entry.source == original) { entry.use = ++border_use_; return entry.texture; }
            if (!selected || entry.use < selected->use) selected = &entry;
        }
        id<MTLTexture> root = original;
        NSUInteger level = 0, slice = 0;
        while (root.parentTexture) {
            level += root.parentRelativeLevel; slice += root.parentRelativeSlice; root = root.parentTexture;
        }
        const MTLTextureSwizzleChannels identity{MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
                                                 MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha};
        id<MTLTexture> use = [root newTextureViewWithPixelFormat:original.pixelFormat textureType:MTLTextureType2D
            levels:NSMakeRange(level, original.mipmapLevelCount) slices:NSMakeRange(slice, 1) swizzle:identity];
        if (!use) return nil;
        *selected = {original, use, ++border_use_};
        return use;
    }

    id<MTLTexture> feedback_snapshot(std::uint64_t source, NSUInteger level, NSUInteger slice,
                                    MTLPixelFormat format, NSUInteger width, NSUInteger height,
                                    MTLTextureType type, NSUInteger levels, NSUInteger array_length, bool depth,
                                    std::uint64_t submission, std::uint64_t epoch, bool& fresh) {
        const auto shape = [&](const auto& entry) {
            return entry.texture && entry.texture.pixelFormat == format &&
                entry.texture.width == width && entry.texture.height == height &&
                entry.texture.textureType == type && entry.texture.mipmapLevelCount == levels &&
                entry.texture.arrayLength == array_length;
        };
        const auto live = [&](const auto& entry) {
            return entry.source && entry.submission == submission && entry.epoch == epoch;
        };
        FeedbackSnapshot* selected = nullptr;
        for (auto& entry : feedback_snapshots_)
            if (entry.source == source && entry.level == level && entry.slice == slice) {
                if (shape(entry)) selected = &entry;
                else entry.source = 0;
                break;
            }
        if (selected && live(*selected)) {
            selected->use = ++feedback_use_;
            fresh = false;
            return selected->texture;
        }
        if (!selected) {
            int best = 4;
            for (auto& entry : feedback_snapshots_) {
                const int rank = live(entry) ? (shape(entry) ? 2 : 3) : (shape(entry) ? 0 : 1);
                if (rank < best || (rank == best && entry.use < selected->use)) {
                    selected = &entry;
                    best = rank;
                }
            }
        }
        if (!shape(*selected)) {
            auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                width:width height:height mipmapped:NO];
            descriptor.textureType = type;
            descriptor.mipmapLevelCount = levels;
            descriptor.arrayLength = array_length;
            descriptor.storageMode = MTLStorageModePrivate;
            descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
            if (!depth) descriptor.allowGPUOptimizedContents = NO;
            selected->texture = [device_ newTextureWithDescriptor:descriptor];
            if (!selected->texture) return nil;
        }
        selected->source = source;
        selected->level = level;
        selected->slice = slice;
        selected->submission = submission;
        selected->epoch = epoch;
        selected->use = ++feedback_use_;
        fresh = true;
        return selected->texture;
    }

    // The frame profiler while ARMIGO_NVN_GPUPROF is set (worker only).
    GpuProfile* gpu_profile() { return gpu_profile_.get(); }

    void wait_retired(const PendingSubmission& previous) {
        std::unique_lock lock(fence_mutex_);
        completion_.wait(lock, [&] { return previous.retired.load(std::memory_order_acquire); });
    }

    // A fence completes when its submission retired: every encoded
    // submission before it retired first, and so did its own queue's earlier
    // work, but another queue's held work may still be pending, as on the GPU.
    bool fence_completed(std::uint64_t fence) const override {
        std::lock_guard lock(fence_mutex_);
        return fence_retired_locked(fence);
    }
    bool fence_retired_locked(std::uint64_t fence) const {
        if (!fence || fence <= completed_fence_) return true;
        const auto after = retired_.upper_bound(fence);
        return after != retired_.begin() && std::prev(after)->second >= fence;
    }

    bool wait_fence(std::uint64_t fence, std::uint64_t timeout_ns) override {
        std::unique_lock lock(fence_mutex_);
        if (abandoned_error_) return false;
        if (fence_retired_locked(fence)) return true;
        if (fence >= next_fence_) return false;
        const auto ready = [&] { return fence_retired_locked(fence) || abandoned_error_; };
        if (timeout_ns == UINT64_MAX) {
            completion_.wait(lock, ready);
            return !abandoned_error_ && fence_retired_locked(fence);
        }
        completion_.wait_for(lock, std::chrono::nanoseconds(std::min<std::uint64_t>(timeout_ns, INT64_MAX)), ready);
        return !abandoned_error_ && fence_retired_locked(fence);
    }

    std::string fatal_error_detail() const override {
        const std::lock_guard lock(fence_mutex_);
        if (!abandoned_error_) return {};
        return "NVN replay abandoned submission " + std::to_string(abandoned_fence_) + ": " +
            abandoned_error_.detail;
    }

    std::optional<nvn::NativeProducerFence> snapshot_native_producer(std::uintptr_t queue) override {
        std::lock_guard lock(native_results_mutex_);
        const auto latest = native_latest_.find(queue);
        if (latest == native_latest_.end()) return nvn::NativeProducerFence{};
        const auto found = native_results_.find(latest->second);
        if (found == native_results_.end())
            return nvn::NativeProducerFence{latest->second, [] { return std::string{"Native producer result is not retained"}; }};
        auto result = found->second.result;
        return nvn::NativeProducerFence{latest->second, [result] { return result->check(); }};
    }

    std::function<std::string()> retain_native_producer_result(std::uint64_t fence) override {
        if (!fence) return [] { return std::string{}; };
        std::lock_guard lock(native_results_mutex_);
        const auto found = native_results_.find(fence);
        if (found == native_results_.end())
            return [] { return std::string{"Native producer result is not retained"}; };
        auto result = found->second.result;
        return [result] { return result->check(); };
    }

    ReplayError last_error() const override {
        std::scoped_lock lock(mutex_, fence_mutex_);
        return abandoned_error_ ? abandoned_error_ : execution_error_ ? execution_error_ :
            last_error_ ? last_error_ : encode_warning_;
    }

    ReplayStatistics statistics() const noexcept override {
        ReplayStatistics result;
        {
            std::scoped_lock lock(mutex_, fence_mutex_);
            result = published_statistics_;
            result.submit_cpu_ns = submit_cpu_ns_;
            result.encode_thread_cpu_ns = encode_thread_ ? encode_thread_->cpu_ns() : stopped_encode_cpu_ns_;
            result.gpu_ns = gpu_ns_;
            result.completion_cpu_ns = completion_cpu_ns_;
            result.counter_timestamp_fallbacks = counter_fallbacks_;
        }
        // Learned counters move on the worker and, at retirement, under
        // fence_mutex_. Copy them after releasing those locks: retirement
        // takes fence_mutex_ and then the learned mutex.
        {
            std::lock_guard lock(learned_.mutex);
            result.guest_write_logs = learned_.logs;
            result.guest_write_pools_learned = learned_.ranges.size();
            result.guest_write_misses = learned_.misses;
            result.guest_write_predicted_submissions = learned_.predicted_submissions;
        }
        return result;
    }

    std::vector<std::pair<std::string, std::uint64_t>> skip_reasons() const override {
        std::lock_guard lock(mutex_);
        return {published_skips_.begin(), published_skips_.end()};
    }

    // A zero-filled buffer the GPU only reads; created on first use.
    // The conditional-render kernel and its argument table, built on first
    // use; nil when they cannot be built (conditional rendering then reads
    // its counters on the CPU).
    id<MTLComputePipelineState> predicate_pipeline() {
        // Built once: at driver install on a background thread (prepare_internal_pipelines),
        // or here when the first conditional comes first; a caller meanwhile waits for it.
        std::call_once(predicate_once_, [&] {
            const auto begin = timeline_now_ns();
            NSError* error = nil;
            id<MTLLibrary> library = [device_ newLibraryWithSource:@(predicate_source) options:nil error:&error];
            id<MTLFunction> function = [library newFunctionWithName:@"armigo_nvn_render_predicate"];
            if (function) predicate_pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
            predicate_vertex_ = [library newFunctionWithName:@"armigo_nvn_render_predicate_vertex"];
            if (predicate_pipeline_) {
                auto* descriptor = [MTL4ArgumentTableDescriptor new];
                descriptor.maxBufferBindCount = 1;
                predicate_table_ = [device_ newArgumentTableWithDescriptor:descriptor error:&error];
            }
            if (!predicate_pipeline_ || !predicate_table_) {
                predicate_pipeline_ = nil;
                armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: the conditional-render kernel is unavailable (%s); conditional "
                             "rendering waits for the GPU instead\n",
                             error ? error.localizedDescription.UTF8String : "no detail");
            }
            // Armigo's own pipelines are not in the archive: their compiles are timed as kind 2.
            timeline_event("pso_compiled", 2, timeline_now_ns() - begin);
        });
        return predicate_pipeline_;
    }

    // Armigo's own pipelines that every title needs, built ahead of the first
    // frame on a background thread so none compiles in the middle of play.
    void prepare_internal_pipelines() {
        internal_preparer_ = std::thread([this] {
            pthread_setname_np("nvn-internal-pso");
            @autoreleasepool {
                (void)predicate_pipeline();
                prepare_clear_shaders(device_);
                // Ahead of the first frame: not first-use compiles.
                (void)take_clear_compiles();
            }
        });
    }
    id<MTL4ArgumentTable> predicate_table() const { return predicate_table_; }
    // A clear configuration the encode thread compiled at first use goes to
    // the pipeline record, so the next launch's warm-up builds it.
    void note_clear_pipeline(const ClearPipelineConfig& config) {
        if (recorder_ && recorder_->enabled()) recorder_->note(clear_row(config));
    }

    // The in-pass predicate: a vertex-only pipeline with rasterization
    // disabled, for one set of pass attachments. Each set compiles once, in
    // the background, from the function the internal preparer built; nil until
    // it is ready or when it cannot be built, and the caller then runs the
    // kernel, so no conditional waits for a compile.
    id<MTLRenderPipelineState> predicate_render_pipeline(const std::array<MTLPixelFormat, 8>& colors,
                                                         MTLPixelFormat depth, MTLPixelFormat stencil,
                                                         NSUInteger samples) {
        if (!predicate_pipeline() || !predicate_vertex_) return nil;
        PredicateDrawPipelines::Key key{};
        for (std::size_t index = 0; index != colors.size(); ++index) key[index] = colors[index];
        key[8] = depth;
        key[9] = stencil;
        key[10] = samples;
        auto pipelines = predicate_draw_pipelines_;
        {
            std::lock_guard lock(pipelines->mutex);
            const auto [found, added] = pipelines->states.try_emplace(key, nil);
            if (!added) return found->second;
        }
        auto* descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = predicate_vertex_;
        descriptor.rasterizationEnabled = NO;
        for (std::size_t index = 0; index != colors.size(); ++index)
            descriptor.colorAttachments[index].pixelFormat = colors[index];
        descriptor.depthAttachmentPixelFormat = depth;
        descriptor.stencilAttachmentPixelFormat = stencil;
        descriptor.rasterSampleCount = std::max<NSUInteger>(samples, 1);
        // The handler holds the table, not the backend, which may be gone by then.
        [device_ newRenderPipelineStateWithDescriptor:descriptor
                                    completionHandler:^(id<MTLRenderPipelineState> state, NSError* error) {
            std::lock_guard lock(pipelines->mutex);
            pipelines->states[key] = state;
            if (!state && !pipelines->reported) {
                pipelines->reported = true;
                armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: an in-pass conditional-render pipeline is unavailable (%s); "
                             "those conditionals use the compute kernel\n",
                             error ? error.localizedDescription.UTF8String : "no detail");
            }
        }];
        return nil;
    }

    // Worker-owned host copy resources. Native program reflection and the
    // command-stream texture cache do not participate in this path.
    id<MTLRenderPipelineState> texture_copy_pipeline(MTLPixelFormat format, std::string& detail) {
        if (!texture_copy_.library_attempted) {
            texture_copy_.library_attempted = true;
            static constexpr char source[] = R"msl(
#include <metal_stdlib>
using namespace metal;
struct CopyVertex { float4 position [[position]]; };
struct CopyParameters { float4 source_origin_scale; float4 destination_source_size; };
vertex CopyVertex armigo_copy_vertex(uint index [[vertex_id]]) {
    const float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(positions[index], 0, 1)};
}
fragment float4 armigo_copy_fragment(CopyVertex input [[stage_in]],
    texture2d<float> source [[texture(0)]], sampler filter [[sampler(0)]],
    constant CopyParameters& parameters [[buffer(0)]]) {
    const float2 texel = parameters.source_origin_scale.xy +
        (input.position.xy - parameters.destination_source_size.xy) * parameters.source_origin_scale.zw;
    return source.sample(filter, texel / parameters.destination_source_size.zw, level(0.0f));
}
)msl";
            auto* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_0;
            NSError* error = nil;
            texture_copy_.library = [device_ newLibraryWithSource:[NSString stringWithUTF8String:source]
                options:options error:&error];
            if (texture_copy_.library) {
                texture_copy_.vertex = [texture_copy_.library newFunctionWithName:@"armigo_copy_vertex"];
                texture_copy_.fragment = [texture_copy_.library newFunctionWithName:@"armigo_copy_fragment"];
            }
            if (!texture_copy_.vertex || !texture_copy_.fragment)
                texture_copy_.library_error = std::string("scaled texture copy shader compilation failed: ") +
                    (error ? error.localizedDescription.UTF8String : "missing entry point");
        }
        if (!texture_copy_.library_error.empty()) {
            detail = texture_copy_.library_error;
            return nil;
        }
        auto [entry, inserted] = texture_copy_.pipelines.try_emplace(format);
        if (inserted) {
            const auto begin = timeline_now_ns();
            auto* descriptor = [MTLRenderPipelineDescriptor new];
            descriptor.vertexFunction = texture_copy_.vertex;
            descriptor.fragmentFunction = texture_copy_.fragment;
            descriptor.colorAttachments[0].pixelFormat = format;
            descriptor.colorAttachments[0].blendingEnabled = NO;
            descriptor.rasterSampleCount = 1;
            NSError* error = nil;
            entry->second.state = [device_ newRenderPipelineStateWithDescriptor:descriptor error:&error];
            if (!entry->second.state)
                entry->second.error = std::string("scaled texture copy pipeline compilation failed: ") +
                    (error ? error.localizedDescription.UTF8String : "unknown error");
            timeline_event("pso_compiled", 2, timeline_now_ns() - begin);
        }
        detail = entry->second.error;
        return entry->second.state;
    }
    id<MTLSamplerState> texture_copy_sampler(bool linear) {
        auto& sampler = texture_copy_.samplers[linear ? 1 : 0];
        if (!sampler) {
            auto* descriptor = [MTLSamplerDescriptor new];
            descriptor.minFilter = descriptor.magFilter = linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
            descriptor.sAddressMode = descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
            descriptor.supportArgumentBuffers = YES;
            sampler = [device_ newSamplerStateWithDescriptor:descriptor];
        }
        return sampler;
    }
    id<MTL4ArgumentTable> texture_copy_arguments() {
        if (!texture_copy_.arguments) {
            auto* descriptor = [MTL4ArgumentTableDescriptor new];
            descriptor.maxBufferBindCount = descriptor.maxTextureBindCount = descriptor.maxSamplerStateBindCount = 1;
            descriptor.initializeBindings = YES;
            texture_copy_.arguments = [device_ newArgumentTableWithDescriptor:descriptor error:nil];
        }
        return texture_copy_.arguments;
    }

    id<MTLBuffer> zero_buffer() {
        if (!zero_buffer_) {
            zero_buffer_ = [device_ newBufferWithLength:64 * 1024 options:MTLResourceStorageModeShared];
            ++statistics_.new_buffer_allocations;
            if (zero_buffer_) std::memset(zero_buffer_.contents, 0, zero_buffer_.length);
        }
        return zero_buffer_;
    }

    id<MTLSamplerState> empty_sampler() {
        if (!empty_sampler_) {
            auto* descriptor = [MTLSamplerDescriptor new];
            descriptor.supportArgumentBuffers = YES;
            empty_sampler_ = [device_ newSamplerStateWithDescriptor:descriptor];
        }
        return empty_sampler_;
    }

    static constexpr std::size_t timestamp_heap_capacity = 1024;
    // Timestamp heaps are one pool for every submission, at most 24, below
    // Metal 4's per-process limit of 32. A submission takes a heap for each
    // 1024 reports when it needs one and returns them when it retires: from
    // the pool, a new heap below the cap, or, at the cap, the heaps an earlier
    // submission returns. Only one submission needing more than all of them
    // fails, explicitly.
    static constexpr std::size_t timestamp_heap_limit = 24;

    id<MTL4CounterHeap> counter_heap(PendingSubmission& pending, std::size_t timestamp) {
        const auto page = timestamp / timestamp_heap_capacity;
        if (page >= timestamp_heap_limit) return nil;
        while (pending.timestamp_heaps.size() <= page) {
            id<MTL4CounterHeap> heap = nil;
            {
                std::unique_lock lock(fence_mutex_);
                // Earlier submissions retire without this worker; this one
                // holds fewer than all heaps, so a wait ends.
                completion_.wait(lock, [&] {
                    return !timestamp_pool_.empty() || timestamp_heaps_made_ < timestamp_heap_limit ||
                           pending.timestamp_heaps.size() >= timestamp_heap_limit;
                });
                if (!timestamp_pool_.empty()) {
                    heap = timestamp_pool_.back();
                    timestamp_pool_.pop_back();
                } else if (timestamp_heaps_made_ < timestamp_heap_limit) {
                    ++timestamp_heaps_made_;
                } else {
                    return nil;
                }
            }
            if (!heap) {
                auto* descriptor = [MTL4CounterHeapDescriptor new];
                descriptor.type = MTL4CounterHeapTypeTimestamp;
                descriptor.count = timestamp_heap_capacity;
                NSError* error = nil;
                heap = [device_ newCounterHeapWithDescriptor:descriptor error:&error];
                if (!heap) {
                    std::lock_guard lock(fence_mutex_);
                    --timestamp_heaps_made_;
                    return nil;
                }
            }
            pending.timestamp_heaps.push_back(heap);
        }
        return pending.timestamp_heaps[page];
    }

    // Worker-owned; readers use the published snapshot. The first skip for
    // each reason is reported, so dropped work is never silent.
    void note_skip(const std::string& reason, unsigned op, std::uint64_t fence,
                   const std::string& detail) {
        ++statistics_.skipped;
        const auto count = ++skips_[reason];
        if (count == 1)
            armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: skipped unsupported work: %s\n", reason.c_str());
        if (count <= skip_detail_limit())
            armigo::diag::opt_in( "Armigo NVN: skip %llu of \"%s\" at op %u frame=%llu fence=%llu: %s\n",
                         static_cast<unsigned long long>(count), reason.c_str(), op,
                         static_cast<unsigned long long>(pool_present_count()),
                         static_cast<unsigned long long>(fence), detail.c_str());
    }

    bool mesh_diagnostics_enabled() const { return mesh_diagnostics_enabled_; }
    void note_mesh_diagnostic(std::string key, const std::string& detail) {
        const std::lock_guard lock(mesh_diagnostics_mutex_);
        ++mesh_diagnostics_seen_;
        if (mesh_diagnostics_keys_.count(key)) { ++mesh_diagnostics_duplicates_; return; }
        if (mesh_diagnostics_keys_.size() == 16) {
            if (++mesh_diagnostics_truncated_ == 1)
                armigo::diag::opt_in("nvn_mesh_diag records_truncated=1 record_limit=16\n");
            return;
        }
        mesh_diagnostics_keys_.insert(std::move(key));
        armigo::diag::opt_in("nvn_mesh_diag %s\n", detail.c_str());
    }

    // ARMIGO_NVN_SKIP_DETAIL=N describes the first N skips of each reason.
    static unsigned skip_detail_limit() {
        static const unsigned limit = [] {
            const char* value = std::getenv("ARMIGO_NVN_SKIP_DETAIL");
            return value ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : 0u;
        }();
        return limit;
    }

    id<MTLDevice> device() const { return device_; }
    // A plain member so the unset path is one predictable branch. The config
    // is fixed when this backend is constructed.
    bool pass_dump_enabled() const { return pass_dump_enabled_; }
    bool pass_dump_wants(std::uint64_t frame) const {
        return pass_dump_enabled_ && pass_dump_ && pass_dump_->wants(frame);
    }
    PassDump& pass_dump() { return *pass_dump_; }
    PoolTableCache& pool_tables() { return pool_tables_; }
    std::uint64_t pool_present_count() const { return pool_tables_.present_count(); }
    void adopt_pool_stats() {
        const auto stats = pool_tables_.statistics();
        statistics_.pool_table_builds = stats.builds;
        statistics_.pool_table_patches = stats.patches;
        statistics_.pool_table_kept = stats.kept;
        statistics_.pool_ids_read = stats.ids_read;
        statistics_.pool_textures_made = stats.textures_made;
        statistics_.pool_textures_left = stats.textures_left;
        statistics_.pool_walk_us = stats.walk_us;
    }
    void set_pool_table_allocation_failure(bool fail) override { pool_tables_.set_allocation_failure(fail); }
    id<MTLBuffer> current_texture_pool_buffer() const override { return pool_tables_.current_texture_buffer(); }
    id<MTL4CommandQueue> queue() const { return queue_; }
    id<MTL4Compiler> compiler() const { return compiler_; }
    ReplayResources& resources() { return *resources_; }
    ReplayStatistics& mutable_statistics() { return statistics_; }
    void accumulate_samples(id<MTLBuffer> visibility, std::size_t begin,
                            std::size_t end) {
        std::lock_guard lock(fence_mutex_);
        const auto* values = static_cast<const std::uint64_t*>(visibility.contents);
        for (std::size_t index = begin; index != end; ++index)
            samples_counter_ += values[index];
    }

    UploadSlice allocate_upload(PendingSubmission& pending, std::size_t size) {
        constexpr std::size_t alignment = 256;
        if (!size || size > std::numeric_limits<std::size_t>::max() - alignment) return {};
        const auto reserved = (size + alignment - 1) & ~(alignment - 1);
        auto* arena = pending.upload_arena;
        if (!arena || reserved > arena->capacity - arena->used) {
            arena = nullptr;
            for (auto& candidate : upload_arenas_) {
                // An arena is free once the submission that used it last
                // retired, also while a held queue keeps the published fence
                // back.
                if (candidate->capacity >= reserved && fence_completed(candidate->fence)) {
                    arena = candidate.get();
                    arena->used = 0;
                    ++statistics_.upload_arena_reuses;
                    break;
                }
            }
            if (!arena) {
                auto created = std::make_unique<UploadArena>();
                created->capacity = std::max<std::size_t>(4 * 1024 * 1024, reserved);
                created->buffer = [device_ newBufferWithLength:created->capacity
                                                     options:MTLResourceStorageModeShared];
                if (!created->buffer) return {};
                created->cpu = static_cast<std::byte*>(created->buffer.contents);
                created->gpu = created->buffer.gpuAddress;
                arena = created.get();
                upload_arenas_.push_back(std::move(created));
                ++statistics_.upload_arena_allocations;
                ++statistics_.new_buffer_allocations;
            }
            // Even an intermediate synchronous segment keeps its slices until
            // the entire submission fence retires; later draws cannot overwrite them.
            arena->fence = pending.fence;
            pending.upload_arena = arena;
            make_resident(pending, arena->buffer);
        }
        const auto offset = arena->used;
        arena->used += reserved;
        statistics_.upload_bytes += size;
        return {arena->buffer, arena->cpu + offset, arena->gpu + offset, offset};
    }

    id<MTLDepthStencilState> depth_state(const DepthStateKey& key) const {
        const auto* found = depth_states_.find(key);
        return found ? found->state : nil;
    }
    void remember_depth_state(const DepthStateKey& key, id<MTLDepthStencilState> state) {
        depth_states_.insert(key, std::make_unique<NativeDepthState>(NativeDepthState{state}));
    }
    // A pass whose opening draw leaves depth and stencil unused opens without
    // that attachment (no load, no store) until a later draw of the same pass
    // kind proves it needs one; the kind then always attaches it.
    bool depth_needed(const OmittedDepthKey& key) const { return depth_needed_.count(key) != 0; }
    void note_depth_needed(const OmittedDepthKey& key) { depth_needed_.insert(key); }
    // Whether the latest pipeline() call started a compile.
    bool created_pipeline() const { return created_pipeline_; }
    bool has_pipeline(const DrawPipelineKey& key) { return pipelines_.find(key) != nullptr; }
    NativePipeline* mesh_pipeline(const MeshPipelineKey& key,
                                  std::shared_ptr<const platform::CompiledGraphicsStage> task,
                                  std::shared_ptr<const platform::CompiledGraphicsStage> mesh,
                                  std::shared_ptr<const platform::CompiledGraphicsStage> fragment,
                                  std::string& detail);
    std::unique_ptr<NativePipeline> build_mesh_pipeline(const MeshPipelineKey& key,
                                  std::shared_ptr<const platform::CompiledGraphicsStage> task,
                                  std::shared_ptr<const platform::CompiledGraphicsStage> mesh,
                                  std::shared_ptr<const platform::CompiledGraphicsStage> fragment,
                                  std::string& detail);
    NativePipeline* pipeline(const DrawPipelineKey& key,
                                           std::shared_ptr<const platform::CompiledGraphicsStage> vertex,
                                           std::shared_ptr<const platform::CompiledGraphicsStage> fragment,
                                           std::string& detail);
    std::optional<NativeComputePipeline> compute_pipeline(
        std::shared_ptr<const platform::CompiledGraphicsStage> stage,
        std::string& detail);
    // With open_form_only (an archive build's open-form row) it captures the
    // open form alone and compiles nothing else; the result is not a usable
    // pipeline.
    std::unique_ptr<NativePipeline> build_pipeline(
        const DrawPipelineKey& key,
        std::shared_ptr<const platform::CompiledGraphicsStage> vertex,
        std::shared_ptr<const platform::CompiledGraphicsStage> fragment,
        std::string& detail, bool open_form_only = false);
    bool capture_open_form(const DrawPipelineKey& key, MTL4RenderPipelineDescriptor* open, std::string& detail);
    std::optional<NativeComputePipeline> build_compute_pipeline(
        std::shared_ptr<const platform::CompiledGraphicsStage> stage,
        std::string& detail);

    bool admit_cut(std::uint64_t draws) const {
        return segment_policy_.admit(draws, outstanding_segments_.load(std::memory_order_relaxed));
    }

    // Opens the submission's next command buffer on its allocator.
    bool begin_commands(PendingSubmission& pending) {
        if (encode_thread_) {
            // The record keeps the submission alive until its command
            // buffer is committed or ended: a failed submission can retire
            // while the encode thread has yet to replay its last records.
            encode_thread_->put(EncodeOp::begin_commands, 0,
                                new std::shared_ptr<PendingSubmission>(pending.shared_from_this()));
            encode_thread_->publish();
        } else {
            pending.commands = [device_ newCommandBuffer];
            if (!pending.commands) return false;
            [pending.retained addObject:pending.commands];
            [pending.commands beginCommandBufferWithAllocator:pending.allocator];
            if (pending.count_residency) pending.command_residency_sets.clear();
            use_residency_set(pending, pending.commands, pending.shared_residency->set());
        }
        pending.commands_open = true;
        return true;
    }

    // Ends the open command buffer without committing it.
    void end_commands(PendingSubmission& pending) {
        if (encode_thread_) {
            encode_thread_->put(EncodeOp::end_commands, 0);
            encode_thread_->publish();
        } else {
            [pending.commands endCommandBuffer];
        }
        pending.commands_open = false;
    }

    void commit_segment(PendingSubmission& submission, SegmentKind kind) {
        auto pending = submission.shared_from_this();
        const auto begin = timeline_now_ns();
        // Sealing uploads records their positions before this consumer.
        UploadFence uploads;
        resources_->commit_uploads(uploads);
        // The segment's allocations join the shared set before its commit,
        // which commits the set.
        pending->shared_residency->add(pending->resident_adds, pending->sequence);
        pending->resident_adds.clear();
        pending->commands_open = false;
        pending->committed_effects = pending->effects.size();
        const auto segment = ++next_segment_;
        const auto ordinal = pending->segments_committed++;
        if (timeline_enabled()) {
            // What the segment holds and its preceding logical upload range.
            const auto& c = pending->counters;
            timeline_event("segment_draws", segment, c.draws - std::exchange(pending->segment_draws, c.draws));
            timeline_event("segment_passes", segment, c.passes - std::exchange(pending->segment_passes, c.passes));
            timeline_event("segment_dispatches", segment,
                           c.dispatches - std::exchange(pending->segment_dispatches, c.dispatches));
            timeline_event("segment_uploads", segment,
                           c.window_uploads - std::exchange(pending->segment_uploads, c.window_uploads));
            if (uploads.value > uploads.floor) timeline_event("segment_upload_range", segment, uploads.value);
        }
        {
            std::lock_guard lock(fence_mutex_);
            pending->segments.submit();
        }
        outstanding_segments_.fetch_add(1, std::memory_order_relaxed);
        const UploadFence range{uploads.value, uploads.floor};
        Metal4Completion completion = ^(id<MTL4CommitFeedback> feedback) {
            // A failed upload batch fails the segment that waited for it: its
            // draws sampled textures without their content.
            if (resources_->uploads_failed(range))
                report_encode_error(*pending, "texture upload failed");
            complete_segment(pending, segment, ordinal, kind, feedback);
        };
        const auto publication = submission_owner_->reserve();
        if (encode_thread_) {
            encode_thread_->put(EncodeOp::commit, static_cast<std::uint32_t>(kind),
                                CommitRecord{(__bridge_retained void*)completion, pending.get(), segment, publication});
            encode_thread_->publish();
        } else {
            commit_now(*pending, pending->commands, completion, segment, kind, publication);
        }
        pending->phases.commit += timeline_now_ns() - begin;
    }

    // Makes the segment's residency current, ends its command buffer and
    // commits it: on the worker, or on the encode thread.
    void commit_now(PendingSubmission& pending, id<MTL4CommandBuffer> commands,
                    Metal4Completion completion, std::uint64_t segment, SegmentKind kind,
                    std::uint64_t publication) {
        if (!commands) {
            submission_owner_->cancel_reserved(publication);
            abandon_segment(pending, "Metal replay segment has no command buffer");
            return;
        }
        pending.shared_residency->commit();
        if (pending.residency) [pending.residency commit];
        [commands endCommandBuffer];
        if (pending.count_residency) {
            const auto count = pending.command_residency_sets.size();
            if (residency_count_log_)
                armigo::diag::opt_in("nvn_residency_sets fence=%llu segment=%llu count=%zu\n",
                    static_cast<unsigned long long>(pending.fence), static_cast<unsigned long long>(segment), count);
            if (scheduling_.observe_residency_sets)
                scheduling_.observe_residency_sets(scheduling_.context, pending.fence, segment, count);
        }
        const auto committed = timeline_now_ns();
        if (!pending.first_commit_ns) {
            pending.first_commit_ns = committed;
            timeline_event_at(committed, "first_commit", pending.fence, segment);
        }
        if (kind == SegmentKind::final) {
            pending.final_commit_ns = committed;
            pending.commit_time = mach_continuous_time();
            timeline_event_at(committed, "commit", pending.fence, segment);
        }
        timeline_event_at(committed, "segment_owner", segment, pending.fence);
        timeline_event_at(committed, "segment_commit", segment, static_cast<unsigned>(kind));
        if (!submission_owner_->publish_reserved(publication, commands, completion))
            abandon_segment(pending, "Metal replay publication position is unavailable");
    }

    // A failure the encode thread meets ends the submission with an
    // execution error, like a failed GPU segment.
    void report_encode_error(PendingSubmission& pending, const std::string& detail) {
        std::lock_guard lock(fence_mutex_);
        if (!pending.execution_error)
            pending.execution_error = {ReplayErrorCode::execution, "Metal replay encoding failed: " + detail};
    }

    void report_encode_warning(PendingSubmission& pending, const std::string& detail) {
        std::lock_guard lock(fence_mutex_);
        if (!pending.encode_warning)
            pending.encode_warning = {ReplayErrorCode::unsupported, detail};
    }

    // Completes a segment that could not be committed.
    void abandon_segment(PendingSubmission& pending, const std::string& detail) {
        outstanding_segments_.fetch_sub(1, std::memory_order_relaxed);
        std::unique_lock lock(fence_mutex_);
        if (!pending.execution_error)
            pending.execution_error = {ReplayErrorCode::execution, detail};
        pending.segments.complete(true);
        finish_segments_locked(pending.shared_from_this());
        publish_retired(lock);
    }

    ReplayError wait_segments(PendingSubmission& pending) {
        std::unique_lock lock(fence_mutex_);
        completion_.wait(lock, [&] { return pending.segments.drained(); });
        return pending.execution_error;
    }

    void seal_submission(const std::shared_ptr<PendingSubmission>& pending) {
        std::unique_lock lock(fence_mutex_);
        pending->segments.seal();
        finish_segments_locked(pending);
        publish_retired(lock);
    }

    void apply_effects(PendingSubmission& pending) {
        std::lock_guard lock(fence_mutex_);
        apply_effects_locked(pending, false);
    }

    // Units of the timestamps a counter report is built from, fixed for the
    // device: the heap's tick frequency, whether sampleTimestamps returns
    // nanoseconds (current systems) or mach ticks, and the offset from that
    // clock to the steady clock, from which the guest clock's origin is
    // taken away when a report is written.
    struct CounterClock {
        std::uint64_t heap_frequency = 0;
        mach_timebase_info_data_t timebase{};
        bool samples_in_ns = false;
        std::int64_t steady_offset_ns = 0;
        std::uint64_t mach_to_ns(std::uint64_t ticks) const {
            return (ticks / timebase.denom) * timebase.numer +
                   (ticks % timebase.denom) * timebase.numer / timebase.denom;
        }
        std::uint64_t heap_to_ns(std::uint64_t ticks) const {
            if (!heap_frequency) return mach_to_ns(ticks);
            return (ticks / heap_frequency) * 1'000'000'000ull +
                   (ticks % heap_frequency) * 1'000'000'000ull / heap_frequency;
        }
    };
    const CounterClock& counter_clock() {
        if (!counter_clock_ready_) {
            mach_timebase_info(&counter_clock_.timebase);
            if (!counter_clock_.timebase.denom) counter_clock_.timebase = {1, 1};
            counter_clock_.heap_frequency = [device_ queryTimestampFrequency];
            const auto steady_ns = [] {
                return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
            };
            MTLTimestamp cpu = 0, gpu = 0;
            const auto steady_before = steady_ns();
            [device_ sampleTimestamps:&cpu gpuTimestamp:&gpu];
            const auto steady_after = steady_ns();
            // A nanosecond value is a multiple of the mach tick count on
            // systems whose ticks are longer than a nanosecond.
            counter_clock_.samples_in_ns = cpu / 2 > mach_absolute_time();
            const auto cpu_ns = static_cast<std::int64_t>(
                counter_clock_.samples_in_ns ? cpu : counter_clock_.mach_to_ns(cpu));
            // One offset for the device: both clocks advance together, so every
            // report keeps the GPU's own spacing.
            counter_clock_.steady_offset_ns = steady_before + (steady_after - steady_before) / 2 - cpu_ns;
            counter_clock_ready_ = true;
        }
        return counter_clock_;
    }
    // Guarded by fence_mutex_, like the effects that use them.
    CounterClock counter_clock_{};
    bool counter_clock_ready_ = false;
    std::uint64_t counter_fallbacks_ = 0;

    // ARMIGO_NVN_DISPATCH_PROFILE totals since the last report (fence_mutex_):
    // GPU nanoseconds and segments per stage, and of all segments.
    std::map<std::string, std::pair<double, std::uint64_t>> dispatch_profile_;
    double profile_total_ns_ = 0;
    std::uint64_t profile_submissions_ = 0;
    void note_dispatch_profile(const PendingSubmission& pending) {
        for (std::size_t ordinal = 0; ordinal != pending.segment_times.size(); ++ordinal) {
            const auto& [start, end] = pending.segment_times[ordinal];
            if (!(start > 0 && end >= start)) continue;
            const double ns = (end - start) * 1e9;
            profile_total_ns_ += ns;
            if (ordinal < pending.segment_labels.size() && !pending.segment_labels[ordinal].empty()) {
                auto& [total, count] = dispatch_profile_[pending.segment_labels[ordinal]];
                total += ns;
                ++count;
            }
        }
        if (++profile_submissions_ % 60) return;
        std::vector<std::pair<std::string, std::pair<double, std::uint64_t>>> rows(dispatch_profile_.begin(),
                                                                                  dispatch_profile_.end());
        std::ranges::sort(rows, [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        double compute = 0;
        for (const auto& row : rows) compute += row.second.first;
        armigo::diag::write(armigo::diag::Level::info,
                            "nvn_dispatch_profile submissions=%llu gpu_ms=%.1f compute_ms=%.1f stages=%zu\n",
                            static_cast<unsigned long long>(profile_submissions_), profile_total_ns_ / 1e6,
                            compute / 1e6, rows.size());
        for (std::size_t index = 0; index != std::min<std::size_t>(rows.size(), 16); ++index)
            armigo::diag::write(armigo::diag::Level::info,
                                "nvn_dispatch_profile stage=%s dispatches=%llu gpu_ms=%.2f avg_us=%.1f\n",
                                rows[index].first.c_str(), static_cast<unsigned long long>(rows[index].second.second),
                                rows[index].second.first / 1e6,
                                rows[index].second.first / 1e3 / double(rows[index].second.second));
        dispatch_profile_.clear();
        profile_total_ns_ = 0;
    }
    // The latest report under the segment timestamp policy (fence_mutex_).
    std::uint64_t last_report_ticks_ = 0;
    void apply_effects_locked(PendingSubmission& pending, bool publish) {
        const auto* visibility = static_cast<const std::uint64_t*>(pending.visibility.contents);
        const auto& clock = counter_clock();
        // Segment times overlap (a segment's start can precede the previous
        // one's end, and a top report takes an earlier time than a bottom one
        // before it): under that policy reports never go back in report order.
        const bool segment_policy = nvn::title_policy().timestamps == nvn::TitlePolicy::Timestamps::segment;
        // The guest clock's origin on the steady clock: GPU times run on the
        // timeline the guest counter reads (the steady clock itself without a
        // running process).
        const auto guest_origin = execution::guest_timeline_origin();
        const std::int64_t origin_ns = guest_origin
            ? std::chrono::duration_cast<std::chrono::nanoseconds>(guest_origin->time_since_epoch()).count()
            : 0;
        // Writes a report at host time `ns`, placed on the clock the driver's
        // current timestamp reads, in the guest's 614.4 MHz ticks.
        const auto write_report = [&](const CompletionEffect& effect, std::uint64_t ns, std::uint64_t counter) {
            const auto steady = static_cast<std::int64_t>(ns) + clock.steady_offset_ns - origin_ns;
            ns = steady > 0 ? static_cast<std::uint64_t>(steady) : 0;
            NVNcounterData value{};
            value.timestamp = (ns / 625) * 384 + (ns % 625) * 384 / 625;
            if (segment_policy) value.timestamp = last_report_ticks_ = std::max(last_report_ticks_, value.timestamp);
            value.counter = counter;
            std::memcpy(effect.destination_cpu, &value, sizeof value);
        };
        const auto apply_one = [&](const CompletionEffect& effect) {
            if (effect.kind == CompletionEffect::Kind::reset) {
                samples_counter_ = 0;
                return;
            }
            if (effect.kind == CompletionEffect::Kind::signal) {
                auto left = effect.value;
                if (effect.add) left = __atomic_add_fetch(effect.event, effect.value, __ATOMIC_SEQ_CST);
                else __atomic_store_n(effect.event, effect.value, __ATOMIC_SEQ_CST);
                const auto observe = scheduling_.observe_event_signal;
                if (observe) observe(scheduling_.context, effect.event, pending.sequence, false);
                // With the value this signal left, so an equal-mode wait sees
                // it even if another signal replaces it first.
                evaluate_holds(effect.event, left, false, pending.sequence);
                if (observe) observe(scheduling_.context, effect.event, pending.sequence, true);
                return;
            }
            for (auto index = effect.visibility_begin; index < effect.visibility_end; ++index)
                samples_counter_ += visibility[index];
            if (effect.kind != CompletionEffect::Kind::report) return;
            if (effect.segment_time) {
                // Host seconds of the segment's start or end, on the clock
                // sampleTimestamps' CPU values use; the completion time when
                // the segment carried no GPU times.
                const auto times = effect.segment < pending.segment_times.size()
                    ? pending.segment_times[effect.segment] : std::array<double, 2>{};
                const double seconds = times[effect.segment_start ? 0 : 1];
                std::uint64_t ns = 0;
                if (seconds > 0) {
                    ns = static_cast<std::uint64_t>(seconds * 1e9);
                } else {
                    MTLTimestamp cpu_now = 0, gpu_now = 0;
                    [device_ sampleTimestamps:&cpu_now gpuTimestamp:&gpu_now];
                    ns = clock.samples_in_ns ? cpu_now : clock.mach_to_ns(cpu_now);
                    if (!counter_fallbacks_)
                        armigo::diag::write(armigo::diag::Level::warning, "Armigo NVN: a timestamp report's segment "
                                            "has no GPU times; it carries the completion time\n");
                    ++counter_fallbacks_;
                }
                write_report(effect, ns, 0);
                return;
            }
            MTLTimestamp cpu_after = 0, gpu_after = 0;
            [device_ sampleTimestamps:&cpu_after gpuTimestamp:&gpu_after];
            NSData* resolved = [effect.timestamp resolveCounterRange:NSMakeRange(effect.timestamp_index, 1)];
            if (!resolved || resolved.length < sizeof(MTL4TimestampHeapEntry)) {
                execution_error_ = {ReplayErrorCode::execution, "GPU timestamp heap resolution failed"};
                return;
            }
            MTL4TimestampHeapEntry entry{};
            std::memcpy(&entry, resolved.bytes, sizeof entry);
            // The heap resolves at the device's timestamp frequency, while
            // sampleTimestamps returns nanoseconds on current systems and mach
            // ticks on older ones. Interpolate between the two sampled pairs in
            // nanoseconds so the guest sees the GPU's own spacing between
            // reports, then place the result on the clock the driver's current
            // timestamp reads through the device's fixed offset.
            const auto sample_ns = [&](std::uint64_t value) {
                return clock.samples_in_ns ? value : clock.mach_to_ns(value);
            };
            const auto cpu_before_ns = sample_ns(effect.cpu_before);
            const auto gpu_before_ns = sample_ns(effect.gpu_before);
            const auto cpu_after_ns = sample_ns(cpu_after);
            const auto gpu_after_ns = sample_ns(gpu_after);
            const auto entry_ns = clock.heap_to_ns(entry.timestamp);
            auto ns = cpu_after_ns;
            if (gpu_after_ns > gpu_before_ns && cpu_after_ns >= cpu_before_ns &&
                entry_ns >= gpu_before_ns && entry_ns <= gpu_after_ns) {
                ns = cpu_before_ns + static_cast<std::uint64_t>(
                    (static_cast<long double>(entry_ns - gpu_before_ns) *
                        (cpu_after_ns - cpu_before_ns)) / (gpu_after_ns - gpu_before_ns));
            } else {
                if (!counter_fallbacks_)
                    armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: a counter report's GPU time is outside the window "
                                         "sampled around its submission; it carries the completion time\n");
                ++counter_fallbacks_;
            }
            write_report(effect, ns, effect.samples ? samples_counter_ : 0);
        };
        // The exact policy applies effects in command order. Replay publishes
        // report bytes, then the completion watermark, and only then the
        // signals that can wake a CPU writer. A drain inside the encode
        // stamps the landed prefix with fence 0 instead of moving the
        // watermark, so a later writer in this submission stays pending.
        if (!pending.replay_conditionals) {
            for (const auto& effect : pending.effects) apply_one(effect);
        } else {
            for (const auto& effect : pending.effects)
                if (effect.kind != CompletionEffect::Kind::signal) apply_one(effect);
            if (publish) reports_.set_completed(pending.sequence);
            else reports_.commit(0);
            for (const auto& effect : pending.effects)
                if (effect.kind == CompletionEffect::Kind::signal) apply_one(effect);
        }
        pending.effects.clear();
    }

private:
    void finish_segments_locked(const std::shared_ptr<PendingSubmission>& pending) {
        if (!pending->segments.retire()) return;
        timeline_event("gpu_complete", pending->fence, pending->segments.submitted);
        if (pending->timed_segments == pending->segments.submitted && pending->gpu_first_ns)
            timeline_event_at(pending->gpu_last_ns, "gpu_end", pending->fence,
                              pending->gpu_last_ns - pending->gpu_first_ns);
        finished_[pending->fence] = true;
        retire_finished();
    }

    void complete_segment(const std::shared_ptr<PendingSubmission>& pending,
                          std::uint64_t segment, std::uint32_t ordinal, SegmentKind kind,
                          id<MTL4CommitFeedback> feedback) {
        const auto cpu_start = thread_cpu_ns();
        // Feedback uses host seconds. Anchor that clock to the timeline clock;
        // callback delay and queue latency are never counted as GPU intervals.
        mach_timebase_info_data_t timebase{};
        mach_timebase_info(&timebase);
        const double host_seconds = double(mach_absolute_time()) * timebase.numer /
                                    timebase.denom / 1e9;
        const auto now = timeline_now_ns();
        std::uint64_t gpu_begin = 0, gpu_end = 0;
        if (feedback.GPUStartTime > 0 && feedback.GPUEndTime >= feedback.GPUStartTime &&
            feedback.GPUEndTime <= host_seconds) {
            const auto age = (host_seconds - feedback.GPUStartTime) * 1e9;
            const auto span = (feedback.GPUEndTime - feedback.GPUStartTime) * 1e9;
            if (age < double(now)) {
                gpu_begin = now - static_cast<std::uint64_t>(age);
                gpu_end = gpu_begin + static_cast<std::uint64_t>(span);
                timeline_event_at(gpu_begin, "segment_gpu_begin", segment, 0);
                timeline_event_at(gpu_end, "segment_gpu_end", segment, 0);
            }
        }
        outstanding_segments_.fetch_sub(1, std::memory_order_relaxed);
        std::unique_lock lock(fence_mutex_);
        if (kind == SegmentKind::final && pending->commit_time) {
            const auto ticks = mach_continuous_time() - pending->commit_time;
            gpu_ns_ += (ticks / timebase.denom) * timebase.numer +
                (ticks % timebase.denom) * timebase.numer / timebase.denom;
        }
        if (feedback.GPUStartTime > 0 && feedback.GPUEndTime >= feedback.GPUStartTime) {
            if (pending->segment_times.size() <= ordinal) pending->segment_times.resize(ordinal + 1);
            pending->segment_times[ordinal] = {feedback.GPUStartTime, feedback.GPUEndTime};
        }
        if (gpu_begin) {
            pending->gpu_interval_ns += gpu_end - gpu_begin;
            pending->gpu_first_ns = pending->gpu_first_ns
                ? std::min(pending->gpu_first_ns, gpu_begin) : gpu_begin;
            pending->gpu_last_ns = std::max(pending->gpu_last_ns, gpu_end);
            ++pending->timed_segments;
        }
        if (feedback.error && !pending->execution_error) {
            pending->execution_error = {ReplayErrorCode::execution,
                std::string("Metal replay segment failed: ") +
                    feedback.error.localizedDescription.UTF8String};
        }
        pending->segments.complete(feedback.error != nil);
        finish_segments_locked(pending);
        completion_cpu_ns_ += thread_cpu_ns() - cpu_start;
        publish_retired(lock);
    }

    id<MTLDevice> device_ = nil;
    bool pass_dump_enabled_ = false;
    std::unique_ptr<PassDump> pass_dump_;
    PoolTableCache pool_tables_;
    id<MTL4CommandQueue> queue_ = nil;
    std::shared_ptr<Metal4SubmissionOwner> submission_owner_;
    id<MTL4Compiler> compiler_ = nil;
    std::shared_ptr<ReplayResources> resources_;
    nvn::ConditionalReports reports_;
    const ReplaySchedulingPolicy scheduling_;
    const bool residency_count_log_ = [] {
        const char* value = std::getenv("ARMIGO_NVN_RESIDENCY_COUNT");
        return value && std::string_view(value) == "1";
    }();
    ReplayResources::LookupCounts reported_lookups_{};
    // Neither mutex is held while encoding or waiting for a GPU segment.
    mutable std::mutex mutex_;
    mutable std::mutex fence_mutex_;
    mutable std::atomic<bool> gpu_frame_work_enabled_{false};
    mutable GpuFrameWork gpu_frame_work_;
    std::condition_variable completion_;
    std::vector<RetiredLog> retired_log_;
    ReplayState state_{};
    ReplayError last_error_{};
    ReplayStatistics statistics_{}, published_statistics_{};
    // Guest-memory store prediction. Retirement holds fence_mutex_ and then
    // this mutex; statistics() copies it only after releasing fence_mutex_.
    // ARMIGO_NVN_PREDICT_GUEST_WRITES=0 keeps every storing stage a write
    // anywhere. Any other value, including unset, predicts from learned pools.
    mutable LearnedGuestWrites learned_{};
    bool predict_guest_writes_ = true;
    ReplayError encoder_error_{}, execution_error_{};
    // First abandoned submission, published before its GPU prefix drains.
    ReplayError abandoned_error_{};
    std::uint64_t abandoned_fence_ = 0;
    // The latest retired submission's encode warning (fence_mutex_).
    ReplayError encode_warning_{};
    std::uint64_t submit_cpu_ns_ = 0, gpu_ns_ = 0, completion_cpu_ns_ = 0;
    std::unordered_map<std::string, std::uint64_t> published_skips_;
    std::unordered_map<std::string, std::uint64_t> skips_;
    // Configuration is immutable after construction; limiter state is guarded
    // even though ordinary replay diagnostics are worker-owned.
    bool mesh_diagnostics_enabled_ = false;
    std::mutex mesh_diagnostics_mutex_;
    std::set<std::string> mesh_diagnostics_keys_;
    std::uint64_t mesh_diagnostics_seen_ = 0, mesh_diagnostics_duplicates_ = 0,
                  mesh_diagnostics_truncated_ = 0;
    struct TextureCopyResources {
        struct Pipeline { id<MTLRenderPipelineState> state = nil; std::string error; };
        bool library_attempted = false;
        std::string library_error;
        id<MTLLibrary> library = nil;
        id<MTLFunction> vertex = nil, fragment = nil;
        std::map<MTLPixelFormat, Pipeline> pipelines;
        std::array<id<MTLSamplerState>, 2> samplers{};
        id<MTL4ArgumentTable> arguments = nil;
    } texture_copy_;
    id<MTLBuffer> zero_buffer_ = nil;
    id<MTLComputePipelineState> predicate_pipeline_ = nil;
    id<MTL4ArgumentTable> predicate_table_ = nil;
    std::once_flag predicate_once_;
    std::thread internal_preparer_;
    id<MTLFunction> predicate_vertex_ = nil;
    struct PredicateDrawPipelines {
        // Color formats, depth and stencil formats, sample count.
        using Key = std::array<std::uint64_t, 11>;
        std::mutex mutex;
        // Nil while compiling or when the compile failed.
        std::map<Key, id<MTLRenderPipelineState>> states;
        bool reported = false;
    };
    std::shared_ptr<PredicateDrawPipelines> predicate_draw_pipelines_ = std::make_shared<PredicateDrawPipelines>();
    struct SubmissionSlot {
        id<MTL4CommandAllocator> allocator = nil;
        id<MTLBuffer> visibility = nil;
        // The guest-memory pool table and the pool-set generation it holds.
        id<MTLBuffer> guest_table = nil;
        std::uint64_t guest_table_generation = 0;
    };
    std::array<SubmissionSlot, max_submission_slots> submission_storage_;
    // The slots no encoded, unretired submission holds (under fence_mutex_).
    std::uint32_t free_slots_ = 0;
    bool slots_sized_ = false;
    // The timestamp heap pool (under fence_mutex_) and the heaps made so far.
    std::vector<id<MTL4CounterHeap>> timestamp_pool_;
    std::size_t timestamp_heaps_made_ = 0;
    // Allocations the last submission made resident: the next one reserves its
    // resident table that large.
    std::size_t resident_hint_ = 0;
    id<MTLSamplerState> empty_sampler_ = nil;
    std::vector<std::unique_ptr<UploadArena>> upload_arenas_;
    std::deque<std::shared_ptr<PendingSubmission>> previous_submissions_;
    std::atomic<std::uint64_t> next_fence_{1};
    std::uint64_t completed_fence_ = 0;
    // completed_fence_ for lock-free readers (completed_fence()).
    std::atomic<std::uint64_t> published_completed_fence_{0};
    std::uint64_t samples_counter_ = 0;
    std::map<std::uint64_t, std::shared_ptr<PendingSubmission>> inflight_;
    std::uint64_t encode_sequence_ = 0;
    // ARMIGO_NVN_GPUPROF, worker-owned; null when off.
    std::unique_ptr<GpuProfile> gpu_profile_;
    std::uint64_t last_encoded_fence_ = 0;
    // The title's replay: core replaces the queue model and submission core
    // in stages; a stage not yet landed runs as standard.
    static bool core_replay() {
        return nvn::title_policy().replay == nvn::TitlePolicy::Replay::core;
    }
    std::atomic<bool> replay_reported_{false};
    SharedResidency shared_residency_;
    struct FeedbackSnapshot {
        std::uint64_t source = 0, submission = 0, epoch = 0, use = 0;
        NSUInteger level = 0, slice = 0;
        id<MTLTexture> texture = nil;
    };
    struct BorderCoverageImage { id<MTLTexture> texture = nil; std::uint64_t use = 0; };
    struct BorderIdentityView { id<MTLTexture> source = nil, texture = nil; std::uint64_t use = 0; };
    std::array<BorderCoverageImage, 8> border_coverage_;
    std::array<BorderIdentityView, 8> border_identity_;
    std::uint64_t border_use_ = 0;
    std::array<FeedbackSnapshot, 8> feedback_snapshots_;
    std::uint64_t feedback_use_ = 0;
    std::unordered_map<const CommandSubmission*, std::weak_ptr<ResidencyCarry>> carries_;
    std::size_t residency_sweep_retirements_ = 0;
    std::map<std::uint64_t, bool> finished_;
    struct QueuedSubmission {
        std::uint64_t fence = 0;
        // The NVN queue that submitted it (0 when the caller names none).
        std::uintptr_t queue = 0;
        std::shared_ptr<const CommandSubmission> captured;
        std::uint64_t scheduling_period_ns = 0;
        bool preserve_encode_spin = false;
        bool frame_boundary = false;
        std::uint64_t frame_end_fence = 0;
        // The records it runs, by walk index: a submission split at its
        // unmet event waits is queued as parts that share its commands.
        std::size_t first_record = 0, end_record = SIZE_MAX;
        // Not a submission: `queue` was finalized, and its replay state goes
        // once the work queued before this is encoded.
        bool forget_queue = false;
        // Not a submission: `queue` waits for this fence to complete
        // (nvnQueueWaitSync); its later work stays behind it, and `fence`
        // completes once it passed.
        std::uint64_t wait_on = 0;
        // Leading waits already seen met when a part was held from submission.
        std::size_t passed_waits = 0;
        std::shared_ptr<std::atomic<bool>> cpu_ready = std::make_shared<std::atomic<bool>>(false);
        std::shared_ptr<std::atomic<bool>> capture_after;
        std::function<void(bool)> capture;
    };
    std::uint64_t active_cpu_fence_ = 0;
    std::shared_ptr<std::atomic<bool>> active_cpu_ready_;
    struct NativeResultEntry {
        std::uintptr_t queue = 0;
        std::shared_ptr<NativeProducerResult> result;
        bool retired = false;
        std::string preflight_error;
    };
    std::mutex native_results_mutex_;
    std::unordered_map<std::uint64_t, NativeResultEntry> native_results_;
    std::unordered_map<std::uintptr_t, std::uint64_t> native_latest_;
    void collect_native_results_locked() {
        std::erase_if(native_results_, [&](const auto& item) {
            const auto latest = native_latest_.find(item.second.queue);
            const bool current = latest != native_latest_.end() && latest->second == item.first;
            return item.second.retired && !current && item.second.result.use_count() == 1;
        });
    }
    void register_native_producer(std::uintptr_t queue, std::uint64_t fence) {
        std::lock_guard lock(native_results_mutex_);
        native_latest_[queue] = fence;
        native_results_[fence] = {queue, std::make_shared<NativeProducerResult>(), false, {}};
        collect_native_results_locked();
    }
    void finish_native_producer(std::uint64_t fence, std::string error) {
        std::lock_guard lock(native_results_mutex_);
        const auto found = native_results_.find(fence);
        if (found == native_results_.end()) return;
        if (!found->second.preflight_error.empty()) error = std::move(found->second.preflight_error);
        found->second.result->finish(std::move(error));
        found->second.retired = true;
        collect_native_results_locked();
    }
    std::mutex native_mutex_;
    std::vector<std::weak_ptr<Metal4NativeSubmission>> native_submissions_;
    // The queue mutex keeps both hint fields coherent when a submission captures them.
    std::uint64_t scheduling_period_ns_ = 0;
    bool preserve_encode_spin_ = false;
    std::deque<QueuedSubmission> queued_;
    std::condition_variable queued_ready_;
    // A queue held at its head submission's leading event waits (see
    // "Queues and event waits" in the project documentation). `passed` counts the
    // waits already seen met, in record order.
    struct LeadingWait {
        const volatile std::uint32_t* value = nullptr;
        std::int32_t mode = 0;
        std::uint32_t wait_value = 0;
        // Captured only when this wait is first reached, in encoding order.
        mutable std::uint64_t predecessor = 0;
        mutable bool reached = false;
    };
    struct QueueHold {
        std::uint64_t fence = 0;
        std::vector<LeadingWait> waits;
        std::size_t passed = 0;
    };
    std::mutex hold_mutex_;
    // The newest signal of each event word whose submission was encoded and
    // has not retired: the value a write leaves, or an add, whose result
    // depends on the writes before it landing. Taken after fence_mutex_ and
    // hold_mutex_.
    struct EncodedEventWrite {
        const volatile std::uint32_t* event = nullptr;
        std::uint64_t fence = 0;
        std::uint32_t value = 0;
        bool exact = false;
        // Guest fences can be held and encoded out of order; sequences cannot.
        std::uint64_t sequence = 0;
    };
    mutable std::mutex encoded_writes_mutex_;
    std::vector<EncodedEventWrite> encoded_writes_;
    // By the fence of the held submission or part.
    std::unordered_map<std::uint64_t, QueueHold> holds_;
    // holds_.size() and the submissions being split (their holds not yet
    // registered) for the signal paths' lock-free check.
    std::atomic<std::size_t> hold_count_{0}, splitting_{0};
    // Bumped when a submission is queued or a hold is released; the worker
    // waits for a change.
    std::atomic<std::uint64_t> schedule_generation_{0};
    // Fences retired past the published fence, as ranges first -> last, and
    // their frame work until they publish, kept only while frame work is
    // collected and for at most retired_work_limit fences (all under
    // fence_mutex_).
    std::map<std::uint64_t, std::uint64_t> retired_;
    static constexpr std::size_t retired_work_limit = 4096;
    std::map<std::uint64_t, std::optional<std::uint64_t>> applied_work_;
    bool hold_stall_reported_ = false;
    // The replay state of the queue encoded last (none after that queue is
    // finalized), every other queue's, and the state a queue starts from
    // (worker only); whether a split submission was reported (under mutex_).
    static constexpr std::uintptr_t no_state_queue = UINTPTR_MAX;
    ReplayState initial_state_{};
    std::uintptr_t state_queue_ = 0;
    std::unordered_map<std::uintptr_t, ReplayState> queue_states_;
    bool split_reported_ = false;
    // The guest-memory pool table of the current pool set (worker only) and
    // the program tags whose misses or task launches were reported (under
    // fence_mutex_).
    std::uint64_t guest_table_generation_ = 0;
    // How many rebuilt tables have reported the pools left out as never addressed.
    int guest_table_admission_logs_ = 0;
    std::vector<std::uint32_t> guest_table_words_;
    std::vector<id<MTLBuffer>> guest_table_buffers_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> guest_table_pools_;
    std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> guest_table_ranges_;
    std::set<std::uint32_t> reported_guest_misses_;
    std::set<std::uint32_t> reported_task_launches_;

public:
    // Gives the submission its guest-memory pool table (shader ABI
    // armigo-guest-memory version 2): the slot's buffer holds the table of
    // the current GPU-accessible pool set with its miss record cleared, every
    // listed pool stays resident for the submission, and the resources keep
    // those pools until it retires. Null on success, else why not.
    const char* prepare_guest_table(PendingSubmission& pending) {
        std::vector<Resources::GuestMemoryPool> pools;
        const auto generation = resources_->claim_guest_memory_pools(guest_table_generation_, pools);
        if (!generation) return "this resource provider lists no memory pools for guest memory";
        if (generation != guest_table_generation_) {
            std::vector<GuestMemoryWindow> windows;
            windows.reserve(pools.size());
            for (const auto& pool : pools) windows.push_back({pool.begin, pool.end, pool.gpu});
            auto built = build_guest_memory_table(windows);
            if (!built) return "more memory pools than the guest-memory table holds (8191)";
            guest_table_words_ = std::move(built->words);
            guest_table_buffers_.clear();
            guest_table_pools_.clear();
            for (const auto& pool : pools) {
                guest_table_buffers_.push_back(pool.buffer);
                guest_table_pools_.emplace_back(pool.begin, pool.end - pool.begin);
            }
            // The covered addresses, merged, for overlap queries.
            std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
            for (const auto& pool : pools) ranges.emplace_back(pool.begin, pool.end);
            std::ranges::sort(ranges);
            std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
            for (const auto& range : ranges)
                if (!merged.empty() && range.first <= merged.back().second)
                    merged.back().second = std::max(merged.back().second, range.second);
                else
                    merged.push_back(range);
            guest_table_ranges_ = std::make_shared<const std::vector<std::pair<std::uint64_t, std::uint64_t>>>(
                std::move(merged));
            guest_table_generation_ = generation;
            const auto left_out = resources_->guest_memory_admission();
            if (guest_table_admission_logs_++ < 8)
                armigo::diag::write(armigo::diag::Level::info,
                                    "Armigo NVN: guest-memory table of %zu pools (%zu dropped as contained, %zu trimmed, "
                                    "%zu hint collisions); %zu pools left out as never addressed (%llu MB)\n",
                                    built->entries, built->dropped, built->trimmed, built->hint_collisions,
                                    left_out.omitted_pools,
                                    static_cast<unsigned long long>(left_out.omitted_bytes >> 20));
            else
                armigo::diag::write(armigo::diag::Level::info,
                                    "Armigo NVN: guest-memory table of %zu pools (%zu dropped as contained, %zu trimmed, "
                                    "%zu hint collisions)\n", built->entries, built->dropped, built->trimmed,
                                    built->hint_collisions);
        }
        auto& slot = submission_storage_[pending.slot];
        if (!slot.guest_table) {
            slot.guest_table = [device_ newBufferWithLength:guest_memory_buffer_bytes
                                                    options:MTLResourceStorageModeShared];
            ++statistics_.new_buffer_allocations;
            if (!slot.guest_table) return "the guest-memory table buffer could not be allocated";
            slot.guest_table_generation = 0;
        }
        auto* words = static_cast<std::uint32_t*>(slot.guest_table.contents);
        if (slot.guest_table_generation != guest_table_generation_) {
            std::memcpy(words, guest_table_words_.data(), guest_memory_table_bytes);
            slot.guest_table_generation = guest_table_generation_;
        }
        if (guest_table_buffers_.empty()) return "no memory pool is GPU-accessible for guest memory";
        // The miss count, tag, first address and latch start clear, as does
        // the write log.
        std::memset(words + 1, 0, 5 * sizeof(std::uint32_t));
        std::memset(reinterpret_cast<std::byte*>(words) + guest_memory_table_bytes, 0, guest_memory_write_log_bytes);
        pending.guest_table = slot.guest_table;
        pending.guest_ranges = guest_table_ranges_;
        [pending.retained addObject:slot.guest_table];
        make_resident(pending, slot.guest_table);
        for (const auto& buffer : guest_table_buffers_)
            if (buffer) make_resident(pending, buffer);
        return nullptr;
    }

    // The pools of the latest guest-memory table, as (guest address, size).
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>* guest_table_pools() const {
        return &guest_table_pools_;
    }
    // The covered ranges of the latest guest-memory table: what a storing
    // stage may write when no prediction applies.
    std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> guest_table_ranges() const {
        return guest_table_ranges_;
    }

    // Hands the pools a submission's guest-memory stages wrote (its table's
    // write log) to the resources as completed writes, once the GPU has run
    // every stage that could set a bit: at retirement, or at a CPU wait inside
    // the submission, which clears the log so retirement names only later
    // writes. A storing submission also teaches the learned-pool set, including
    // a retirement whose log the wait already cleared.
    void publish_guest_writes(const PendingSubmission& pending, bool clear) {
        if (!pending.guest_table || pending.execution_error) return;
        auto* contents = static_cast<std::byte*>(pending.guest_table.contents);
        const std::span words(reinterpret_cast<const std::uint32_t*>(contents), guest_memory_table_bytes / 4);
        const std::span log(reinterpret_cast<std::uint32_t*>(contents + guest_memory_table_bytes),
                            guest_memory_write_log_bytes / 4);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> written;
        visit_guest_memory_writes(words, log, [&](std::uint64_t start, std::uint64_t end) {
            resources_->note_completed_gpu_write(start, end - start);
            written.emplace_back(start, end);
        });
        if (clear) std::ranges::fill(log, 0u);
        if (predict_guest_writes_ && pending.guest_writes) learn_guest_writes(pending, written);
    }

    // Replaces the submission's write-anywhere ranges with the learned pools
    // when that set has settled. False leaves the table ranges in place.
    bool apply_guest_write_prediction(PendingSubmission& pending) {
        if (!predict_guest_writes_) return false;
        std::size_t learned = 0;
        std::uint64_t logs = 0;
        bool announce = false;
        {
            std::lock_guard lock(learned_.mutex);
            if (learned_.over || learned_.stable < LearnedGuestWrites::stable_after) return false;
            pending.guest_writes_predicted = true;
            pending.guest_ranges = learned_.snapshot;
            ++learned_.predicted_submissions;
            if (!learned_.announced) {
                learned_.announced = true;
                announce = true;
                learned = learned_.ranges.size();
                logs = learned_.logs;
            }
        }
        if (announce)
            armigo::diag::write(armigo::diag::Level::info,
                                "Armigo NVN: guest-memory stores are predicted from %zu learned pools after %llu logs\n",
                                learned, static_cast<unsigned long long>(logs));
        return true;
    }

    // A read that did not wait, of an address the full table covers and the
    // learned set does not. `pending` is the submission whose prediction was
    // relied on (this one, or an earlier one a later read skipped), and the
    // ring records its encoding sequence.
    void note_predicted_guest_read(const PendingSubmission& pending, std::uint64_t address,
                                   std::uint64_t size) {
        if (!pending.guest_writes_predicted || !size || size > UINT64_MAX - address) return;
        if (guest_ranges_overlap(pending.guest_ranges.get(), address, size)) return;
        if (!guest_ranges_overlap(guest_table_ranges_.get(), address, size)) return;
        std::lock_guard lock(learned_.mutex);
        auto& read = learned_.ring[learned_.ring_n % LearnedGuestWrites::ring_size];
        read = {pending.sequence, address, address + size};
        ++learned_.ring_n;
    }

    void set_learned_guest_write_cap(std::size_t cap) override {
        std::lock_guard lock(learned_.mutex);
        learned_.cap = cap == 0 ? LearnedGuestWrites::default_cap : cap;
        if (learned_.ranges.size() > learned_.cap) learned_.over = true;
    }

private:
    // One publish of a storing submission. A pool no learned range contains is
    // added, unless the set is already at its cap, which ends prediction.
    // Adding a pool before the set has settled restarts the count; after that,
    // prediction continues and a read that relied on the old set is reported.
    void learn_guest_writes(const PendingSubmission& pending,
                            const std::vector<std::pair<std::uint64_t, std::uint64_t>>& written) {
        int added = 0;
        int shadowed = 0;
        bool unchecked = false;
        bool emit_miss = false;
        bool miss_warning = false;
        bool emit_over = false;
        std::size_t learned = 0;
        std::size_t over_cap = 0;
        std::uint64_t logs = 0;
        std::uint64_t misses = 0;
        {
            std::lock_guard lock(learned_.mutex);
            ++learned_.logs;
            bool became_over = false;
            for (const auto& pool : written) {
                if (learned_.over) break;
                if (pool.second <= pool.first) continue;
                bool known = false;
                for (const auto& range : learned_.ranges)
                    if (pool.first >= range.first && pool.second <= range.second) {
                        known = true;
                        break;
                    }
                if (known) continue;
                if (learned_.ranges.size() == learned_.cap) {
                    learned_.over = true;
                    became_over = true;
                    break;
                }
                if (pending.guest_writes_predicted) {
                    const auto count = learned_.ring_n;
                    const auto first = count > LearnedGuestWrites::ring_size
                        ? count - LearnedGuestWrites::ring_size : 0;
                    for (std::uint64_t index = first; index < count; ++index) {
                        const auto& read = learned_.ring[index % LearnedGuestWrites::ring_size];
                        if (read.sequence < pending.sequence) continue;
                        if (pool.first < read.end && read.begin < pool.second) {
                            ++shadowed;
                            break;
                        }
                    }
                }
                const auto at = std::lower_bound(learned_.ranges.begin(), learned_.ranges.end(), pool.first,
                    [](const auto& range, std::uint64_t start) { return range.first < start; });
                learned_.ranges.insert(at, pool);
                ++added;
            }
            if (added && learned_.stable < LearnedGuestWrites::stable_after) learned_.stable = 0;
            else if (learned_.stable < std::numeric_limits<int>::max()) ++learned_.stable;
            if (pending.guest_writes_predicted && added)
                learned_.misses += static_cast<std::uint64_t>(added);
            if (added)
                learned_.snapshot = std::make_shared<const std::vector<std::pair<std::uint64_t, std::uint64_t>>>(
                    learned_.ranges);
            if (pending.guest_writes_predicted && added) {
                const auto count = learned_.ring_n;
                const auto first = count > LearnedGuestWrites::ring_size
                    ? count - LearnedGuestWrites::ring_size : 0;
                if (first && learned_.ring[first % LearnedGuestWrites::ring_size].sequence >= pending.sequence)
                    unchecked = true;
                const bool bad = shadowed || unchecked;
                emit_miss = bad || learned_.info_lines++ < 8;
                miss_warning = bad;
                learned = learned_.ranges.size();
                logs = learned_.logs;
                misses = learned_.misses;
            }
            if (became_over && !learned_.over_logged) {
                learned_.over_logged = true;
                emit_over = true;
                over_cap = learned_.cap;
            }
        }
        if (emit_over)
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN: guest-memory stores wrote in more than %zu pools; they count as a write "
                                "anywhere from now on\n", over_cap);
        if (emit_miss) {
            const char* suffix = shadowed ? "; a CPU read of that submission relied on one"
                : unchecked ? "; the read record had wrapped, so not all could be checked" : "";
            armigo::diag::write(miss_warning ? armigo::diag::Level::warning : armigo::diag::Level::info,
                                "Armigo NVN: guest-memory stores wrote in %d pool(s) the host had not learned "
                                "(log %llu, %llu such pools so far, %zu learned)%s\n",
                                added, static_cast<unsigned long long>(logs),
                                static_cast<unsigned long long>(misses), learned, suffix);
        }
    }

private:

    // Reports a submission's guest-memory misses once per program tag. The
    // table records the first program that missed in the submission: a later
    // program that misses in it too is counted but not named.
    void report_guest_misses(const PendingSubmission& pending) {
        if (!pending.guest_table) return;
        const auto* words = static_cast<const std::uint32_t*>(pending.guest_table.contents);
        if (!words[5] || !reported_guest_misses_.insert(words[2]).second) return;
        const auto first = std::uint64_t(words[3]) | std::uint64_t(words[4]) << 32;
        if (resources_->address_in_unaddressed_pool(first))
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN: program %08x read or wrote guest memory outside every pool, first at "
                                "%#llx (%u misses in the submission, by any program); it fell in an unaddressed pool, "
                                "so those accesses read zero and drop stores\n",
                                words[2], static_cast<unsigned long long>(first), words[1]);
        else
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN: program %08x read or wrote guest memory outside every pool, first at "
                                "%#llx (%u misses in the submission, by any program); those accesses read zero and drop stores;"
                                "%s\n",
                                words[2], static_cast<unsigned long long>(first), words[1],
                                resources_->describe_guest_address(first).c_str());
    }

    // Reports a task workgroup that asked for more mesh workgroups than Metal
    // launches from one, once per program tag. The status words hold the
    // largest such request and the largest tag that made one.
    void report_task_launches(const PendingSubmission& pending) {
        if (!pending.task_status || !pending.task_status[0] ||
            !reported_task_launches_.insert(pending.task_status[1]).second) return;
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo NVN: task program %08x asked one workgroup for %u mesh workgroups, beyond "
                            "Metal's 1,024; that workgroup's mesh draws were refused\n",
                            pending.task_status[1], pending.task_status[0]);
    }
    bool hold_reported_ = false;

    // The event waits a submission begins with, before any other record; the
    // replay's own sampler stamps do not end them. A wait without an event or
    // with an unknown mode ends them too: the encoder refuses it explicitly.
    // A part of a split submission begins with the wait it was split at.
    static std::vector<LeadingWait> leading_waits(const CommandSubmission& submission, std::size_t first_record) {
        struct Scan {
            std::vector<LeadingWait> waits;
            std::size_t index = 0, first = 0;
        } scan{{}, 0, first_record};
        static_cast<void>(walk_commands(submission, [](const CommandRecordView& record, void* context) {
            auto& scan = *static_cast<Scan*>(context);
            if (scan.index++ < scan.first) return true;
            if (record.op == Op::sampler_registrations) return true;
            if (record.op != Op::wait_event || record.payload_size < sizeof(WaitEvent)) return false;
            WaitEvent command{};
            std::memcpy(&command, record.payload, sizeof command);
            if (!command.value || (command.mode != NVN_EVENT_WAIT_MODE_EQUAL &&
                                   command.mode != NVN_EVENT_WAIT_MODE_GEQUAL_WRAP)) return false;
            scan.waits.push_back({command.value, command.mode, command.wait_value});
            return true;
        }, &scan));
        return std::move(scan.waits);
    }

    // Where a submission is split: the walk index of each event wait after
    // other work whose event is not met at submission (a wait another queue
    // or the CPU signals later parks only this queue there; the waits that
    // directly follow belong to the same part), and each fence sync it
    // records with its walk index.
    struct SubmissionSplit {
        std::vector<std::size_t> starts;
        std::vector<std::pair<std::size_t, SyncState*>> syncs;
    };
    SubmissionSplit split_at_unmet_waits(const CommandSubmission& submission) const {
        struct Scan {
            const ReplayImpl* self = nullptr;
            SubmissionSplit split;
            std::size_t index = 0;
            bool work = false, in_split_waits = false;
        } scan;
        scan.self = this;
        static_cast<void>(walk_commands(submission, [](const CommandRecordView& record, void* context) {
            auto& scan = *static_cast<Scan*>(context);
            const auto at = scan.index++;
            if (record.op == Op::sampler_registrations) return true;
            if (record.op == Op::fence_sync && record.payload_size >= sizeof(FenceSync)) {
                FenceSync command{};
                std::memcpy(&command, record.payload, sizeof command);
                if (command.sync) scan.split.syncs.emplace_back(at, command.sync);
            }
            if (record.op != Op::wait_event || record.payload_size < sizeof(WaitEvent)) {
                scan.work = true;
                scan.in_split_waits = false;
                return true;
            }
            WaitEvent command{};
            std::memcpy(&command, record.payload, sizeof command);
            // Leading waits hold the queue already; a malformed wait is
            // refused by the encoder.
            if (!scan.work || scan.in_split_waits || !command.value ||
                (command.mode != NVN_EVENT_WAIT_MODE_EQUAL && command.mode != NVN_EVENT_WAIT_MODE_GEQUAL_WRAP))
                return true;
            if (!scan.self->wait_met({command.value, command.mode, command.wait_value})) {
                scan.split.starts.push_back(at);
                scan.in_split_waits = true;
            }
            return true;
        }, &scan));
        return std::move(scan.split);
    }

    static bool wait_met_by(const LeadingWait& wait, std::uint32_t value) {
        return wait.mode == NVN_EVENT_WAIT_MODE_EQUAL ? value == wait.wait_value
            : std::int32_t(value - wait.wait_value) >= 0;
    }

    // The newest encoded, unretired signal of `event`, if any: it runs
    // before work encoded later, so it decides what such work sees.
    std::optional<EncodedEventWrite> newest_encoded_write(const volatile std::uint32_t* event) const {
        std::lock_guard lock(encoded_writes_mutex_);
        const auto found = std::ranges::find(encoded_writes_, event, &EncodedEventWrite::event);
        if (found == encoded_writes_.end()) return std::nullopt;
        return *found;
    }

    // Whether a wait is met once a signal left `value` in its event. A newer
    // encoded writer of the event still runs before the waiting work, so it
    // decides; an add's result is known only when it lands. CPU memory is
    // not evidence that a pending writer has yet to execute: its deferred
    // completion effect may still be waiting for retirement.
    bool met_after(const LeadingWait& wait, std::uint32_t value,
                   std::uint64_t publishing_sequence = 0) const {
        // A GPU signal from the writer this wait reached, or one encoded
        // after it, latches the parked wait. A later writer cannot erase
        // that pulse merely by having been encoded in the meantime.
        if (publishing_sequence && publishing_sequence >= wait.predecessor)
            return wait_met_by(wait, value);
        // Encode-time release still uses the newest pending exact write.
        if (const auto write = newest_encoded_write(wait.value))
            return write->exact && wait_met_by(wait, write->value);
        return wait_met_by(wait, value);
    }

    // Met by the newest encoded write of the event, which the queue runs
    // before anything encoded after it, or without one by the value in
    // memory.
    bool wait_met(const LeadingWait& wait) const {
        // Read the pending writer and CPU word as one snapshot. Loading the
        // word before this lock could carry an old value past retirement.
        std::lock_guard lock(encoded_writes_mutex_);
        const auto write = std::ranges::find(encoded_writes_, wait.value, &EncodedEventWrite::event);
        const bool first = !wait.reached;
        if (first) {
            wait.predecessor = write == encoded_writes_.end() ? 0 : write->sequence;
            wait.reached = true;
        }
        const bool met = write != encoded_writes_.end()
            ? write->exact && wait_met_by(wait, write->value)
            : wait_met_by(wait, __atomic_load_n(wait.value, __ATOMIC_SEQ_CST));
        if (first && !met && scheduling_.observe_event_wait)
            scheduling_.observe_event_wait(scheduling_.context, wait.value, wait.wait_value, wait.predecessor);
        return met;
    }

    // Passes the waits that are met now, in record order (hold_mutex_ held).
    // Returns whether all are passed.
    bool advance_hold(QueueHold& hold) const {
        while (hold.passed < hold.waits.size() && wait_met(hold.waits[hold.passed])) ++hold.passed;
        return hold.passed == hold.waits.size();
    }

    // Re-evaluates every held queue right after a signal left `value` in
    // `event`. A wait on that event is judged by the value the signal left,
    // not by a later store's; the waits after it read their events now. A
    // CPU signal releases a held queue's wait on its own value even with an
    // encoded write of the event still pending: the queue is parked at the
    // wait and sees the value before that write lands.
    void evaluate_holds(const volatile std::uint32_t* event, std::uint32_t value, bool cpu = false,
                        std::uint64_t publishing_sequence = 0) {
        // A submission being split may be about to hold on this event: its
        // read of the event and this signal's store are ordered, so either
        // the read saw the store or this evaluation waits for its holds.
        if (!hold_count_.load(std::memory_order_seq_cst) && !splitting_.load(std::memory_order_seq_cst)) return;
        bool released = false;
        {
            std::lock_guard hold(hold_mutex_);
            for (auto& [fence, held] : holds_) {
                if (held.passed == held.waits.size()) continue;
                if (held.waits[held.passed].value == event &&
                    (cpu ? wait_met_by(held.waits[held.passed], value)
                         : met_after(held.waits[held.passed], value, publishing_sequence)))
                    ++held.passed;
                if (advance_hold(held)) released = true;
            }
        }
        if (released) {
            schedule_generation_.fetch_add(1, std::memory_order_release);
            queued_ready_.notify_all();
        }
    }

    static void cpu_event_signaled(void* context, const volatile std::uint32_t* event, std::uint32_t value) noexcept {
        auto& self = *static_cast<ReplayImpl*>(context);
        self.evaluate_holds(event, value, true);
    }

    // Picks the first queued submission that may be encoded now (mutex_
    // held). Without unmet leading waits that is the front of the queue. A
    // submission whose leading event waits are unmet holds its queue, and
    // the submissions of other queues pass it; a held submission takes no
    // submission slot, so any number of them can be passed.
    bool pick_locked(QueuedSubmission& item, std::size_t& passed_waits) {
        std::vector<std::uintptr_t> blocked;
        for (auto it = queued_.begin(); it != queued_.end(); ++it) {
            if (it->capture) {
                if (it->capture_after && !it->capture_after->load(std::memory_order_acquire)) continue;
                item = std::move(*it);
                queued_.erase(it);
                return true;
            }
            if (!it->frame_boundary) {
                const auto skip = [&] { blocked.push_back(it->queue); };
                if (std::ranges::find(blocked, it->queue) != blocked.end()) {
                    skip();
                    continue;
                }
                if (it->forget_queue) {
                    item = std::move(*it);
                    queued_.erase(it);
                    return true;
                }
                // nvnQueueWaitSync: the queue goes on once the fence completed.
                if (it->wait_on) {
                    if (!fence_completed(it->wait_on)) {
                        skip();
                        continue;
                    }
                    item = std::move(*it);
                    queued_.erase(it);
                    return true;
                }
                bool held = false;
                bool known = false;
                std::size_t passed = 0;
                {
                    std::lock_guard hold(hold_mutex_);
                    if (const auto found = holds_.find(it->fence); found != holds_.end()) {
                        known = true;
                        if (advance_hold(found->second)) {
                            passed = found->second.waits.size();
                            holds_.erase(found);
                            hold_count_.store(holds_.size(), std::memory_order_release);
                        } else {
                            held = true;
                        }
                    }
                }
                if (!known && it->passed_waits) {
                    passed = it->passed_waits;
                } else if (!known) {
                    auto waits = leading_waits(*it->captured, it->first_record);
                    if (!waits.empty()) {
                        QueueHold candidate{it->fence, std::move(waits), 0};
                        // Evaluated and registered under one lock, so a signal
                        // landing meanwhile either is seen here or evaluates
                        // the new hold.
                        std::lock_guard hold(hold_mutex_);
                        if (advance_hold(candidate)) {
                            passed = candidate.waits.size();
                        } else {
                            held = true;
                            holds_.emplace(it->fence, std::move(candidate));
                            hold_count_.store(holds_.size(), std::memory_order_release);
                            if (!hold_reported_) {
                                hold_reported_ = true;
                                armigo::diag::write(armigo::diag::Level::info,
                                    "Armigo NVN: a queue is held at an unmet event wait; later queues' "
                                    "submissions pass it\n");
                            }
                        }
                    }
                }
                if (held) {
                    skip();
                    continue;
                }
                passed_waits = passed;
            }
            item = std::move(*it);
            queued_.erase(it);
            return true;
        }
        return false;
    }

    // Nothing can be encoded while a queue is held and no GPU work is left
    // to signal it: only a CPU signal can release it now.
    void report_hold_stall() {
        if (hold_stall_reported_) return;
        {
            std::lock_guard fences(fence_mutex_);
            if (!inflight_.empty()) return;
        }
        hold_stall_reported_ = true;
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo NVN: a queue is held at an event wait that no submitted GPU work can "
                            "signal; it waits for a CPU signal\n");
    }

    bool stopping_ = false;
    bool worker_stopped_ = false;
    std::condition_variable worker_exited_;
    std::timed_mutex shutdown_mutex_;
    std::shared_ptr<RetirementTails> retirement_tails_ = std::make_shared<RetirementTails>();
    std::atomic<bool> accepting_submissions_{true};
    std::thread worker_;
    SegmentPolicy segment_policy_;
    std::atomic<std::uint64_t> outstanding_segments_{0};
    std::uint64_t next_segment_ = 0;
    std::unique_ptr<EncodeThread> encode_thread_;
    std::uint64_t stopped_encode_cpu_ns_ = 0;
    std::shared_ptr<PendingSubmission> encoding_submission_;
    DrawTable<DrawPipelineKey, NativePipeline> pipelines_;
    // Mesh draw pipelines by the bytes of their MeshPipelineKey (worker only).
    std::map<std::string, std::unique_ptr<NativePipeline>> mesh_pipelines_;
    bool created_pipeline_ = false;
    // One Metal library per shader stage, shared by every pipeline that uses
    // it. The entry owns its stage, so a freed stage's address cannot reach
    // another stage's library.
    struct StageLibrary {
        std::shared_ptr<const platform::CompiledGraphicsStage> stage;
        id<MTLLibrary> library = nil;
    };
    std::unordered_map<const platform::CompiledGraphicsStage*, StageLibrary> stage_libraries_;
    std::mutex libraries_mutex_;
    // Pipeline warm-up, recording and archive (configure_pipeline_warmup).
    std::unique_ptr<PsoRecorder> recorder_;
    std::unique_ptr<PipelineWarmup> warm_;
    MTL4CompilerTaskOptions* task_options_ = nil;
    id<MTL4PipelineDataSetSerializer> serializer_ = nil;
    std::filesystem::path archive_directory_;
    std::vector<std::filesystem::path> loaded_archives_;
    // The archives pipelines are taken from without compiling (lookup only),
    // opened by archive_loader_ (archives_pending_ is set before any lookup).
    NSArray* lookup_archives_ = nil;
    std::atomic<bool> archives_pending_{false};
    // Lookup is on (ARMIGO_PSO_ARCHIVE not off, not capturing): compiles are archive misses.
    std::atomic<bool> archive_lookup_{false};
    std::atomic<bool> archives_loaded_{false};
    std::mutex archive_load_mutex_;
    std::condition_variable archive_load_changed_;
    std::thread archive_loader_;
    std::atomic<std::uint64_t> archive_hits_{0}, archive_open_hits_{0}, archive_misses_{0},
        archive_warmup_misses_{0};
    std::string last_archive_;
    // Open (color-unspecialized) forms a capturing build has compiled.
    std::unordered_set<std::string> captured_open_;
    std::mutex archive_mutex_;
    // Guarded by archive_keys_mutex_: see PipelineWarmupReport.
    std::mutex archive_keys_mutex_;
    std::vector<std::string> unarchived_keys_, warmup_missed_keys_;
    // Capturing compiles hold it shared; a flush holds it alone.
    std::shared_mutex capture_mutex_;
    std::atomic<std::uint64_t> captures_since_flush_{0};
    // A capturing compile, made under the capture lock. The serializer's
    // working store fails once it holds too many pipelines (Metal then writes
    // no archive at all), so every archive_chunk_pipelines captures are
    // written out as one archive file and the next ones start a new file.
    template <typename Compile>
    auto capture(Compile&& compile) {
        auto result = [&] {
            const std::shared_lock lock(capture_mutex_);
            return compile();
        }();
        if (captures_since_flush_.fetch_add(1) + 1 == archive_chunk_pipelines) flush_archive(false);
        return result;
    }
    bool archive_consolidated_ = false, session_archives_ = false, archive_write_failed_ = false;
    id<MTLLibrary> stage_library(const std::shared_ptr<const platform::CompiledGraphicsStage>& stage,
                                 std::string& detail);
    DrawTable<DepthStateKey, NativeDepthState> depth_states_;
    std::set<OmittedDepthKey> depth_needed_;
    std::unordered_map<const platform::CompiledGraphicsStage*, NativeComputePipeline>
        compute_pipelines_;
};

template <class T> T record_value(const void* payload) {
    T value;
    std::memcpy(&value, payload, sizeof value);
    return value;
}

void EncodeThread::run() {
    pthread_setname_np("nvn-encode");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    cpu_mark_ = thread_cpu_ns();
    // Inside a submission the thread spins for the next records (spin_ns_).
    ring_.consume([this](const EncodeRing::Header& header, const void* payload) {
                      return exec(header, payload);
                  },
                  [this] { return idle_; }, spin_ns_);
    owned_.reset();
    if (pool_) objc_autoreleasePoolPop(pool_);
}

bool EncodeThread::exec(const EncodeRing::Header& header, const void* payload) {
    const auto argument = header.argument;
    // encoder_ owns the active encoder until end_encoder. These aliases are
    // used only during this record and never outlive or replace that owner.
    __unsafe_unretained id<MTL4RenderCommandEncoder> render = (id<MTL4RenderCommandEncoder>)encoder_;
    __unsafe_unretained id<MTL4ComputeCommandEncoder> compute = (id<MTL4ComputeCommandEncoder>)encoder_;
    const auto table = [&] { return (__bridge id<MTL4ArgumentTable>)record_value<TableRecord>(payload).table; };
    const auto report = [&](const char* detail) {
        if (owner_) backend_.report_encode_error(*owner_, detail);
    };
    switch (static_cast<EncodeOp>(header.op)) {
    case EncodeOp::stop:
        return false;
    case EncodeOp::begin_commands:
        // Objects Metal autoreleases while encoding a submission go with its
        // final commit.
        idle_ = false;
        if (!pool_) pool_ = objc_autoreleasePoolPush();
        {
            std::unique_ptr<std::shared_ptr<PendingSubmission>> owner(
                record_value<std::shared_ptr<PendingSubmission>*>(payload));
            owned_ = std::move(*owner);
            owner_ = owned_.get();
            const auto spin = !owner_->preserve_encode_spin &&
                owner_->scheduling_period_ns >= spin_paced_min_period_ns ? 0 : spin_unpaced_ns;
            if (!spin_reported_ || spin != spin_ns_) {
                timeline_event("encode_spin_budget", owner_->fence, spin);
                spin_reported_ = true;
            }
            spin_ns_ = spin;
            backend_.observe_encode_spin(*owner_, spin_ns_);
        }
        commands_ = [backend_.device() newCommandBuffer];
        if (!commands_) {
            report("command buffer allocation failed");
            break;
        }
        [owner_->encode_retained addObject:commands_];
        [commands_ beginCommandBufferWithAllocator:owner_->allocator];
        if (owner_->count_residency) owner_->command_residency_sets.clear();
        use_residency_set(*owner_, commands_, owner_->shared_residency->set());
        break;
    case EncodeOp::end_commands:
        [commands_ endCommandBuffer];
        commands_ = nil;
        owner_ = nullptr;
        owned_.reset();
        break;
    case EncodeOp::upload_commit: {
        std::unique_ptr<std::function<void()>> commit(record_value<std::function<void()>*>(payload));
        (*commit)();
        break;
    }
    case EncodeOp::commit: {
        const auto record = record_value<CommitRecord>(payload);
        const auto scheduling_period_ns = record.owner->scheduling_period_ns;
        const auto kind = static_cast<SegmentKind>(argument);
        Metal4Completion completion = (__bridge_transfer Metal4Completion)record.completion;
        backend_.commit_now(*record.owner, commands_, completion, record.segment, kind, record.publication);
        commands_ = nil;
        owner_ = nullptr;
        owned_.reset();
        if (kind == SegmentKind::final) {
            idle_ = true;
            completion = nil;
            if (pool_) objc_autoreleasePoolPop(pool_);
            pool_ = nullptr;
            const auto now = thread_cpu_ns();
            cpu_ns_.fetch_add(now - cpu_mark_, std::memory_order_relaxed);
            cpu_mark_ = now;
            backend_.update_encode_budget(scheduling_period_ns);
        }
        break;
    }
    case EncodeOp::residency_set: {
        const auto record = record_value<ResidentRecord>(payload);
        if (commands_)
            use_residency_set(*owner_, commands_, (__bridge id<MTLResidencySet>)record.allocation);
        break;
    }
    case EncodeOp::call: {
        EncodeCall call = (__bridge_transfer EncodeCall)record_value<void*>(payload);
        EncodeCallContext context{commands_, owner_ ? call_residency(*owner_, commands_) : nil,
                                  owner_ ? owner_->encode_retained : nil, {}};
        call(context);
        if (!context.error.empty()) report(context.error.c_str());
        break;
    }
    case EncodeOp::render_begin: {
        const auto record = record_value<RenderBeginRecord>(payload);
        MTL4RenderPassDescriptor* pass = (__bridge_transfer MTL4RenderPassDescriptor*)record.descriptor;
        id<MTL4RenderCommandEncoder> encoder = [commands_ renderCommandEncoderWithDescriptor:pass];
        encoder_ = encoder;
        if (!encoder) {
            report("render encoder creation failed");
            break;
        }
        [encoder barrierAfterQueueStages:record.relaxed ? MTLStageFragment : MTLStageAll
                            beforeStages:record.relaxed ? MTLStageFragment : MTLStageAll
                       visibilityOptions:MTL4VisibilityOptionDevice];
        if (record.vertex_table)
            [encoder setArgumentTable:(__bridge id<MTL4ArgumentTable>)record.vertex_table
                             atStages:MTLRenderStageVertex];
        if (record.fragment_table)
            [encoder setArgumentTable:(__bridge id<MTL4ArgumentTable>)record.fragment_table
                             atStages:MTLRenderStageFragment];
        break;
    }
    case EncodeOp::transfer_begin: {
        id<MTL4ComputeCommandEncoder> encoder = [commands_ computeCommandEncoder];
        encoder_ = encoder;
        if (!encoder) {
            report("transfer encoder creation failed");
            break;
        }
        [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit | MTLStageDispatch
                       visibilityOptions:MTL4VisibilityOptionDevice];
        break;
    }
    case EncodeOp::end_encoder:
        [(id<MTL4CommandEncoder>)encoder_ barrierAfterStages:record_value<std::uint64_t>(payload)
                                           beforeQueueStages:argument ? MTLStageFragment : MTLStageAll
                                           visibilityOptions:MTL4VisibilityOptionDevice];
        [(id<MTL4CommandEncoder>)encoder_ endEncoding];
        encoder_ = nil;
        break;
    case EncodeOp::encoder_barrier: {
        const auto record = record_value<BarrierRecord>(payload);
        [(id<MTL4CommandEncoder>)encoder_ barrierAfterEncoderStages:record.after
                                                beforeEncoderStages:record.before
                                                  visibilityOptions:MTL4VisibilityOptionDevice | MTL4VisibilityOptions(argument)];
        break;
    }
    case EncodeOp::render_pipeline: {
        auto* compile = static_cast<PipelineCompile*>(const_cast<void*>(record_value<const void*>(payload)));
        const auto wait_begin = timeline_now_ns();
        id<MTLRenderPipelineState> state = compile->wait();
        if (const auto waited = timeline_now_ns() - wait_begin; waited > 20'000)
            timeline_event("pso_wait", 0, waited);
        // A pipeline that failed to compile drops the draws that use it.
        skip_draws_ = !state;
        // The replay's pipeline table owns the state for its whole life.
        if (state) {
            [render setRenderPipelineState:state];
        } else {
            if (owner_) backend_.report_encode_warning(*owner_, compile->error);
        }
        break;
    }
    case EncodeOp::depth_stencil:
        [render setDepthStencilState:(__bridge id<MTLDepthStencilState>)record_value<void*>(payload)];
        break;
    case EncodeOp::stencil_reference: {
        const auto record = record_value<StencilRecord>(payload);
        [render setStencilFrontReferenceValue:record.front backReferenceValue:record.back];
        break;
    }
    case EncodeOp::cull: [render setCullMode:MTLCullMode(argument)]; break;
    case EncodeOp::winding: [render setFrontFacingWinding:MTLWinding(argument)]; break;
    case EncodeOp::fill: [render setTriangleFillMode:MTLTriangleFillMode(argument)]; break;
    case EncodeOp::depth_clip: [render setDepthClipMode:MTLDepthClipMode(argument)]; break;
    case EncodeOp::depth_bias: {
        const auto bias = record_value<std::array<float, 3>>(payload);
        [render setDepthBias:bias[0] slopeScale:bias[1] clamp:bias[2]];
        break;
    }
    case EncodeOp::blend_color: {
        const auto color = record_value<std::array<float, 4>>(payload);
        [render setBlendColorRed:color[0] green:color[1] blue:color[2] alpha:color[3]];
        break;
    }
    case EncodeOp::viewports: {
        std::array<MTLViewport, 16> viewports;
        std::memcpy(viewports.data(), payload, argument * sizeof(MTLViewport));
        [render setViewports:viewports.data() count:argument];
        break;
    }
    case EncodeOp::scissors: {
        std::array<MTLScissorRect, 16> scissors;
        std::memcpy(scissors.data(), payload, argument * sizeof(MTLScissorRect));
        [render setScissorRects:scissors.data() count:argument];
        break;
    }
    case EncodeOp::visibility:
        [render setVisibilityResultMode:MTLVisibilityResultMode(argument)
                                 offset:record_value<std::uint64_t>(payload)];
        break;
    case EncodeOp::table_address:
        [table() setAddress:record_value<TableRecord>(payload).value atIndex:argument];
        break;
    case EncodeOp::table_vertex: {
        const auto record = record_value<TableRecord>(payload);
        [table() setAddress:record.value attributeStride:record.stride atIndex:argument];
        break;
    }
    case EncodeOp::table_texture:
        [table() setTexture:MTLResourceID{record_value<TableRecord>(payload).value} atIndex:argument];
        break;
    case EncodeOp::table_sampler:
        [table() setSamplerState:MTLResourceID{record_value<TableRecord>(payload).value} atIndex:argument];
        break;
    case EncodeOp::mesh_tables: {
        const auto record = record_value<MeshTablesRecord>(payload);
        if (!render) break;
        [render setArgumentTable:(__bridge id<MTL4ArgumentTable>)record.object_table atStages:MTLRenderStageObject];
        [render setArgumentTable:(__bridge id<MTL4ArgumentTable>)record.mesh_table atStages:MTLRenderStageMesh];
        break;
    }
    case EncodeOp::draw_mesh: {
        const auto record = record_value<DrawMeshRecord>(payload);
        if (skip_draws_) break;
        const auto object = MTLSizeMake(record.object[0], record.object[1], record.object[2]);
        const auto mesh = MTLSizeMake(record.mesh[0], record.mesh[1], record.mesh[2]);
        if (record.indirect)
            [render drawMeshThreadgroupsWithIndirectBuffer:record.indirect threadsPerObjectThreadgroup:object
                                 threadsPerMeshThreadgroup:mesh];
        else
            [render drawMeshThreadgroups:MTLSizeMake(record.groups, 1, 1) threadsPerObjectThreadgroup:object
                 threadsPerMeshThreadgroup:mesh];
        break;
    }
    case EncodeOp::draw: {
        const auto record = record_value<DrawRecord>(payload);
        if (skip_draws_) break;
        [render drawPrimitives:MTLPrimitiveType(record.primitive) vertexStart:record.start
                   vertexCount:record.count instanceCount:record.instances
                  baseInstance:record.base_instance];
        break;
    }
    case EncodeOp::draw_indexed: {
        const auto record = record_value<DrawIndexedRecord>(payload);
        if (skip_draws_) break;
        [render drawIndexedPrimitives:MTLPrimitiveType(record.primitive) indexCount:record.count
                            indexType:MTLIndexType(record.type) indexBuffer:record.address
                    indexBufferLength:record.length instanceCount:record.instances
                           baseVertex:record.base_vertex baseInstance:record.base_instance];
        break;
    }
    case EncodeOp::compute_pipeline_compile: {
        auto* compile = static_cast<ComputeCompile*>(const_cast<void*>(record_value<const void*>(payload)));
        const auto wait_begin = timeline_now_ns();
        id<MTLComputePipelineState> state = compile->wait();
        if (const auto waited = timeline_now_ns() - wait_begin; waited > 20'000)
            timeline_event("pso_wait", 1, waited);
        // A pipeline that failed to compile drops the dispatches that use it.
        skip_dispatches_ = !state;
        if (state) {
            [compute setComputePipelineState:state];
        } else if (owner_) {
            backend_.report_encode_warning(*owner_, compile->error);
        }
        break;
    }
    case EncodeOp::compute_table:
        [compute setArgumentTable:(__bridge id<MTL4ArgumentTable>)record_value<void*>(payload)];
        break;
    case EncodeOp::dispatch: {
        const auto record = record_value<DispatchRecord>(payload);
        if (skip_dispatches_) break;
        [compute dispatchThreadgroups:record.grid threadsPerThreadgroup:record.threads];
        break;
    }
    case EncodeOp::dispatch_indirect: {
        const auto record = record_value<DispatchIndirectRecord>(payload);
        if (skip_dispatches_) break;
        [compute dispatchThreadgroupsWithIndirectBuffer:record.address threadsPerThreadgroup:record.threads];
        break;
    }
    case EncodeOp::copy_buffer: {
        const auto record = record_value<CopyRecord>(payload);
        id<MTLBuffer> source = (__bridge_transfer id<MTLBuffer>)record.source;
        id<MTLBuffer> destination = (__bridge_transfer id<MTLBuffer>)record.destination;
        [compute copyFromBuffer:source sourceOffset:record.source_offset toBuffer:destination
              destinationOffset:record.destination_offset size:record.size];
        break;
    }
    case EncodeOp::copy_buffer_texture: {
        const auto record = record_value<CopyRecord>(payload);
        id<MTLBuffer> source = (__bridge_transfer id<MTLBuffer>)record.source;
        id<MTLTexture> destination = (__bridge_transfer id<MTLTexture>)record.destination;
        [compute copyFromBuffer:source sourceOffset:record.source_offset sourceBytesPerRow:record.row
            sourceBytesPerImage:record.image sourceSize:record.extent toTexture:destination
               destinationSlice:record.destination_slice destinationLevel:record.destination_level
              destinationOrigin:record.destination_origin];
        break;
    }
    case EncodeOp::copy_texture_buffer: {
        const auto record = record_value<CopyRecord>(payload);
        id<MTLTexture> source = (__bridge_transfer id<MTLTexture>)record.source;
        id<MTLBuffer> destination = (__bridge_transfer id<MTLBuffer>)record.destination;
        [compute copyFromTexture:source sourceSlice:record.source_slice sourceLevel:record.source_level
                    sourceOrigin:record.source_origin sourceSize:record.extent toBuffer:destination
               destinationOffset:record.destination_offset destinationBytesPerRow:record.row
        destinationBytesPerImage:record.image];
        break;
    }
    case EncodeOp::copy_texture: {
        const auto record = record_value<CopyRecord>(payload);
        id<MTLTexture> source = (__bridge_transfer id<MTLTexture>)record.source;
        id<MTLTexture> destination = (__bridge_transfer id<MTLTexture>)record.destination;
        [compute copyFromTexture:source sourceSlice:record.source_slice sourceLevel:record.source_level
                    sourceOrigin:record.source_origin sourceSize:record.extent toTexture:destination
                destinationSlice:record.destination_slice destinationLevel:record.destination_level
               destinationOrigin:record.destination_origin];
        break;
    }
    case EncodeOp::timestamp: {
        const auto record = record_value<TimestampRecord>(payload);
        [commands_ writeTimestampIntoHeap:(__bridge id<MTL4CounterHeap>)record.heap atIndex:record.index];
        break;
    }
    case EncodeOp::profile_stamp: {
        // Bits 0/1 select render/end; higher bits carry the begin stage.
        const auto record = record_value<TimestampRecord>(payload);
        auto* heap = (__bridge id<MTL4CounterHeap>)record.heap;
        if (argument & 1u)
            GpuProfile::stamp_render((id<MTL4RenderCommandEncoder>)encoder_, heap, record.index, argument & 2u,
                                     static_cast<MTLRenderStages>(argument >> 2));
        else
            GpuProfile::stamp_compute((id<MTL4ComputeCommandEncoder>)encoder_, heap, record.index);
        break;
    }
    case EncodeOp::render_timestamp: {
        const auto record = record_value<TimestampRecord>(payload);
        [render writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageFragment
                                     intoHeap:(__bridge id<MTL4CounterHeap>)record.heap atIndex:record.index];
        break;
    }
    case EncodeOp::predicate: {
        // The predicate binds its own pipeline: a failed game compute
        // pipeline (skip_dispatches_) must not drop it, or every block would
        // render.
        const auto record = record_value<PredicateRecord>(payload);
        const auto predicate_table = (__bridge id<MTL4ArgumentTable>)record.table;
        [compute setComputePipelineState:(__bridge id<MTLComputePipelineState>)record.pipeline];
        [predicate_table setAddress:record.job atIndex:0];
        [compute setArgumentTable:predicate_table];
        [compute dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        break;
    }
    case EncodeOp::predicate_draw: {
        const auto record = record_value<PredicateDrawRecord>(payload);
        const auto predicate_table = (__bridge id<MTL4ArgumentTable>)record.table;
        [render setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)record.pipeline];
        [predicate_table setAddress:record.job atIndex:0];
        [render setArgumentTable:predicate_table atStages:MTLRenderStageVertex];
        [render drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1];
        [render barrierAfterEncoderStages:MTLStageVertex beforeEncoderStages:MTLStageVertex
                        visibilityOptions:MTL4VisibilityOptionDevice];
        if (record.vertex_table)
            [render setArgumentTable:(__bridge id<MTL4ArgumentTable>)record.vertex_table
                            atStages:MTLRenderStageVertex];
        break;
    }
    case EncodeOp::draw_indirect_args: {
        const auto record = record_value<DrawIndirectArgsRecord>(payload);
        if (skip_draws_) break;
        [render drawPrimitives:MTLPrimitiveType(record.primitive) indirectBuffer:record.arguments];
        break;
    }
    case EncodeOp::draw_indexed_indirect_args: {
        const auto record = record_value<DrawIndexedIndirectArgsRecord>(payload);
        if (skip_draws_) break;
        [render drawIndexedPrimitives:MTLPrimitiveType(record.primitive) indexType:MTLIndexType(record.type)
                          indexBuffer:record.address indexBufferLength:record.length
                       indirectBuffer:record.arguments];
        break;
    }
    case EncodeOp::copy_memory: {
        const auto record = record_value<CopyMemoryRecord>(payload);
        std::memcpy(record.destination, record.source, record.size);
        break;
    }
    default:
        armigo::diag::write(armigo::diag::Level::warning,  "nvn_encode_thread unknown record %u\n", unsigned(header.op));
        std::abort();
    }
    return true;
}

id<MTLLibrary> ReplayImpl::stage_library(
    const std::shared_ptr<const platform::CompiledGraphicsStage>& stage, std::string& detail) {
    // The warm-up threads load libraries too, so the compile runs outside the
    // lock: a replay-worker miss never waits behind a warm-up thread's load.
    {
        const std::lock_guard lock(libraries_mutex_);
        if (const auto found = stage_libraries_.find(stage.get()); found != stage_libraries_.end())
            return found->second.library;
    }
    id<MTLLibrary> library = load_stage_library(device_, *stage, detail);
    if (!library) return nil;
    const std::lock_guard lock(libraries_mutex_);
    // A concurrent load of the same stage may have inserted first; keep that one.
    return stage_libraries_.try_emplace(stage.get(), StageLibrary{stage, library}).first->second.library;
}

NativePipeline* ReplayImpl::pipeline(
    const DrawPipelineKey& key,
    std::shared_ptr<const platform::CompiledGraphicsStage> vertex_stage,
    std::shared_ptr<const platform::CompiledGraphicsStage> fragment_stage,
                                                   std::string& detail) {
    created_pipeline_ = false;
    if (auto* found = pipelines_.find(key)) {
        if (found->predicted && !found->used) {
            found->used = true;
            ++statistics_.predicted_used;
            timeline_event("pso_predict_used", 0, 0);
        }
        return found;
    }
    const auto create_begin = timeline_now_ns();
    // The row's text (about a kilobyte of hex) is built only when the warm-up
    // table or an enabled recorder uses it.
    const bool recording = recorder_ && recorder_->enabled();
    const auto row = warm_ || recording ? pipeline_row(key) : std::nullopt;
    std::unique_ptr<NativePipeline> built;
    if (warm_ && row) built = warm_->take_render(row->text());
    const bool warmed = built != nullptr;
    if (!built) built = build_pipeline(key, std::move(vertex_stage), std::move(fragment_stage), detail);
    if (!built) return nullptr;
    if (recording && row) {
        recorder_->note({'S', {key.vertex->identity}});
        if (key.fragment) recorder_->note({'S', {key.fragment->identity}});
        recorder_->note(*row);
    }
    ++statistics_.graphics_pipelines;
    // A pipeline the warm-up built started no compile for this draw.
    created_pipeline_ = !warmed;
    timeline_event(warmed ? "pso_take" : "pso_render", 0, timeline_now_ns() - create_begin);
    return pipelines_.insert(key, std::move(built));
}

// Captures a pipeline's open form into the archive being built, once per
// distinct open form (synchronously, as every capturing compile). False when
// Metal refuses it.
bool ReplayImpl::capture_open_form(const DrawPipelineKey& key, MTL4RenderPipelineDescriptor* open,
                                   std::string& detail) {
    {
        const std::lock_guard lock(archive_mutex_);
        if (!captured_open_.insert(open_key(key)).second) return true;
    }
    NSError* error = nil;
    if (capture([&] {
            return [compiler_ newRenderPipelineStateWithDescriptor:open compilerTaskOptions:task_options_
                                                             error:&error];
        }))
        return true;
    detail = std::string("open-form pipeline creation failed: ") +
        (error ? error.localizedDescription.UTF8String : "unknown Metal error");
    return false;
}

// Color attachment formats, write masks and blending of a pipeline key.
bool configure_color_attachments(MTL4RenderPipelineColorAttachmentDescriptorArray* attachments,
                                 const DrawPipelineKey& key, std::string& detail) {
    for (std::size_t index = 0; index != key.colors.size(); ++index) {
        auto* attachment = attachments[index];
        attachment.pixelFormat = static_cast<MTLPixelFormat>(key.colors[index]);
        if (key.colors[index] == MTLPixelFormatInvalid || !key.fragment) {
            attachment.writeMask = MTLColorWriteMaskNone;
            continue;
        }
        const auto mask = (key.channel_mask >> (index * 4)) & 0xf;
        attachment.writeMask = static_cast<MTLColorWriteMask>(
            ((mask & 1u) ? MTLColorWriteMaskRed : 0) |
            ((mask & 2u) ? MTLColorWriteMaskGreen : 0) |
            ((mask & 4u) ? MTLColorWriteMaskBlue : 0) |
            ((mask & 8u) ? MTLColorWriteMaskAlpha : 0));
        if (!(key.blend_enable & (1u << index))) continue;
        const auto& blend = key.blend[index];
        const auto src_color = blend_factor(blend.source_color);
        const auto dst_color = blend_factor(blend.destination_color);
        const auto src_alpha = blend_factor(blend.source_alpha);
        const auto dst_alpha = blend_factor(blend.destination_alpha);
        const auto color_op = blend_operation(blend.color_equation);
        const auto alpha_op = blend_operation(blend.alpha_equation);
        if (!src_color || !dst_color || !src_alpha || !dst_alpha || !color_op || !alpha_op) {
            detail = "draw uses an unsupported blend equation or factor";
            return false;
        }
        attachment.blendingState = MTL4BlendStateEnabled;
        attachment.sourceRGBBlendFactor = *src_color;
        attachment.destinationRGBBlendFactor = *dst_color;
        attachment.rgbBlendOperation = *color_op;
        attachment.sourceAlphaBlendFactor = *src_alpha;
        attachment.destinationAlphaBlendFactor = *dst_alpha;
        attachment.alphaBlendOperation = *alpha_op;
    }
    return true;
}

// The fragment function of a pipeline key, specialized with the function
// constants its stage declares.
MTL4SpecializedFunctionDescriptor* fragment_function(const DrawPipelineKey& key, id<MTLLibrary> library) {
    auto* fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = library;
    fragment.name = [NSString stringWithUTF8String:key.fragment->entry_point.c_str()];
    auto* constants = [MTLFunctionConstantValues new];
    for (std::size_t slot = 0; key.fragment->fragment_output_types_from_render_targets &&
                              slot != key.colors.size(); ++slot) {
        std::uint32_t scalar = fragment_output_scalar(key.colors[slot]);
        [constants setConstantValue:&scalar type:MTLDataTypeUInt atIndex:slot];
    }
    if (key.fragment->emulates_depth_bias) {
        bool enabled = false;
        [constants setConstantValue:&enabled type:MTLDataTypeBool atIndex:8];
    }
    if (key.fragment->consumes_y_direction) {
        // Native fragment coordinates use the upper-left Metal convention.
        // An omitted function constant does not provide a library default.
        const bool negate = false;
        [constants setConstantValue:&negate type:MTLDataTypeBool atIndex:9];
    }
    // Reflected coverage records are populated for each admitted sampling pair.
    if (key.fragment->custom_border) {
        bool records = true;
        [constants setConstantValue:&records type:MTLDataTypeBool atIndex:12];
    }
    auto* specialized = [MTL4SpecializedFunctionDescriptor new];
    specialized.functionDescriptor = fragment;
    specialized.constantValues = constants;
    return specialized;
}

NativePipeline* ReplayImpl::mesh_pipeline(const MeshPipelineKey& key,
                                          std::shared_ptr<const platform::CompiledGraphicsStage> task,
                                          std::shared_ptr<const platform::CompiledGraphicsStage> mesh,
                                          std::shared_ptr<const platform::CompiledGraphicsStage> fragment,
                                          std::string& detail) {
    created_pipeline_ = false;
    const std::string bytes(reinterpret_cast<const char*>(&key), sizeof key);
    if (const auto found = mesh_pipelines_.find(bytes); found != mesh_pipelines_.end()) {
        if (!found->second->compile->wait()) {
            detail = "mesh draw pipeline failed to compile";
            return nullptr;
        }
        return found->second.get();
    }
    const bool recording = recorder_ && recorder_->enabled();
    auto effective_mesh = mesh;
    if ((warm_ || recording) && fragment) {
#ifdef ARMIGO_HAS_NATIVE_SHADERS
        effective_mesh = nvn::mesh_stage_for(mesh, *fragment, detail);
        if (!effective_mesh) return nullptr;
#endif
    }
    auto recorded_key = key;
    recorded_key.draw.vertex = effective_mesh.get();
    const auto row = warm_ || recording ? mesh_pipeline_row(recorded_key) : std::nullopt;
    // Record before first use can wait on the compiler, so a slow first draw
    // still teaches the next launch which task/mesh pipeline to prepare.
    if (recording && row) {
        if (task) recorder_->note({'S', {row->fields[0]}});
        recorder_->note({'S', {row->fields[1]}});
        if (fragment) recorder_->note({'S', {row->fields[2]}});
        recorder_->note(*row);
    }
    std::unique_ptr<NativePipeline> built;
    if (warm_ && row) built = warm_->take_render(row->text());
    const bool warmed = built != nullptr;
    if (!built) built = build_mesh_pipeline(key, task, mesh, fragment, detail);
    if (!built) return nullptr;
    // The warmed pipeline may retain package-loaded stages with different
    // addresses. Keep the runtime stages named by the pointer cache key too.
    built->mesh_source_stage = std::move(mesh);
    built->task_stage = std::move(task);
    built->fragment_stage = std::move(fragment);
    ++statistics_.graphics_pipelines;
    created_pipeline_ = !warmed;
    auto* pointer = built.get();
    const bool ready = pointer->compile->wait() != nil;
    mesh_pipelines_.emplace(bytes, std::move(built));
    if (!ready) detail = "mesh draw pipeline failed to compile";
    return ready ? pointer : nullptr;
}

std::unique_ptr<NativePipeline> ReplayImpl::build_mesh_pipeline(const MeshPipelineKey& key,
                                          std::shared_ptr<const platform::CompiledGraphicsStage> task,
                                          std::shared_ptr<const platform::CompiledGraphicsStage> mesh,
                                          std::shared_ptr<const platform::CompiledGraphicsStage> fragment,
                                          std::string& detail) {
    if (!mesh || !mesh->mesh || mesh.get() != key.draw.vertex || task.get() != key.task ||
        fragment.get() != key.draw.fragment || (task && !task->task)) {
        detail = "mesh pipeline stages differ from its key or lack launch facts";
        return nullptr;
    }
    auto source_mesh = mesh;
    // The mesh stage as its fragment stage reads it: Metal links each
    // fragment input to a mesh output.
    if (fragment) {
#ifdef ARMIGO_HAS_NATIVE_SHADERS
        auto partnered = nvn::mesh_stage_for(mesh, *fragment, detail);
        if (!partnered) return nullptr;
        mesh = std::move(partnered);
#else
        // Mesh stages come only from the native front end, which this build lacks.
        detail = "mesh draws need native shader translation, which this build does not have";
        return nullptr;
#endif
    }
    auto result = std::make_unique<NativePipeline>();
    result->vertex_slots.fill(-1);
    result->mesh_source_stage = std::move(source_mesh);
    result->task_stage = task;
    result->vertex_stage = mesh;
    result->fragment_stage = fragment;
    if (task && !(result->task_library = stage_library(task, detail))) return nullptr;
    result->vertex_library = stage_library(mesh, detail);
    if (!result->vertex_library) return nullptr;
    if (fragment && !(result->fragment_library = stage_library(fragment, detail))) return nullptr;
    const auto function = [](id<MTLLibrary> library, const std::string& name) {
        auto* descriptor = [MTL4LibraryFunctionDescriptor new];
        descriptor.library = library;
        descriptor.name = [NSString stringWithUTF8String:name.c_str()];
        return descriptor;
    };
    auto* descriptor = [MTL4MeshRenderPipelineDescriptor new];
    if (task) {
        descriptor.objectFunctionDescriptor = function(result->task_library, task->entry_point);
        const auto& size = task->task->local_size;
        descriptor.maxTotalThreadsPerObjectThreadgroup = size[0] * size[1] * size[2];
        // Metal sizes the payload in 16-byte units; the task stage sets its
        // own mesh grid, up to Metal's 1024 threadgroups.
        descriptor.payloadMemoryLength = (task->task->payload_bytes + 15u) & ~15u;
        descriptor.maxTotalThreadgroupsPerMeshGrid = 1024;
    }
    descriptor.meshFunctionDescriptor = function(result->vertex_library, mesh->entry_point);
    const auto& mesh_size = mesh->mesh->local_size;
    descriptor.maxTotalThreadsPerMeshThreadgroup = mesh_size[0] * mesh_size[1] * mesh_size[2];
    if (fragment) descriptor.fragmentFunctionDescriptor = fragment_function(key.draw, result->fragment_library);
    descriptor.rasterizationEnabled = !key.draw.rasterizer_discard;
    descriptor.rasterSampleCount = key.draw.samples > 1 ? key.draw.samples : 1;
    descriptor.alphaToCoverageState = key.draw.alpha_to_coverage
        ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
    if (!configure_color_attachments(descriptor.colorAttachments, key.draw, detail)) return nullptr;
    result->compile = std::make_shared<PipelineCompile>();
    if (NSArray* archives = loaded_archives()) {
        const auto lookup_begin = timeline_now_ns();
        for (id<MTL4Archive> archive in archives) {
            if (id<MTLRenderPipelineState> state = [archive newRenderPipelineStateWithDescriptor:descriptor error:nil]) {
                ++archive_hits_;
                timeline_event("pso_archive_hit", 0, timeline_now_ns() - lookup_begin);
                result->compile->finish(state, {});
                return result;
            }
        }
    }
    const std::string stages = (task ? task->identity : "-") + " " + mesh->identity;
    if (archive_lookup_.load(std::memory_order_acquire))
        note_archive_miss(0, stages, fragment ? fragment->identity : "-");
    const auto compile_begin = timeline_now_ns();
    NSError* error = nil;
    const auto create = [&] {
        return [compiler_ newRenderPipelineStateWithDescriptor:descriptor
                                         compilerTaskOptions:task_options_ error:&error];
    };
    // Capture synchronously, as render/compute pipelines do: the serializer
    // cannot safely record asynchronous compiles. No shared replay cache is
    // modified here, so the warm-up may call this factory on its threads.
    id<MTLRenderPipelineState> state = serializer_ ? capture(create) : create();
    timeline_event("pso_compiled", 0, timeline_now_ns() - compile_begin);
    result->compile->finish(state, state ? std::string{}
        : std::string("mesh draw pipeline failed to compile: ") +
            (error ? error.localizedDescription.UTF8String : "unknown error"));
    if (!state) {
        detail = "mesh draw pipeline failed to compile";
        armigo::diag::write(armigo::diag::Level::warning, "Armigo NVN: %s: %s\n", detail.c_str(),
                            error ? error.localizedDescription.UTF8String : "unknown error");
    }
    return result;
}

// Builds the pipeline for a key and starts its compile. Safe off the replay
// worker (the warm-up calls it), so it touches nothing but the compiler and
// the stage library table.
std::unique_ptr<NativePipeline> ReplayImpl::build_pipeline(
    const DrawPipelineKey& key,
    std::shared_ptr<const platform::CompiledGraphicsStage> vertex_stage,
    std::shared_ptr<const platform::CompiledGraphicsStage> fragment_stage,
                                                   std::string& detail, bool open_form_only) {
    const auto has_artifact = [](const platform::CompiledGraphicsStage& stage) {
        return !stage.metallib.empty() || !stage.msl_source.empty() || !stage.msl_source_file.empty();
    };
    if (!key.vertex || !has_artifact(*key.vertex) || (key.fragment && !has_artifact(*key.fragment))) {
        detail = "draw requires resolved vertex and fragment artifacts";
        return nullptr;
    }
    NativePipeline result;
    result.vertex_stage = std::move(vertex_stage);
    result.fragment_stage = std::move(fragment_stage);
    if (result.vertex_stage.get() != key.vertex ||
        (key.fragment && result.fragment_stage.get() != key.fragment)) {
        detail = "draw pipeline stages differ from its key";
        return nullptr;
    }
    result.vertex_library = stage_library(result.vertex_stage, detail);
    if (!result.vertex_library) return nullptr;
    if (key.fragment) {
        result.fragment_library = stage_library(result.fragment_stage, detail);
        if (!result.fragment_library) return nullptr;
    }

    auto* vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = result.vertex_library;
    const auto& vertex_entry = key.rasterizer_discard && !key.vertex->discard_entry_point.empty()
        ? key.vertex->discard_entry_point : key.vertex->entry_point;
    vertex.name = [NSString stringWithUTF8String:vertex_entry.c_str()];
    auto* descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vertex;
    if (key.fragment && !key.rasterizer_discard)
        descriptor.fragmentFunctionDescriptor = fragment_function(key, result.fragment_library);
    descriptor.rasterizationEnabled = !key.rasterizer_discard;
    // Metal refuses a point-size export with a triangle or line topology
    // class but accepts it with the class unspecified.
    descriptor.inputPrimitiveTopology = key.vertex->writes_point_size
        ? MTLPrimitiveTopologyClassUnspecified : topology_class(key.primitive);
    const NSUInteger samples = key.samples > 1 ? key.samples : 1;
    descriptor.rasterSampleCount = samples;
    descriptor.alphaToCoverageState = key.alpha_to_coverage
        ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;

    result.vertex_slots.fill(-1);
    // A fetching vertex stage reads its inputs through its fetch table: the
    // pipeline has no vertex descriptor and binds no vertex streams.
    if (!key.vertex->vertex_fetch_buffer_index) {
        std::array<bool, 31> occupied{};
        const auto reserve = [&](std::uint32_t index) { if (index < occupied.size()) occupied[index] = true; };
        for (const auto& binding : key.vertex->constant_buffers) reserve(binding.metal_buffer_index);
        for (const auto& binding : key.vertex->constant_buffer_windows) reserve(binding.metal_buffer_index);
        for (const auto& binding : key.vertex->storage_buffers) reserve(binding.metal_buffer_index);
        for (const auto& resource : key.vertex->resources) {
            if (resource.kind != platform::CompiledShaderResourceKind::TexturePool &&
                resource.kind != platform::CompiledShaderResourceKind::SamplerPool)
                continue;
            if (resource.location.kind == platform::CompiledShaderResourceLocationKind::Direct &&
                resource.location.resource_namespace == platform::CompiledShaderResourceNamespace::Buffer)
                reserve(resource.location.first);
        }
        if (key.vertex->constant_buffer_support_index) reserve(*key.vertex->constant_buffer_support_index);
        if (key.vertex->draw_support_buffer_index) reserve(*key.vertex->draw_support_buffer_index);
        if (key.vertex->argument_buffer) reserve(key.vertex->argument_buffer->buffer_index);
        if (key.vertex->custom_border) reserve(key.vertex->custom_border->buffer_index);
        if (key.vertex->guest_memory) reserve(key.vertex->guest_memory->buffer_index);
        auto* vertices = [MTLVertexDescriptor vertexDescriptor];
        std::array<bool, 31> attribute_enabled{};
        for (std::size_t index = 0; index != key.attributes.size(); ++index) {
            const auto& attribute = key.attributes[index];
            const auto format = vertex_format(attribute.format);
            if (!format) {
                if (attribute.format == NVN_FORMAT_NONE) continue;
                detail = "vertex attribute format " + std::to_string(attribute.format) + " is unsupported";
                return nullptr;
            }
            if (attribute.stream >= key.streams.size()) {
                detail = "vertex attribute stream index exceeds the replay limit";
                return nullptr;
            }
            vertices.attributes[index].format = *format;
            vertices.attributes[index].offset = attribute.offset;
            attribute_enabled[index] = true;
            auto& slot = result.vertex_slots[attribute.stream];
            if (slot < 0) {
                const auto free = std::find(occupied.begin(), occupied.end(), false);
                if (free == occupied.end()) {
                    detail = "vertex streams exhaust the shader buffer namespace";
                    return nullptr;
                }
                slot = static_cast<std::int32_t>(free - occupied.begin());
                occupied[slot] = true;
            }
            vertices.attributes[index].bufferIndex = slot;
        }
        for (std::size_t index = 0; index != key.streams.size(); ++index) {
            const auto slot = result.vertex_slots[index];
            if (slot < 0) continue;
            const auto& stream = key.streams[index];
            vertices.layouts[slot].stride = stream.stride;
            vertices.layouts[slot].stepFunction = !stream.stride ? MTLVertexStepFunctionConstant :
                stream.divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
            vertices.layouts[slot].stepRate = stream.stride ? std::max(stream.divisor, 1u) : 0;
        }
        // A vertex input the shader reads that the vertex state leaves disabled
        // reads constant zero on the hardware, but Metal refuses to build the
        // pipeline unless every input location the shader declares is in the
        // descriptor. Feed those locations from one constant zero stream.
        const auto zero_input = [&](std::uint32_t location, int input_class) {
            if (location >= attribute_enabled.size() || attribute_enabled[location]) return;
            if (result.zero_attribute_slot < 0) {
                const auto free = std::find(occupied.begin(), occupied.end(), false);
                if (free == occupied.end()) return;
                result.zero_attribute_slot = static_cast<std::int32_t>(free - occupied.begin());
                occupied[result.zero_attribute_slot] = true;
                vertices.layouts[result.zero_attribute_slot].stride = 16;
                vertices.layouts[result.zero_attribute_slot].stepFunction =
                    MTLVertexStepFunctionConstant;
                vertices.layouts[result.zero_attribute_slot].stepRate = 0;
            }
            vertices.attributes[location].format =
                input_class == int(graphics::VertexFormatClass::uint_class) ? MTLVertexFormatUInt4 :
                input_class == int(graphics::VertexFormatClass::sint_class) ? MTLVertexFormatInt4 :
                MTLVertexFormatFloat4;
            vertices.attributes[location].offset = 0;
            vertices.attributes[location].bufferIndex = result.zero_attribute_slot;
        };
        for (const auto& input : key.vertex->vertex_class_inputs) {
            // A float32x4 fetch produces raw register bits for an integer input.
            // Preserve all four components without numeric conversion. Shorter
            // vectors need different default-component handling before adaptation.
            if (input.location < attribute_enabled.size() && attribute_enabled[input.location] &&
                input.input_class != graphics::VertexFormatClass::float_class) {
                auto* attribute = vertices.attributes[input.location];
                if (attribute.format == MTLVertexFormatFloat4)
                    attribute.format = input.input_class == graphics::VertexFormatClass::uint_class
                        ? MTLVertexFormatUInt4 : MTLVertexFormatInt4;
            }
            zero_input(input.location, int(input.input_class));
        }
        for (const auto& input : key.vertex->vertex_attributes)
            zero_input(input.attribute_index, int(graphics::VertexFormatClass::float_class));
        descriptor.vertexDescriptor = vertices;
    }

    if (!configure_color_attachments(descriptor.colorAttachments, key, detail)) return nullptr;
    auto compile = std::make_shared<PipelineCompile>();
    result.compile = compile;
    // The same pipeline with its color attachments left open (format, blending,
    // write mask): one archived pipeline serves every attachment configuration
    // of its stages, vertex layout and raster state, at the cost of a
    // specialization instead of a compile.
    MTL4RenderPipelineDescriptor* open = nil;
    static const bool open_forms = [] {
        const char* value = std::getenv("ARMIGO_PSO_OPEN");
        return !(value && *value == '0');
    }();
    NSArray* archives = loaded_archives();
    if (open_forms && key.fragment && (archives || serializer_)) {
        open = [descriptor copy];
        for (std::size_t index = 0; index != key.colors.size(); ++index) {
            auto* attachment = open.colorAttachments[index];
            attachment.pixelFormat = MTLPixelFormatUnspecialized;
            attachment.writeMask = MTLColorWriteMaskUnspecialized;
            attachment.blendingState = MTL4BlendStateUnspecialized;
        }
    }
    if (open_form_only) {
        if (!open || !serializer_) {
            detail = "an open-form row needs a fragment stage, open forms and archive capture";
            return nullptr;
        }
        if (!capture_open_form(key, open, detail)) return nullptr;
        return std::make_unique<NativePipeline>(std::move(result));
    }
    if (archives) {
        const auto lookup_begin = timeline_now_ns();
        for (id<MTL4Archive> archive in archives) {
            if (id<MTLRenderPipelineState> state = [archive newRenderPipelineStateWithDescriptor:descriptor
                                                                                        error:nil]) {
                ++archive_hits_;
                timeline_event("pso_archive_hit", 0, timeline_now_ns() - lookup_begin);
                compile->finish(state, {});
                return std::make_unique<NativePipeline>(std::move(result));
            }
        }
        if (open) {
            for (id<MTL4Archive> archive in archives) {
                id<MTLRenderPipelineState> flexible = [archive newRenderPipelineStateWithDescriptor:open error:nil];
                if (!flexible) continue;
                const auto specialize_begin = timeline_now_ns();
                NSError* error = nil;
                if (id<MTLRenderPipelineState> state =
                        [compiler_ newRenderPipelineStateBySpecializationWithDescriptor:descriptor
                                                                               pipeline:flexible
                                                                                  error:&error]) {
                    ++archive_hits_;
                    ++archive_open_hits_;
                    timeline_event("pso_archive_flex", 0, timeline_now_ns() - specialize_begin);
                    compile->finish(state, {});
                    return std::make_unique<NativePipeline>(std::move(result));
                }
                break;
            }
        }
    }
    if (archive_lookup_.load(std::memory_order_acquire))
        note_archive_miss(0, key.vertex->identity, key.fragment ? key.fragment->identity : "-");
    // A capturing build also archives the open form.
    if (open && serializer_) {
        std::string failure;
        if (!capture_open_form(key, open, failure)) {
            armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: open form not archived (%s %s): %s\n", key.vertex->identity.c_str(),
                         key.fragment->identity.c_str(), failure.c_str());
            const std::lock_guard lock(archive_keys_mutex_);
            unarchived_keys_.push_back(key.vertex->identity + " " + key.fragment->identity);
        }
    }
    const auto compile_begin = timeline_now_ns();
    // Metal's asynchronous compiles fail inside the compiler service while a
    // pipeline data set serializer captures them, so a capturing session
    // compiles synchronously on the calling thread (the warm-up's threads).
    if (serializer_) {
        NSError* error = nil;
        id<MTLRenderPipelineState> state = capture([&] {
            return [compiler_ newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:task_options_
                                                             error:&error];
        });
        timeline_event("pso_compiled", 0, timeline_now_ns() - compile_begin);
        std::string message;
        if (!state)
            message = std::string("render pipeline creation failed: ") +
                (error ? error.localizedDescription.UTF8String : "unknown Metal error");
        compile->finish(state, std::move(message));
        return std::make_unique<NativePipeline>(std::move(result));
    }
    [compiler_ newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:task_options_
        completionHandler:^(id<MTLRenderPipelineState> state, NSError* error) {
            // How long Metal took for this pipeline (an archive lookup or a full compile).
            timeline_event("pso_compiled", 0, timeline_now_ns() - compile_begin);
            std::string message;
            if (!state) {
                try {
                    message = std::string("render pipeline creation failed: ") +
                        (error ? error.localizedDescription.UTF8String : "unknown Metal error");
                } catch (...) {}
            }
            compile->finish(state, std::move(message));
        }];
    return std::make_unique<NativePipeline>(std::move(result));
}

std::optional<NativeComputePipeline> ReplayImpl::compute_pipeline(
    std::shared_ptr<const platform::CompiledGraphicsStage> stage,
    std::string& detail) {
    if (!stage) {
        detail = "dispatch requires one resolved compute artifact";
        return std::nullopt;
    }
    if (const auto found = compute_pipelines_.find(stage.get()); found != compute_pipelines_.end())
        return found->second;
    const auto create_begin = timeline_now_ns();
    std::optional<NativeComputePipeline> built;
    if (warm_ && !stage->identity.empty()) built = warm_->take_compute(stage->identity);
    const bool warmed = built.has_value();
    if (!built) built = build_compute_pipeline(stage, detail);
    if (!built) return std::nullopt;
    if (recorder_ && recorder_->enabled() && !stage->identity.empty()) {
        recorder_->note({'S', {stage->identity}});
        recorder_->note({'C', {stage->identity}});
    }
    timeline_event(warmed ? "pso_take" : "pso_compute", 1, timeline_now_ns() - create_begin);
    ++statistics_.compute_pipelines;
    compute_pipelines_.emplace(stage.get(), *built);
    return built;
}

// Safe off the replay worker, like build_pipeline.
std::optional<NativeComputePipeline> ReplayImpl::build_compute_pipeline(
    std::shared_ptr<const platform::CompiledGraphicsStage> stage,
    std::string& detail) {
    if (stage->stage != shader_input::Stage::Compute ||
        (stage->metallib.empty() && stage->msl_source.empty() && stage->msl_source_file.empty())) {
        detail = "dispatch requires one resolved compute artifact";
        return std::nullopt;
    }
    NativeComputePipeline result;
    result.stage = stage;
    result.library = load_stage_library(device_, *stage, detail);
    if (!result.library) {
        detail = "compute " + detail;
        return std::nullopt;
    }
    auto* function = [MTL4LibraryFunctionDescriptor new];
    function.library = result.library;
    function.name = [NSString stringWithUTF8String:stage->entry_point.c_str()];
    auto* descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    // The pipeline is built before a dispatch can check its limit, so the
    // block the stage runs with is asked for up front: the compile fails
    // where the hardware could not run it.
    const auto block = stage->compute_local_size;
    if (!block[0] || !block[1] || !block[2]) {
        detail = "compute local size is zero";
        return std::nullopt;
    }
    descriptor.maxTotalThreadsPerThreadgroup =
        NSUInteger(block[0]) * NSUInteger(block[1]) * NSUInteger(block[2]);
    auto compile = std::make_shared<ComputeCompile>();
    result.compile = compile;
    if (NSArray* archives = loaded_archives()) {
        const auto lookup_begin = timeline_now_ns();
        for (id<MTL4Archive> archive in archives) {
            if (id<MTLComputePipelineState> state = [archive newComputePipelineStateWithDescriptor:descriptor
                                                                                          error:nil]) {
                ++archive_hits_;
                timeline_event("pso_archive_hit", 1, timeline_now_ns() - lookup_begin);
                compile->finish(state, {});
                return result;
            }
        }
    }
    if (archive_lookup_.load(std::memory_order_acquire)) note_archive_miss(1, stage->identity, "-");
    const auto compile_begin = timeline_now_ns();
    if (serializer_) {   // synchronous while capturing, as for render pipelines
        NSError* error = nil;
        id<MTLComputePipelineState> state = capture([&] {
            return [compiler_ newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:task_options_
                                                              error:&error];
        });
        timeline_event("pso_compiled", 1, timeline_now_ns() - compile_begin);
        std::string message;
        if (!state)
            message = std::string("compute pipeline creation failed: ") +
                (error ? error.localizedDescription.UTF8String : "unknown Metal error");
        compile->finish(state, std::move(message));
        return result;
    }
    [compiler_ newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:task_options_
        completionHandler:^(id<MTLComputePipelineState> state, NSError* compile_error) {
            timeline_event("pso_compiled", 1, timeline_now_ns() - compile_begin);
            std::string message;
            if (!state) {
                try {
                    message = std::string("compute pipeline creation failed: ") +
                        (compile_error ? compile_error.localizedDescription.UTF8String : "unknown Metal error");
                } catch (...) {}
            }
            compile->finish(state, std::move(message));
        }];
    return result;
}

bool Encoder::fail(ReplayErrorCode code, std::string detail) {
    if (!failed) error = {code, std::move(detail)};
    failed = true;
    return false;
}

bool Encoder::flush_clears(std::size_t count) {
    count = std::min(count, deferred_clears.size());
    std::vector<DeferredClear> clears(std::make_move_iterator(deferred_clears.begin()),
                                      std::make_move_iterator(deferred_clears.begin() + count));
    deferred_clears.erase(deferred_clears.begin(), deferred_clears.begin() + count);
    for (const auto& clear : clears) {
        ClearBindings bindings{};
        bindings.level = clear.plan.target.level;
        bindings.slice = clear.plan.target.layer;
        if (clear.plan.color_mask) bindings.color = clear.texture;
        if (clear.plan.clear_depth) bindings.depth = clear.texture;
        if (clear.plan.clear_stencil) bindings.stencil = clear.texture;
        if (!encode_planned_clear(clear.plan, bindings, "")) return false;
    }
    return true;
}

// Encodes a planned clear into the submission's open command buffer. With
// the encode thread the plan's checks run here, so an unsupported clear still
// fails its command, and the Metal work runs there.
bool Encoder::encode_planned_clear(const graphics::ClearPlan& plan, ClearBindings bindings,
                                   const char* prefix) {
    note_clear(true);
    if (queue) {
        const auto checked = validate_clear(plan, bindings);
        if (const auto* failure = std::get_if<graphics::RendererError>(&checked))
            return fail(ReplayErrorCode::unsupported,
                        prefix + std::string(failure->command().detail));
    }
    id<MTLDevice> device = backend.device();
    // A block captures a reference as a reference: the plan goes by value.
    const graphics::ClearPlan planned = plan;
    // Separate clear passes bypass close_render. Capture the command identity
    // by value because the encode worker can execute this call after traversal.
    const std::uint64_t trace_clear = trace_submission_id ? ++trace_encoder_sequence : 0;
    const std::uint64_t trace_submission = trace_submission_id;
    const std::uint64_t trace_fence = pending.fence;
    const std::size_t trace_index = records_seen;
    const unsigned trace_op = unsigned(current_op_);
    return encode_call(^(EncodeCallContext& context) {
        auto into = bindings;
        into.into = context.commands;
        into.into_residency = context.residency;
        auto result = encode_clear(device, planned, into);
        // A clear that had to compile its shader or pipeline (not prepared at
        // install or from a warm-up row) is counted like Armigo's other own
        // pipelines, and its configuration is recorded for the next warm-up.
        // A wait for a compile another thread ran (the warm-up's) is timed as
        // a pipeline wait.
        for (const auto& compile : take_clear_compiles()) {
            timeline_event(compile.wait ? "pso_wait" : "pso_compiled", 2, compile.nanoseconds);
            if (compile.pipeline && !compile.wait) backend.note_clear_pipeline(*compile.pipeline);
        }
        if (const auto* failure = std::get_if<graphics::RendererError>(&result)) {
            context.error = prefix + std::string(failure->command().detail);
            return;
        }
        const auto& encoded = std::get<EncodedClear>(result);
        if (trace_clear) {
            const char* reason = planned.fold_into_load || planned.fold_layers_into_load
                ? "clear_load" : "clear_draw";
            for (const char* event : {"begin", "end"})
                armigo::diag::opt_in(
                    "nvn_encoder_trace trace=%llu fence=%llu index=%zu op=%u encoder=%llu event=%s reason=%s\n",
                    static_cast<unsigned long long>(trace_submission),
                    static_cast<unsigned long long>(trace_fence), trace_index, trace_op,
                    static_cast<unsigned long long>(trace_clear), event, reason);
        }
        if (encoded.retained) [context.retained addObjectsFromArray:encoded.retained];
    }, ReplayErrorCode::unsupported);
}

bool Encoder::apply_clears(MTL4RenderPassDescriptor* pass) {
    // A clear folds only when it names an attached view. Clears carried over
    // a render-target change may name views this pass does not attach: those
    // land first, in order, and every clear before the last such one lands
    // with them so the original order of overlapping clears is kept.
    const auto foldable = [&](const DeferredClear& clear) {
        bool matches = !clear.plan.color_mask;
        for (unsigned i = 0; i < 8; ++i)
            matches |= pass.colorAttachments[i].texture == clear.texture;
        if (!matches || (clear.plan.clear_depth && pass.depthAttachment.texture != clear.texture) ||
            (clear.plan.clear_stencil && pass.stencilAttachment.texture != clear.texture) ||
            clear.plan.target.layer_count != 1 || clear.plan.target.layer != 0)
            return false;
        // The load action clears only the pass's render area, the smallest
        // attachment: a larger cleared view keeps its separate clear.
        if (clear.texture.width != attachment_width || clear.texture.height != attachment_height)
            return false;
        if (clear.plan.color_mask) {
            const auto values = metal_clear_color(clear.plan.color_bits, clear.plan.target.texture.format);
            if (!std::get_if<std::array<double, 4>>(&values)) return false;
        }
        return true;
    };
    const auto landing = clears_to_land(deferred_clears.size(),
        [&](std::size_t index) { return foldable(deferred_clears[index]); });
    if (landing && !flush_clears(landing)) return false;
    for (const auto& clear : deferred_clears) {
        note_clear(false);
        if (clear.plan.color_mask) {
            const auto values = metal_clear_color(clear.plan.color_bits, clear.plan.target.texture.format);
            const auto* color = std::get_if<std::array<double, 4>>(&values);
            for (unsigned i = 0; i < 8; ++i) {
                auto* attachment = pass.colorAttachments[i];
                if (attachment.texture != clear.texture) continue;
                attachment.loadAction = MTLLoadActionClear;
                attachment.clearColor = MTLClearColorMake((*color)[0], (*color)[1], (*color)[2], (*color)[3]);
            }
        }
        if (clear.plan.clear_depth) {
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.clearDepth = clear.plan.depth_value;
        }
        if (clear.plan.clear_stencil) {
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.clearStencil = clear.plan.stencil_value;
        }
    }
    deferred_clears.clear();
    return true;
}

void Encoder::trace_pass_event(const char* event, const char* reason,
                               std::uint64_t a, std::uint64_t b,
                               std::uint64_t c, std::uint64_t d) const {
    if (!trace_submission_id) return;
    // Report values: address, effect index, first slot, end slot. Predicate
    // reads: operand index, first slot, end slot, address. Slots are half-open;
    // finalized names the end of the prefix already closed into earlier passes.
    armigo::diag::opt_in(
        "nvn_encoder_trace trace=%llu fence=%llu index=%zu op=%u encoder=%llu event=%s reason=%s conditional=%llu "
        "open=%u predicate=%u visibility=%zu finalized=%zu reset=%zu a=%llu b=%llu c=%llu d=%llu\n",
        static_cast<unsigned long long>(trace_submission_id),
        static_cast<unsigned long long>(pending.fence), records_seen, unsigned(current_op_),
        static_cast<unsigned long long>(trace_active_encoder), event, reason,
        static_cast<unsigned long long>(trace_conditional), unsigned(render != nil), unsigned(predicate_open()),
        visibility_cursor, finalized_visibility, reset_slot,
        static_cast<unsigned long long>(a), static_cast<unsigned long long>(b),
        static_cast<unsigned long long>(c), static_cast<unsigned long long>(d));
}

// Labels an encoder the profiled frame opens and stamps its start; `encoder`
// is nil when the encode thread makes it, which then stamps it.
int Encoder::profile_begin(char kind, MTL4RenderPassDescriptor* pass, id encoder) {
    auto* profile = backend.gpu_profile();
    if (!profile) return -1;
    GpuProfile::Label label;
    label.kind = kind;
    if (kind == 'R') {
        const auto& task = state.stage_programs[NVN_SHADER_STAGE_TESS_CONTROL];
        const auto* stage = task ? task->stages[NVN_SHADER_STAGE_TESS_CONTROL].get() : nullptr;
        label.begin_stage = mesh_draw ? (stage && stage->task ? MTLRenderStageObject : MTLRenderStageMesh)
                                      : MTLRenderStageVertex;
    }
    label.fence = pending.fence;
    label.segment = static_cast<std::uint32_t>(pending.segments_committed);
    label.pass = pending.counters.passes;
    label.draws = pending.counters.draws;
    label.dispatches = pending.counters.dispatches;
    label.fills = pending.counters.fill_calls;
    const auto stage_identity = [&](std::size_t stage) -> const std::string* {
        const auto& program = state.stage_programs[stage];
        const auto* shader = program ? program->stages[stage].get() : nullptr;
        return shader && !shader->identity.empty() ? &shader->identity : nullptr;
    };
    const std::string* identity = nullptr;
    if (kind == 'R') {
        for (const auto stage : {NVN_SHADER_STAGE_FRAGMENT, NVN_SHADER_STAGE_TESS_EVALUATION, NVN_SHADER_STAGE_VERTEX})
            if (!identity) identity = stage_identity(stage);
    } else if (kind == 'C') {
        identity = stage_identity(NVN_SHADER_STAGE_COMPUTE);
    }
    if (identity)
        std::snprintf(label.program.data(), label.program.size(), "%s", identity->c_str());
    if (kind == 'R') {
        const auto* geometry = stage_identity(NVN_SHADER_STAGE_TESS_EVALUATION);
        if (!geometry) geometry = stage_identity(NVN_SHADER_STAGE_VERTEX);
        if (geometry) std::snprintf(label.geometry.data(), label.geometry.size(), "%s", geometry->c_str());
    }
    if (pass) {
        for (NSUInteger index = 0; index < 8; ++index) {
            id<MTLTexture> texture = pass.colorAttachments[index].texture;
            if (!texture) continue;
            ++label.colors;
            label.width = static_cast<std::uint32_t>(std::max<NSUInteger>(texture.width >> pass.colorAttachments[index].level, 1));
            label.height = static_cast<std::uint32_t>(std::max<NSUInteger>(texture.height >> pass.colorAttachments[index].level, 1));
        }
        if (id<MTLTexture> depth = pass.depthAttachment.texture) {
            label.depth = true;
            label.width = static_cast<std::uint32_t>(std::max<NSUInteger>(depth.width >> pass.depthAttachment.level, 1));
            label.height = static_cast<std::uint32_t>(std::max<NSUInteger>(depth.height >> pass.depthAttachment.level, 1));
        }
    }
    const int slot = profile->begin(label);
    if (slot < 0) return -1;
    const auto index = std::size_t(slot) * GpuProfile::timestamp_slots;
    if (queue)
        queue->put(EncodeOp::profile_stamp, kind == 'R' ? 1u | (std::uint32_t(label.begin_stage) << 2) : 0u,
                   TimestampRecord{(__bridge void*)profile->heap(), index});
    else if (kind == 'R')
        GpuProfile::stamp_render((id<MTL4RenderCommandEncoder>)encoder, profile->heap(), index, false,
                                 label.begin_stage);
    else
        GpuProfile::stamp_compute((id<MTL4ComputeCommandEncoder>)encoder, profile->heap(), index);
    return slot;
}

int Encoder::profile_call(int slot) {
    auto* profile = backend.gpu_profile();
    if (!profile) return -1;
    std::size_t index = 0;
    if (slot < 0) {
        GpuProfile::Label label;
        label.kind = 'X';
        label.fence = pending.fence;
        label.segment = static_cast<std::uint32_t>(pending.segments_committed);
        label.pass = pending.counters.passes;
        slot = profile->begin(label);
        if (slot < 0) return -1;
        index = std::size_t(slot) * GpuProfile::timestamp_slots;
    } else {
        profile->mark_call_end(slot);
        index = std::size_t(slot) * GpuProfile::timestamp_slots + 1;
    }
    if (queue) queue->put(EncodeOp::timestamp, 0, TimestampRecord{(__bridge void*)profile->heap(), index});
    else [pending.commands writeTimestampIntoHeap:profile->heap() atIndex:index];
    return slot;
}

void Encoder::profile_end(int& slot, char kind, id encoder) {
    if (slot < 0) return;
    auto* profile = backend.gpu_profile();
    if (!profile) return;
    profile->end(slot, pending.counters.draws, pending.counters.dispatches, pending.counters.fill_calls);
    const auto index = std::size_t(slot) * GpuProfile::timestamp_slots + 1;
    if (queue)
        queue->put(EncodeOp::profile_stamp, kind == 'R' ? 3u : 0u, TimestampRecord{(__bridge void*)profile->heap(), index});
    else if (kind == 'R')
        GpuProfile::stamp_render((id<MTL4RenderCommandEncoder>)encoder, profile->heap(), index, true);
    else
        GpuProfile::stamp_compute((id<MTL4ComputeCommandEncoder>)encoder, profile->heap(), index);
    slot = -1;
}

bool Encoder::close_render(PassBreak why, bool keep_clears) {
    if (render) {
        trace_pass_event("end", pass_break_names[std::size_t(why)]);
        finalized_visibility = visibility_cursor;
        if (why == PassBreak::depth_sampled)
            ++backend.mutable_statistics().pass_break_depth_sampled;
        note_pass_break(why);
        // Why a pass ended is what decides whether two passes with the same
        // attachments could have been one encoder.
        static const bool close_reason = [] {
            const char* value = std::getenv("ARMIGO_NVN_PASS_GEOMETRY");
            return value && *value == '1';
        }();
        if (close_reason)
            armigo::diag::opt_in( "nvn_pass_close frame=%llu op=%u to=%u\n",
                         static_cast<unsigned long long>(pending.fence),
                         unsigned(current_op_), unsigned(pending.counters.passes));
        if (std::exchange(pass_relaxed, false)) ++backend.mutable_statistics().relaxed_pass_barriers;
        relaxed_producer = game_barriers();
        profile_end(profile_render, 'R', render);
        end_encoder(render, MTLStageAll, relaxed_producer);
        after_render_pass = !pass_storage_writes;
        render = nil;
        feedback_written = feedback_direct_reads = 0;
        feedback_depth_written = feedback_depth_direct_reads = false;
        if (backend.pass_dump_enabled() && backend.pass_dump_wants(backend.pool_present_count()))
            dump_render_pass(why);
        pass_depth_only = true;
        depth_barrier_allocation = nil;
        depth_barrier_reads = false;
        depth_omitted = false;
        attachmentless_pass = false;
        if (!copy_block_aliases(false)) return false;
        block_aliases.fill({});
        fixed_valid = 0;
        dirty.begin_pass();
        encoded_pipeline = nullptr;
        render_primitive = -1;
        pass_visibility = std::numeric_limits<std::size_t>::max();
        pass_visibility_buffer = false;
    }
    invalidate_args_all();
    return keep_clears || flush_clears();
}

namespace {

std::string stage_label(const platform::CompiledGraphicsStage* shader) {
    if (!shader) return "-";
    if (!shader->identity.empty()) return shader->identity;
    if (!shader->entry_point.empty()) return shader->entry_point;
    return "?";
}

bool dump_input_shader_matches(const platform::CompiledGraphicsStage* shader, std::string_view filters) {
    if (filters.empty()) return true;
    if (!shader) return false;
    while (!filters.empty()) {
        const auto comma = filters.find(',');
        const auto filter = filters.substr(0, comma);
        if (!filter.empty() && shader->identity.find(filter) != std::string::npos) return true;
        if (comma == std::string_view::npos) break;
        filters.remove_prefix(comma + 1);
    }
    return false;
}

void append_attachment(std::string& text, const PassDumpImage& image) {
    if (!text.empty()) text += ' ';
    text += image.kind;
    text += ':';
    text += std::to_string(image.serial);
    text += ':';
    text += image.format.empty() ? "?" : image.format;
    text += ':';
    text += std::to_string(image.width);
    text += 'x';
    text += std::to_string(image.height);
    char native[48];
    std::snprintf(native, sizeof native, ":native_id:%#llx", static_cast<unsigned long long>(image.native_id));
    text += native;
    std::snprintf(native, sizeof native, ":root_id:%#llx", static_cast<unsigned long long>(image.native_root_id));
    text += native;
}

} // namespace

std::vector<std::string> Encoder::dump_binding_metadata(std::size_t stage) const {
    std::vector<std::string> rows;
    const auto& bound = bindings[stage];
    const auto take = [&](const auto& slots, const char* kind) {
        for (std::size_t index = 0; index < slots.size(); ++index) {
            const auto& value = slots[index];
            if (!value.present) continue;
            char text[192];
            std::snprintf(text, sizeof text,
                "binding submission_draw=%llu stage=%zu kind=%s index=%zu gpu=%#llx stride=%u",
                static_cast<unsigned long long>(pending.counters.draws), stage, kind, index,
                static_cast<unsigned long long>(value.value), value.stride);
            rows.emplace_back(text);
        }
    };
    take(bound.buffers, "buffer");
    take(bound.textures, "texture");
    take(bound.samplers, "sampler");
    return rows;
}

void Encoder::dump_render_pass(PassBreak why) {
    auto reads = std::exchange(render_dump_reads, {});
    auto draws = std::exchange(render_dump_draws, {});
    auto inputs = std::exchange(render_dump_textures, {});
    auto buffers = std::exchange(render_dump_buffers, {});
    auto& dump = backend.pass_dump();
    if (dump.stopped() || !dump.ensure_directory() || !render_descriptor) return;
    PassDumpLine line;
    line.directory = dump.directory();
    line.metadata_only = dump.metadata_only();
    line.frame = backend.pool_present_count();
    line.index = dump.next_index(line.frame);
    line.why = pass_break_names[std::size_t(why)];
    line.draws = pending.counters.draws - pass_draws_at_open;
    line.vs = stage_label(bindings[NVN_SHADER_STAGE_VERTEX].shader);
    line.fs = stage_label(bindings[NVN_SHADER_STAGE_FRAGMENT].shader);
    line.cs = "-";
    line.reads.assign(reads.begin(), reads.end());
    line.draw_metadata = std::move(draws);
    std::vector<PassDumpJob> jobs;
    const auto take = [&](std::string kind, id<MTLTexture> texture, id<MTLTexture> resolve, MTLStoreAction store,
                          NSUInteger level, NSUInteger slice, NSUInteger plane, NSUInteger resolve_level,
                          NSUInteger resolve_slice, NSUInteger resolve_plane, std::uint64_t serial) {
        if (store == MTLStoreActionDontCare || store == MTLStoreActionUnknown) return;
        id<MTLTexture> stored = texture;
        NSUInteger stored_level = level, stored_slice = slice, stored_plane = plane;
        if (texture && texture.sampleCount > 1 && resolve && resolve.sampleCount <= 1 &&
            (store == MTLStoreActionMultisampleResolve || store == MTLStoreActionStoreAndMultisampleResolve)) {
            stored = resolve;
            stored_level = resolve_level;
            stored_slice = resolve_slice;
            stored_plane = resolve_plane;
        }
        if (!stored) return;
        auto job = dump.prepare(stored, std::move(kind), serial, stored_level, stored_slice, stored_plane);
        append_attachment(line.attachments, job.image);
        line.images.push_back(job.image);
        if (job.pipeline) {
            make_resident(pending, job.sample);
            make_resident(pending, job.blit_from);
            make_resident(pending, job.image.pixels);
            make_resident(pending, job.image.raw);
            make_resident(pending, job.args);
            make_resident(pending, job.bound_raw);
            jobs.push_back(std::move(job));
        }
    };
    for (unsigned index = 0; index < 8; ++index) {
        auto* attachment = render_descriptor.colorAttachments[index];
        const auto* texture_state = index < static_cast<unsigned>(std::clamp(state.targets.count, 0, 8))
                                        ? state.targets.color[index] : nullptr;
        take("c" + std::to_string(index), attachment.texture, attachment.resolveTexture, attachment.storeAction,
             attachment.level, attachment.slice, attachment.depthPlane, attachment.resolveLevel, attachment.resolveSlice,
             attachment.resolveDepthPlane, texture_state ? texture_state->serial : 0);
    }
    if (render_descriptor.depthAttachment.texture || render_descriptor.depthAttachment.resolveTexture) {
        auto* attachment = render_descriptor.depthAttachment;
        take("d", attachment.texture, attachment.resolveTexture, attachment.storeAction, attachment.level,
             attachment.slice, attachment.depthPlane, attachment.resolveLevel, attachment.resolveSlice,
             attachment.resolveDepthPlane, state.targets.depth ? state.targets.depth->serial : 0);
    }
    for (auto& job : buffers) {
        line.images.push_back(job.image);
        if (!job.pipeline) continue;
        make_resident(pending, job.image.raw);
        make_resident(pending, job.args);
        jobs.push_back(std::move(job));
    }
    for (const auto& input : inputs) {
        if (!dump.first_input(input.texture)) continue;
        const auto type = input.texture.textureType;
        const NSUInteger slices = type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray
            ? input.texture.arrayLength * 6 : type == MTLTextureType2DArray ? input.texture.arrayLength : 1;
        for (NSUInteger slice = 0; slice < std::min<NSUInteger>(slices, 64); ++slice) {
            auto job = dump.prepare(input.texture, input.kind + "-s" + std::to_string(slice), input.serial,
                                    0, slice, 0, true);
            line.images.push_back(job.image);
            if (job.pipeline) {
                make_resident(pending, job.sample);
                make_resident(pending, job.blit_from);
                make_resident(pending, job.image.pixels);
                make_resident(pending, job.image.raw);
                make_resident(pending, job.args);
                make_resident(pending, job.bound_raw);
                jobs.push_back(std::move(job));
            }
        }
    }
    if (!jobs.empty()) {
        auto held = std::make_shared<std::vector<PassDumpJob>>(std::move(jobs));
        encode_call(^(EncodeCallContext& context) {
            encode_pass_dump_jobs(context.commands, context.residency, context.retained, *held);
        }, ReplayErrorCode::allocation);
    }
    pending.pass_dumps.push_back(std::move(line));
}

void Encoder::dump_compute_dispatch() {
    auto& dump = backend.pass_dump();
    if (dump.stopped() || !dump.ensure_directory() || (!dump.metadata_only() && compute_dump_images.empty())) return;
    PassDumpLine line;
    line.directory = dump.directory();
    line.metadata_only = dump.metadata_only();
    line.frame = backend.pool_present_count();
    line.index = dump.next_index(line.frame);
    line.compute = true;
    line.why = pass_break_names[std::size_t(PassBreak::dispatch)];
    line.vs = "-";
    line.fs = "-";
    line.cs = stage_label(bindings[NVN_SHADER_STAGE_COMPUTE].shader);
    line.reads = bindings[NVN_SHADER_STAGE_COMPUTE].dump_reads;
    if (dump.metadata_only()) {
        auto metadata = dump_binding_metadata(NVN_SHADER_STAGE_COMPUTE);
        line.reads.insert(line.reads.end(), std::make_move_iterator(metadata.begin()), std::make_move_iterator(metadata.end()));
    }
    std::vector<PassDumpJob> jobs;
    if (const auto* layout = compute_layout()) {
        std::string detail = "compute_driver texture_base=" + std::to_string(layout->texture_base()) +
            " separate_texture_base=" + std::to_string(layout->separate_texture_base()) +
            " image_base=" + std::to_string(layout->image_base()) + " size=" + std::to_string(layout->size);
        for (std::size_t index = 0; index < layout->images && index < state.images_at(NVN_SHADER_STAGE_COMPUTE).size(); ++index)
            detail += " image" + std::to_string(index) + "=" + std::to_string(state.images_at(NVN_SHADER_STAGE_COMPUTE)[index]);
        line.reads.push_back(std::move(detail));
        const auto& stage = bindings[NVN_SHADER_STAGE_COMPUTE];
        std::uint64_t address = 0, bytes = 0;
        const auto select = [&](std::uint32_t index, std::uint64_t size) {
            if (index < stage.buffers.size() && stage.buffers[index].present) {
                address = stage.buffers[index].value;
                bytes = std::min<std::uint64_t>(layout->size, size);
            }
        };
        if (stage.shader) {
            for (const auto& buffer : stage.shader->constant_buffers)
                if (buffer.guest_binding == 0)
                    select(buffer.metal_buffer_index, buffer.minimum_size ? buffer.minimum_size : layout->size);
            for (const auto& buffer : stage.shader->constant_buffer_windows)
                if (buffer.guest_binding == 0) select(buffer.metal_buffer_index, buffer.window_bytes);
        }
        auto job = dump.prepare_buffer("compute-bound-c0", address, bytes);
        line.images.push_back(job.image);
        if (job.pipeline) {
            make_resident(pending, job.image.raw);
            make_resident(pending, job.args);
            jobs.push_back(std::move(job));
        }
    }
    for (const auto& image : compute_dump_images) {
        auto job = dump.prepare(image.texture, "img" + std::to_string(image.slot), image.serial, 0, 0, 0);
        append_attachment(line.attachments, job.image);
        line.images.push_back(job.image);
        if (job.pipeline) {
            make_resident(pending, job.sample);
            make_resident(pending, job.blit_from);
            make_resident(pending, job.image.pixels);
            make_resident(pending, job.image.raw);
            make_resident(pending, job.args);
            make_resident(pending, job.bound_raw);
            jobs.push_back(std::move(job));
        }
    }
    if (!jobs.empty()) {
        auto held = std::make_shared<std::vector<PassDumpJob>>(std::move(jobs));
        encode_call(^(EncodeCallContext& context) {
            encode_pass_dump_jobs(context.commands, context.residency, context.retained, *held);
        }, ReplayErrorCode::allocation);
    }
    pending.pass_dumps.push_back(std::move(line));
}

bool Encoder::close_transfer() {
    if (!transfer) return true;
    profile_end(profile_transfer, 'C', transfer);
    end_encoder(transfer, MTLStageBlit | MTLStageDispatch);
    transfer = nil;
    return true;
}

bool Encoder::barrier(PassBreak why) {
    barrier_requested = true;
    close_render(why);
    close_transfer();
    return true;
}

// Depth testing is ordered within a render pass, but texture sampling must see
// the stored depth. Match the underlying allocation so views cannot bypass it.
void Encoder::note_depth_barrier_read(__unsafe_unretained id<MTLTexture> texture) {
    if (!depth_barrier_allocation || !texture) return;
    while (__unsafe_unretained id<MTLTexture> parent = texture.parentTexture)
        texture = parent;
    depth_barrier_reads |= texture == depth_barrier_allocation;
}

// A draw that samples one of its pass's own attachments needs the full
// boundary at a later barrier: a draw after it in a continued pass could write
// that attachment before the sample reads it.
void Encoder::note_attachment_sample(__unsafe_unretained id<MTLTexture> texture) {
    if (!pass_depth_only || !render_descriptor || !texture) return;
    __unsafe_unretained id<MTLTexture> sampled = texture;
    while (__unsafe_unretained id<MTLTexture> parent = sampled.parentTexture) sampled = parent;
    for (unsigned index = 0; index != attachment_root_count && pass_depth_only; ++index)
        pass_depth_only = attachment_roots[index] != sampled;
}

void Encoder::note_writes() {
    while (pending.notified_textures < pending.texture_writes.size())
        backend.resources().note_gpu_write(pending.texture_writes[pending.notified_textures++]);
    while (pending.notified_buffers < pending.buffer_writes.size()) {
        const auto& [address, size] = pending.buffer_writes[pending.notified_buffers++];
        if (size) backend.resources().note_gpu_write(address, size);
    }
}

void Encoder::note_shader_writes(std::size_t stage) {
    other_buffer_writes.insert(other_buffer_writes.end(), bindings[stage].writes.begin(), bindings[stage].writes.end());
    pending.buffer_writes.insert(pending.buffer_writes.end(), bindings[stage].writes.begin(), bindings[stage].writes.end());
    for (const auto& [address, size] : bindings[stage].writes) discard_gpu_uniform_captures(address, size);
    note_report_writes(stage);
}

// The report table's view of a stage's storage and guest-memory stores.
void Encoder::note_report_writes(std::size_t stage) {
    if (!reports) return;
    bool unknown = false;
    for (const auto& [address, size] : bindings[stage].writes) {
        if (!size) unknown = true;
        else reports->gpu_write(address, size);
    }
    // A guest-memory store, a writable stage with no recorded range, or a
    // zero-sized binding can replace any counter. The epoch covers those
    // after the ranges that were named.
    const auto* shader = bindings[stage].shader;
    const bool guest_stores = shader && shader->guest_memory && shader->guest_memory->writes;
    if (guest_stores || (bindings[stage].writes_buffers && bindings[stage].writes.empty()) || unknown)
        reports->gpu_write_unbounded();
}

bool Encoder::order_buffer_writes() {
    const auto writer = buffer_write_order.consume();
    if (writer == DrawWriteOrder::Writer::none || !render) return true;
    // A render encoder's barrier can wait only for vertex-stage work: a
    // fragment of an earlier draw may still run after later draws' work in
    // the same pass. Fragment-stage (or other) buffer writes end the pass, and
    // the next encoder's queue barrier orders all of its stages after them.
    if (writer == DrawWriteOrder::Writer::vertex) {
        ++pending.counters.buffer_barriers;
        encoder_barrier(render, MTLStageVertex, MTLStageVertex | MTLStageFragment);
        return true;
    }
    return close_render(PassBreak::buffer_writes);
}

bool Encoder::restart_segment() {
    // Sequential command buffers append to one allocator. Its memory and all
    // residency/arena ownership remain tied to the final guest fence.
    if (!backend.begin_commands(pending))
        return fail(ReplayErrorCode::allocation, "replay segment allocation failed");
    // The new command buffer has no residency sets yet. Versions already held
    // for the submission are attached again the next time a stage binds them.
    pool_residency_attached.clear();
    cut_draws = pending.counters.draws;
    return true;
}

bool Encoder::profile_cut(std::string label) {
    if (failed || render || !pending.commands_open || predicate_open()) return !failed;
    if (!close_transfer()) return false;
    note_writes();
    if (pending.segment_labels.size() <= pending.segments_committed)
        pending.segment_labels.resize(pending.segments_committed + 1);
    pending.segment_labels[pending.segments_committed] = std::move(label);
    backend.commit_segment(pending, SegmentKind::cut);
    note_segment_cut();
    return restart_segment();
}

bool Encoder::cut_submission() {
    // A conditional block's kernel must run after all of its argument
    // records are written: the block never spans a commit.
    if (failed || render || !deferred_clears.empty() || !pending.commands_open || predicate_open() ||
        !backend.admit_cut(pending.counters.draws - cut_draws)) return !failed;
    if (!close_transfer()) return false;
    note_writes();
    backend.commit_segment(pending, SegmentKind::cut);
    note_segment_cut();
    return restart_segment();
}

bool Encoder::synchronize_gpu() {
    // An open conditional block ends with this commit; once the GPU has run
    // it, the counters are in memory and the rest of the block reads them.
    const bool resolve = predicate_open();
    predicate.arguments = nullptr;
    predicate.job = 0;
    if (!synchronize_gpu_commit()) return false;
    return !resolve || read_predicate(predicate.mode, predicate.address);
}

bool Encoder::synchronize_gpu_commit() {
    const SiteScope site_scope(*this, TransferSite::synchronize);
    ++backend.mutable_statistics().synchronize_gpu_waits;
    ++gpu_synchronizations;
    if (!pending.commands_open)
        return fail(ReplayErrorCode::encoding, "CPU drain has no open command buffer");
    if (!flush_uniforms() || !close_render(PassBreak::cpu_drain) || !close_transfer()) return false;
    note_writes();
    backend.commit_segment(pending, SegmentKind::drain);
    // The next pass follows a CPU wait: it takes the full barrier.
    after_render_pass = relaxed_producer = false;
    const auto execution_error = backend.wait_segments(pending);
    if (execution_error) return fail(execution_error.code, execution_error.detail);
    // CPU effects from earlier guest submissions precede this prefix's effects.
    backend.wait_prior_effects(pending);
    retired_fence = backend.completed_fence();
    backend.apply_effects(pending);
    pending.committed_effects = 0;
    // The prefix's guest-memory stores are in memory: textures with storage of
    // their own over the pools they reached upload again at their next use.
    if (guest_writes_pending) backend.publish_guest_writes(pending, true);
    // Applying cleared pending.effects: the reports are in memory now, and
    // later effects reuse the indices that gpu_counters names.
    gpu_counters.clear();
    effect_span.clear();
    if (state.samples_active && counter_begin != visibility_cursor) {
        backend.accumulate_samples(pending.visibility, counter_begin, visibility_cursor);
        counter_begin = visibility_cursor;
    }
    buffer_write_order.clear();
    cpu_writes_completed = cpu_writes_indexed = other_buffer_writes.size();
    guest_writes_pending = false;
    current_cpu_writes.clear();
    cpu_writes_unknown = false;
    uniform_copy_mappings.clear();
    uniform_copy_count = 0;
    uniform_copy_unknown = false;
    return restart_segment();
}

void Encoder::drain_failed_prefix() {
    // A failed record does not undo earlier commits. Try to land its valid
    // encoded prefix, then preserve only effects whose commands were submitted.
    const auto original = error;
    failed = false;
    error = {};
    const bool drained = pending.commands_open && synchronize_gpu();
    if (!drained) {
        if (pending.commands_open) {
            close_render(PassBreak::cpu_drain);
            close_transfer();
            backend.end_commands(pending);
        }
        const auto execution_error = backend.wait_segments(pending);
        backend.wait_prior_effects(pending);
        pending.effects.resize(std::min(pending.effects.size(), pending.committed_effects));
        if (!execution_error) backend.apply_effects(pending);
    }
    // synchronize_gpu starts a new empty segment on success; it has no work to
    // submit. All earlier segments have drained and remain owned until sealing.
    if (pending.commands_open) backend.end_commands(pending);
    // Sealing may retire the fence and hand its upload memory to the next
    // submission: every record of this one, uniform copies included, has run
    // by then.
    if (queue) queue->drain();
    pending.effects.clear();
    pending.committed_effects = 0;
    gpu_counters.clear();
    effect_span.clear();
    failed = true;
    error = original;
}

bool Encoder::sync_draw_resources() {
    const auto registration = backend.resources().registration_generation();
    const auto writes = memory::WriteTracker::now();
    if (registration != dirty.registration) ++pending.counters.syncs_registration;
    else if (writes != dirty.cpu_writes) ++pending.counters.syncs_writes;
    const bool changed = dirty.sync_resources(registration, writes);
    if (changed)
        for (auto& stage : bindings) stage.uniform_slots = UINT32_MAX;
    return changed;
}

bool Encoder::report_destination_isolated(std::uint64_t address, std::uint64_t size) {
    if (!size || size > UINT64_MAX - address) return false;
    const auto end = address + size;
    const auto overlaps = [&](std::uint64_t begin, std::uint64_t length) {
        return begin && length && (length > UINT64_MAX - begin || (address < begin + length && begin < end));
    };
    for (const auto& buffer : state.vertex_buffers)
        if (overlaps(buffer.address, buffer.size)) return false;
    if (overlaps(state.index_buffer.address, state.index_buffer.size)) return false;
    for (const auto& stage : state.uniform)
        for (const auto& buffer : stage)
            if (overlaps(buffer.address, buffer.size)) return false;
    for (const auto& stage : state.storage)
        for (const auto& buffer : stage)
            if (overlaps(buffer.address, buffer.size)) return false;
    for (const auto* target : state.targets.color)
        if (target && overlaps(reinterpret_cast<std::uint64_t>(target->memory), target->size)) return false;
    if (state.targets.depth && overlaps(reinterpret_cast<std::uint64_t>(state.targets.depth->memory),
                                       state.targets.depth->size)) return false;
    for (const auto& clear : deferred_clears)
        if (overlaps(clear.backing_begin, clear.backing_size)) return false;
    const auto first_word = address & ~std::uint64_t{3};
    const auto last_word = (end - 1) & ~std::uint64_t{3};
    if (last_word > UINT64_MAX - 4 ||
        captured_guest_bytes.may_intersect(first_word, last_word + 4 - first_word)) return false;
    // A negative result for the whole page is reusable until registrations
    // change. Unknown or boundary-spanning storage keeps the ordered path.
    const auto page = address >> 12;
    if (((end - 1) >> 12) != page) return false;
    const auto generation = backend.resources().registration_generation();
    auto& cached = report_pages[page & (report_pages.size() - 1)];
    if (cached.page != page || cached.generation != generation) {
        cached = {page, generation, backend.resources().overlaps_texture_backing(page << 12, 4096)};
    }
    return !cached.aliases && !nvn::window_texture_overlaps(address, size);
}

bool guest_ranges_overlap(const std::vector<std::pair<std::uint64_t, std::uint64_t>>* ranges,
                          std::uint64_t address, std::uint64_t size);

bool Encoder::order_texture_backing(std::uint64_t begin, std::uint64_t size, bool& ordered) {
    if (!begin || !size) return true;
    if (size > UINT64_MAX - begin)
        return fail(ReplayErrorCode::missing_resource, "texture backing range overflows");
    // A title whose stores reach textures at retirement uses the texels the
    // texture holds and never waits for a store here, so whether the backing
    // feeds a texture, and which predictions the use relied on, does not
    // matter: only report writes into the range are ordered.
    const bool ordered_stores =
        nvn::title_policy().guest_store_textures == nvn::TitlePolicy::GuestStoreTextures::ordered;
    if (!ordered_stores)
        return order_completion_read(nil, 0, reinterpret_cast<const std::byte*>(begin),
                                     size, "texture_report_wait", &ordered);
    // Guest-memory stores reach a texture with storage of its own through the
    // write log alone, published once the GPU ran them. Under the ordered
    // policy, wait for an earlier submission's retirement, or for this
    // submission's work so far, before a use whose texture data comes from
    // those pools.
    const auto prior_writer = [&](const auto& previous) {
        return previous->fence > retired_fence && previous->guest_writes &&
               guest_ranges_overlap(previous->guest_ranges.get(), begin, size);
    };
    const bool current_writer = guest_written(begin, size);
    const bool prior_writers = prior_guest_writer_fence > retired_fence &&
                               std::ranges::any_of(previous_submissions, prior_writer);
    const bool feeds = backend.resources().backing_feeds_texture(begin, size);
    // A predicted earlier submission whose learned pools exclude this backing
    // is not waited for. Record that skip even when this submission stores
    // nothing; the wait below is unchanged.
    if (feeds)
        for (const auto& previous : previous_submissions) {
            if (previous->fence <= retired_fence || !previous->guest_writes ||
                !previous->guest_writes_predicted || prior_writer(previous)) continue;
            backend.note_predicted_guest_read(*previous, begin, size);
        }
    if ((current_writer || prior_writers) && feeds) {
        for (const auto& previous : previous_submissions) {
            if (!prior_writer(previous)) continue;
            backend.wait_retired(*previous);
            retired_fence = backend.completed_fence();
        }
        if (current_writer && !synchronize_gpu()) return false;
        ++backend.mutable_statistics().guest_write_texture_waits;
        ++pending.counters.guest_waits;
        ordered = true;
    } else if (feeds && !current_writer && pending.guest_writes_predicted && guest_writes_pending) {
        // Write-anywhere would have drained a use of an unlearned pool.
        backend.note_predicted_guest_read(pending, begin, size);
    }
    return order_completion_read(nil, 0, reinterpret_cast<const std::byte*>(begin),
                                 size, "texture_report_wait", &ordered);
}

id<MTLTexture> Encoder::texture_for_use(const TextureState& state, const TextureViewState* view) {
    bool ordered = false;
    if (!order_texture_backing(reinterpret_cast<std::uint64_t>(state.memory), state.size, ordered))
        return nil;
    return backend.resources().texture(state, view);
}

id<MTLTexture> Encoder::sampled_texture(std::int32_t texture_id) {
    auto& resources = backend.resources();
    auto writes = memory::WriteTracker::now();
    auto& cached = resolved_textures[static_cast<std::uint32_t>(texture_id) &
                                     (resolved_textures.size() - 1)];
    if (cached.texture_id == texture_id && cached.writes == writes && cached.texture &&
        (cached.stamp.valid ? resources.texture_stamp_current(texture_id, cached.stamp)
                            : cached.registration == resources.registration_generation())) {
        bool ordered = false;
        if (!order_texture_backing(cached.stamp.backing_begin, cached.stamp.backing_size, ordered)) return nil;
        if (!ordered) return cached.texture;
        // Publication can change separately backed texels without changing the
        // native descriptor. Resolve again so its upload sees the new bytes.
        writes = memory::WriteTracker::now();
    }
    const auto registration = resources.registration_generation();
    ReplayResources::DescriptorStamp stamp;
    id<MTLTexture> texture = resources.texture_for_id_stamped(texture_id, stamp);
    bool ordered = false;
    if (texture && !order_texture_backing(stamp.backing_begin, stamp.backing_size, ordered)) return nil;
    if (ordered) {
        writes = memory::WriteTracker::now();
        texture = resources.texture_for_id_stamped(texture_id, stamp);
    }
    // Stamps precede resolution: a concurrent change must invalidate this
    // result on its next use, even if the lookup itself observed the change.
    if (texture) cached = {texture_id, registration, writes, stamp, texture};
    return texture;
}

BufferSpan Encoder::buffer_at(std::uint64_t address, std::uint64_t size) {
    const auto span = backend.resources().buffer_at(address, size);
    if (span.buffer && !order_completion_read(span, size)) return {};
    return span;
}

BufferAddress Encoder::buffer_address(std::uint64_t address, std::uint64_t size) {
    const auto range = backend.resources().buffer_address(address, size);
    if (range.buffer && !order_completion_read(range.buffer, range.offset, range.cpu, size)) return {};
    return range;
}

void Encoder::update_effect_ranges() {
    if (pending.effects.size() < effect_span.counted) effect_span.clear();
    for (; effect_span.counted < pending.effects.size(); ++effect_span.counted) {
        const auto& effect = pending.effects[effect_span.counted];
        std::uintptr_t low = 0, high = 0;
        if (effect.kind == CompletionEffect::Kind::report) {
            low = reinterpret_cast<std::uintptr_t>(effect.destination_cpu);
            high = low + sizeof(NVNcounterData);
        } else if (effect.kind == CompletionEffect::Kind::signal) {
            low = reinterpret_cast<std::uintptr_t>(effect.event);
            high = low + sizeof(std::uint32_t);
        } else {
            continue;
        }
        effect_span.low = std::min(effect_span.low, low);
        effect_span.high = std::max(effect_span.high, high);
        effect_span.writes.add(low, high - low);
    }
}

bool Encoder::order_completion_read(const BufferSpan& span, std::uint64_t size) {
    auto* contents = static_cast<std::byte*>(span.buffer.contents);
    return order_completion_read(span.buffer, span.offset, contents ? contents + span.offset : nullptr,
                                 size);
}

bool Encoder::order_completion_read(id<MTLBuffer> __unsafe_unretained buffer, std::uint64_t offset,
                                    const std::byte* cpu, std::uint64_t size,
                                    const char* wait_tag, bool* ordered) {
    const auto begin = reinterpret_cast<std::uintptr_t>(cpu);
    const auto drain = [&] {
        if (ordered) *ordered = true;
        const auto start = wait_tag ? timeline_now_ns() : 0;
        const bool result = synchronize_gpu();
        if (wait_tag) timeline_event(wait_tag, pending.fence, timeline_now_ns() - start);
        return result;
    };
    update_effect_ranges();
    const bool outside_pending = cpu && (begin + size <= effect_span.low || effect_span.high <= begin ||
        !effect_span.writes.may_intersect(begin, size));
    for (const auto& previous : previous_submissions) {
        if (previous->fence <= retired_fence) continue;
        if (cpu && (begin + size <= previous->completion_low || previous->completion_high <= begin ||
                    !previous->completion_ranges.may_intersect(begin, size))) continue;
        for (const auto& [write_begin, write_end] : previous->completion_writes)
            if (begin < write_end && write_begin < begin + size) {
                if (ordered) *ordered = true;
                const auto start = wait_tag ? timeline_now_ns() : 0;
                // A held queue's fence does not hold this read back.
                backend.wait_retired(*previous);
                if (wait_tag) timeline_event(wait_tag, pending.fence, timeline_now_ns() - start);
                retired_fence = backend.completed_fence();
                break;
            }
    }
    if (outside_pending) return true;
    for (const auto& effect : pending.effects) {
        if (effect.kind == CompletionEffect::Kind::report) {
            const auto address = reinterpret_cast<std::uintptr_t>(effect.destination_cpu);
            const bool overlap = cpu
                ? begin < address + sizeof(NVNcounterData) && address < begin + size
                : effect.destination.buffer == buffer &&
                    offset < effect.destination.offset + sizeof(NVNcounterData) &&
                    effect.destination.offset < offset + size;
            if (overlap) return drain();
        }
        if (effect.kind == CompletionEffect::Kind::signal && cpu) {
            const auto address = reinterpret_cast<std::uintptr_t>(effect.event);
            if (begin < address + sizeof(std::uint32_t) && address < begin + size)
                return drain();
        }
    }
    return true;
}

bool Encoder::enqueue_counter(const ReportCounter& command, const BufferSpan& destination,
                              std::byte* destination_cpu) {
    const SiteScope site_scope(*this, TransferSite::counter);
    if (!flush_uniforms()) return false;
    discard_uniform_captures(reinterpret_cast<std::uintptr_t>(destination_cpu), 0, sizeof(NVNcounterData));
    CompletionEffect effect{};
    effect.kind = CompletionEffect::Kind::report;
    effect.destination = destination;
    effect.destination_cpu = destination_cpu;
    effect.samples = command.type == NVN_COUNTER_TYPE_SAMPLES_PASSED;
    if (state.samples_active) {
        effect.visibility_begin = counter_begin;
        effect.visibility_end = visibility_cursor;
        counter_begin = visibility_cursor;
    }
    // Under the segment policy a timestamp report keeps the pass and takes
    // its segment's GPU start or end when the segment's feedback arrives.
    if (!effect.samples && nvn::title_policy().timestamps == nvn::TitlePolicy::Timestamps::segment) {
        effect.segment_time = true;
        effect.segment_start = command.type == NVN_COUNTER_TYPE_TIMESTAMP_TOP;
        effect.segment = pending.segments_committed;
        [pending.retained addObject:destination.buffer];
        note_report(false);
        pending.effects.push_back(effect);
        note_report_counter(command.type, command.address);
        trace_pass_event("report", "segment", command.address, pending.effects.size() - 1, 0, 0);
        invalidate_args_all();
        pending.buffer_writes.emplace_back(command.address, sizeof(NVNcounterData));
        other_buffer_writes.emplace_back(command.address, sizeof(NVNcounterData));
        note_writes();
        return true;
    }
    // A samples-passed report inside a pass created with the visibility
    // buffer keeps the pass: its slots are those of the draws before it, and
    // the render encoder writes its timestamp. Other reports end the pass,
    // so a GPU timestamp marks completed work precisely.
    const bool in_pass = render && pass_visibility_buffer && effect.samples;
    if (!in_pass) {
        close_render(PassBreak::counter);
        close_transfer();
        // A command-buffer timestamp after a relaxed pass end waits for that
        // pass through an empty encoder with full barriers.
        if (relaxed_producer && (!open_transfer() || !close_transfer())) return false;
    }
    if (pending.timestamp_count >= ReplayImpl::timestamp_heap_capacity * ReplayImpl::timestamp_heap_limit)
        return fail(ReplayErrorCode::allocation,
                    "submission timestamp limit of " +
                        std::to_string(ReplayImpl::timestamp_heap_capacity * ReplayImpl::timestamp_heap_limit) +
                        " reports is exhausted");
    effect.timestamp = backend.counter_heap(pending, pending.timestamp_count);
    if (!effect.timestamp)
        return fail(ReplayErrorCode::allocation, "GPU counter heap creation failed");
    effect.timestamp_index = pending.timestamp_count++ % ReplayImpl::timestamp_heap_capacity;
    [backend.device() sampleTimestamps:&effect.cpu_before gpuTimestamp:&effect.gpu_before];
    if (in_pass) write_render_timestamp(effect.timestamp, effect.timestamp_index);
    else write_timestamp(effect.timestamp, effect.timestamp_index);
    [pending.retained addObjectsFromArray:@[effect.timestamp, destination.buffer]];
    note_report(effect.samples);
    if (effect.samples && state.samples_active && counter_reset)
        gpu_counters.push_back({pending.effects.size(), std::uint32_t(reset_slot),
                                std::uint32_t(effect.visibility_end)});
    pending.effects.push_back(effect);
    trace_pass_event("report", effect.samples ? "samples" : "timestamp", command.address,
                     pending.effects.size() - 1, effect.visibility_begin, effect.visibility_end);
    // A report can change texels behind an otherwise unchanged binding.
    invalidate_args_all();
    pending.buffer_writes.emplace_back(command.address, sizeof(NVNcounterData));
    other_buffer_writes.emplace_back(command.address, sizeof(NVNcounterData));
    note_writes();
    note_report_counter(command.type, command.address);
    return true;
}

bool Encoder::resolve_carried_predicate() {
    if (!state.predicate_pending) return true;
    state.predicate_pending = false;
    backend.wait_prior_effects(pending);
    retired_fence = backend.completed_fence();
    return read_predicate(state.predicate_mode, state.predicate_address);
}

// ARMIGO_NVN_CONDITIONAL_LOG=N names the first N conditional-render decisions:
// the record, both counters (with whether each was known) and the outcome.
// A CPU-drained condition logs its outcome only.
void Encoder::log_conditional(std::uint64_t address, bool if_equal, std::uint64_t left, bool left_known,
                              std::uint64_t right, bool right_known, bool drawn) {
    static const unsigned long limit = [] {
        const char* value = std::getenv("ARMIGO_NVN_CONDITIONAL_LOG");
        return value ? std::strtoul(value, nullptr, 10) : 0ul;
    }();
    static std::atomic<unsigned long> logged{0};
    if (!limit || logged.fetch_add(1, std::memory_order_relaxed) >= limit) return;
    armigo::diag::write(armigo::diag::Level::warning,
        "nvn_conditional fence=%llu addr=%llx %s left=%llu%s right=%llu%s -> %s\n",
        static_cast<unsigned long long>(pending.fence), static_cast<unsigned long long>(address),
        if_equal ? "if_equal" : "if_not_equal", static_cast<unsigned long long>(left), left_known ? "" : "?",
        static_cast<unsigned long long>(right), right_known ? "" : "?", drawn ? "drawn" : "skipped");
}

// Sets render_enabled from the two counters at `address` in memory; their
// writers must have completed (a drain, or the previous submission retired).
bool Encoder::read_predicate(std::int32_t mode, std::uint64_t address) {
    const auto source = buffer_at(address, 2 * sizeof(NVNcounterData));
    if (!source.buffer.contents)
        return fail(ReplayErrorCode::missing_resource, "conditional render counters are unavailable");
    std::array<NVNcounterData, 2> counters{};
    std::memcpy(counters.data(), static_cast<const std::byte*>(source.buffer.contents) + source.offset,
                sizeof counters);
    const bool equal = counters[0].counter == counters[1].counter;
    state.render_enabled = mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL ? equal : !equal;
    return true;
}

// Opens a GPU-predicated block for the conditional when both counters are
// known to the GPU at this point of the stream: a samples-passed report of
// this submission after a reset (its visibility slots), or memory that no
// pending CPU effect of this submission writes and whose earlier CPU writers
// have completed. `handled` stays false otherwise, and the caller drains.
// Only completion effects count as writers: a GPU write to the counter's
// address (ReportValue, a buffer copy) after a samples report is not seen, as
// in the CPU read, where the report's effect lands after the GPU work.
// ARMIGO_NVN_GPU_PREDICATE=0 always drains.
bool Encoder::gpu_predicate(const RenderEnableConditional& command, bool& handled) {
    handled = false;
    static const bool enabled = [] {
        const char* value = std::getenv("ARMIGO_NVN_GPU_PREDICATE");
        return !(value && *value == '0');
    }();
    if (!enabled) return true;
    id<MTLComputePipelineState> pipeline = backend.predicate_pipeline();
    if (!pipeline) return true;
    PredicateJob job{};
    job.visibility = pending.visibility.gpuAddress;
    job.render_if_equal = command.mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL;
    for (std::size_t index = 0; index != 2; ++index) {
        const auto address = command.address + index * sizeof(NVNcounterData);
        // The raw lookup: Encoder::buffer_at would wait for this submission's
        // own pending writers of the counter.
        const auto span = backend.resources().buffer_at(address, sizeof(NVNcounterData));
        if (!span.buffer || !span.buffer.contents) return true;
        const auto* cpu = static_cast<const std::byte*>(span.buffer.contents) + span.offset;
        // The latest pending effect that writes the counter decides its value.
        std::optional<std::size_t> writer;
        update_effect_ranges();
        if (effect_span.writes.may_intersect(reinterpret_cast<std::uintptr_t>(cpu), sizeof(std::uint64_t))) {
            for (std::size_t effect = pending.effects.size(); effect-- != 0;) {
                const auto& candidate = pending.effects[effect];
                bool writes = false;
                if (candidate.kind == CompletionEffect::Kind::report)
                    writes = candidate.destination.buffer == span.buffer &&
                        span.offset < candidate.destination.offset + sizeof(NVNcounterData) &&
                        candidate.destination.offset < span.offset + sizeof(std::uint64_t);
                else if (candidate.kind == CompletionEffect::Kind::signal) {
                    const auto event = reinterpret_cast<std::uintptr_t>(candidate.event);
                    const auto begin = reinterpret_cast<std::uintptr_t>(cpu);
                    writes = begin < event + sizeof(std::uint32_t) && event < begin + sizeof(std::uint64_t);
                }
                if (writes) { writer = effect; break; }
            }
        }
        if (writer) {
            const auto found = std::find_if(gpu_counters.rbegin(), gpu_counters.rend(),
                [&](const GpuCounter& counter) { return counter.effect == *writer; });
            if (found == gpu_counters.rend() ||
                pending.effects[*writer].destination.offset != span.offset) return true;
            job.begin[index] = found->begin;
            job.end[index] = found->end;
            trace_pass_event("predicate_read", "visibility", index, found->begin, found->end, address);
        } else {
            if (!order_completion_read(span.buffer, span.offset, cpu, sizeof(std::uint64_t))) return false;
            make_resident(pending, span.buffer);
            job.memory[index] = span.buffer.gpuAddress + span.offset;
            trace_pass_event("predicate_read", "memory", index, 0, 0, address);
        }
    }
    // A report's slots can belong to an earlier closed pass even when later
    // draws are still open. Only the closed prefix is ready for an in-pass
    // vertex predicate; fragment-to-vertex dependencies keep the boundary.
    // Reuse is enabled by default for all titles. Set
    // ARMIGO_NVN_BATCH_FINALIZED_QUERIES=0 to restore the unbatched path.
    static const bool batch_finalized = [] {
        const char* value = std::getenv("ARMIGO_NVN_BATCH_FINALIZED_QUERIES");
        return !(value && *value == '0');
    }();
    id<MTLRenderPipelineState> reuse_pipeline = nil;
    if (batch_finalized && render) {
        const char* rejection = nullptr;
        // Preserve all guest-buffer write boundaries, including writes from
        // shaders that have since been rebound and buffer-backed attachments.
        if (pass_storage_writes || other_buffer_writes.size() != pass_buffer_writes_begin)
            rejection = "writes";
        if (color_aliases.any || std::any_of(block_aliases.begin(), block_aliases.end(),
                [](const BlockAlias& alias) { return alias.alias != nil; }))
            rejection = "alias";
        for (unsigned index = 0; index != attachment_root_count; ++index)
            if (attachment_roots[index].buffer) rejection = "attachment_buffer";
        for (std::size_t index = 0; !rejection && index != 2; ++index) {
            if (job.memory[index]) {
                if (!report_destination_isolated(command.address + index * sizeof(NVNcounterData),
                                                  sizeof(std::uint64_t)))
                    rejection = "memory";
            } else if (job.begin[index] != job.end[index] && job.end[index] > finalized_visibility) {
                rejection = "visibility";
            }
        }
        if (!rejection) {
            reuse_pipeline = predicate_draw_pipeline(render_descriptor);
            if (!reuse_pipeline) rejection = "pipeline";
        }
        if (rejection) trace_pass_event("predicate_batch", rejection);
    }
    if (!reuse_pipeline && !close_render(PassBreak::predicate)) return false;
    const auto slice = upload(64 + std::size_t(predicate_capacity) * predicate_record);
    if (!slice) return false;
    job.arguments = slice.gpu + 64;
    std::memcpy(slice.cpu, &job, sizeof job);
    // A retained pass runs the predicate below. Otherwise it waits for the
    // block's new pass, or for a kernel before intervening transfer work.
    predicate.job = slice.gpu;
    predicate.arguments = slice.cpu + 64;
    predicate.record_count = reinterpret_cast<std::uint32_t*>(slice.cpu + offsetof(PredicateJob, record_count));
    predicate.arguments_gpu = slice.gpu + 64;
    predicate.used = 0;
    predicate.mode = command.mode;
    predicate.address = command.address;
    if (reuse_pipeline) {
        // The helper must not reset the preceding guest draw's visibility
        // slot. The next guest draw selects its own slot and pipeline again.
        set_visibility(MTLVisibilityResultModeDisabled, 0);
        encode_predicate_draw(reuse_pipeline);
        encoded_pipeline = nullptr;
        trace_pass_event("predicate_batch", "reuse");
    }
    handled = true;
    return true;
}

// The predicate as a compute kernel, in its own compute encoder.
bool Encoder::dispatch_predicate(std::uint64_t job) {
    const SiteScope site_scope(*this, TransferSite::predicate);
    trace_pass_event("predicate_execute", "compute", job);
    id<MTLComputePipelineState> pipeline = backend.predicate_pipeline();
    if (!pipeline) return fail(ReplayErrorCode::unsupported, "the conditional-render kernel is unavailable");
    if (!open_transfer()) return false;
    if (queue) {
        queue->put(EncodeOp::predicate, 0,
                   PredicateRecord{(__bridge void*)pipeline, (__bridge void*)backend.predicate_table(), job});
        queue->publish();
    } else {
        id<MTL4ArgumentTable> table = backend.predicate_table();
        auto compute = (id<MTL4ComputeCommandEncoder>)transfer;
        [compute setComputePipelineState:pipeline];
        [table setAddress:job atIndex:0];
        [compute setArgumentTable:table];
        [compute dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
    transfer_untracked = true;
    ++backend.mutable_statistics().dispatches;
    ++pending.counters.dispatches;
    ++backend.mutable_statistics().predicate_kernels;
    return true;
}

// The pipeline that runs the pending predicate first in the pass about to
// open, as a point draw with rasterization disabled; nil when the kernel runs
// before the pass instead: with ARMIGO_NVN_PREDICATE_IN_PASS=0, or while no
// pipeline for the pass's attachments is ready (one compiles in the background).
id<MTLRenderPipelineState> Encoder::predicate_draw_pipeline(MTL4RenderPassDescriptor* pass) {
    static const bool in_pass = [] {
        const char* value = std::getenv("ARMIGO_NVN_PREDICATE_IN_PASS");
        return !(value && *value == '0');
    }();
    if (!in_pass) return nil;
    const auto depth = pass.depthAttachment.texture ? pass.depthAttachment.texture.pixelFormat
                                                    : MTLPixelFormatInvalid;
    const auto stencil = pass.stencilAttachment.texture ? pass.stencilAttachment.texture.pixelFormat
                                                        : MTLPixelFormatInvalid;
    return backend.predicate_render_pipeline(attachment_formats, depth, stencil, attachment_samples);
}

// The pending predicate at the start of the render pass that holds the
// block's first draw, before any of the pass's state: the point draw writes
// the argument records, and an encoder barrier holds the vertex stage of later
// commands in the pass (where a draw reads its indirect arguments) until it
// completes. The game's vertex argument table is bound again after it.
void Encoder::encode_predicate_draw(id<MTLRenderPipelineState> pipeline) {
    trace_pass_event("predicate_execute", "vertex", predicate.job);
    const auto job = std::exchange(predicate.job, 0);
    id<MTL4ArgumentTable> table = backend.predicate_table();
    if (queue) {
        queue->put(EncodeOp::predicate_draw, 0,
                   PredicateDrawRecord{(__bridge void*)pipeline, (__bridge void*)table, job,
                                       (__bridge void*)vertex_table});
        queue->publish();
    } else {
        [render setRenderPipelineState:pipeline];
        [table setAddress:job atIndex:0];
        [render setArgumentTable:table atStages:MTLRenderStageVertex];
        [render drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1];
        [render barrierAfterEncoderStages:MTLStageVertex beforeEncoderStages:MTLStageVertex
                        visibilityOptions:MTL4VisibilityOptionDevice];
        if (vertex_table) [render setArgumentTable:vertex_table atStages:MTLRenderStageVertex];
    }
    ++backend.mutable_statistics().predicate_draws;
}

// The GPU address of a new argument record in the open block, or 0 after a
// failure. draw() and draw_indexed() resolve a full block before encoding.
std::uint64_t Encoder::predicate_arguments(const void* record, std::size_t size) {
    if (predicate.used == predicate_capacity) {
        fail(ReplayErrorCode::encoding, "conditional block argument records are exhausted");
        return 0;
    }
    const auto offset = std::size_t(predicate.used++) * predicate_record;
    std::memcpy(predicate.arguments + offset, record, size);
    // Publish the extent after the payload. Both reach the GPU only when the
    // segment commits: automatic cuts reject open predicates, and a drain or
    // submission end seals the block before committing its argument storage.
    *predicate.record_count = predicate.used;
    return predicate.arguments_gpu + offset;
}

bool Encoder::block_alias(const nvn::TextureState& state, const nvn::TextureViewState* view,
                          BlockAlias& out) {
    out = {};
    const auto& description = state.description;
    if (!view || !view->sets_format || view->format != NVN_FORMAT_RGBA16UI ||
        (description.format != NVN_FORMAT_RGTC1_UNORM && description.format != NVN_FORMAT_RGTC1_SNORM))
        return true;
    const auto refuse = [&](const char* reason) {
        return fail(ReplayErrorCode::unsupported,
                    std::string("BC4 render target through an RGBA16UI view: ") + reason);
    };
    if (view->sets_target && view->target != description.target)
        return refuse("retargeted block views are unsupported");
    if (description.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET))
        return refuse("pitch-linear storage is unsupported");
    if (description.samples > 1) return refuse("multisampled storage is unsupported");
    const bool volume = description.target == NVN_TEXTURE_TARGET_3D;
    const NSUInteger levels = std::max<std::uint32_t>(description.levels, 1);
    const NSUInteger level = view->sets_levels ? view->base_level : 0;
    if (level >= levels) return refuse("the view level is outside the storage");
    NSUInteger layers = 1;
    switch (description.target) {
        case NVN_TEXTURE_TARGET_2D: case NVN_TEXTURE_TARGET_RECTANGLE: layers = 1; break;
        case NVN_TEXTURE_TARGET_2D_ARRAY: layers = std::max<std::uint32_t>(description.depth, 1); break;
        case NVN_TEXTURE_TARGET_CUBEMAP: layers = 6; break;
        case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: layers = 6 * std::max<std::uint32_t>(description.depth, 1); break;
        case NVN_TEXTURE_TARGET_3D: layers = std::max<NSUInteger>(std::max<std::uint32_t>(description.depth, 1) >> level, 1); break;
        default: return refuse("the texture target is unsupported");
    }
    const long first = view->sets_layers ? view->min_layer : 0;
    const long count = view->sets_layers ? (view->layer_count ? view->layer_count : long(layers) - first) : long(layers);
    if (first < 0 || NSUInteger(first) >= layers) return refuse("the view layer is outside the storage");
    if (count != 1) return refuse("layered rendering through a block view is unsupported");
    id<MTLTexture> storage = texture_for_use(state, nullptr);
    if (!storage) return fail(ReplayErrorCode::missing_resource, "render target texture is unavailable");
    if (storage.pixelFormat != MTLPixelFormatBC4_RUnorm && storage.pixelFormat != MTLPixelFormatBC4_RSnorm)
        return refuse("the storage does not hold BC4 blocks");
    if (storage.sampleCount != 1 || level >= storage.mipmapLevelCount ||
        (storage.textureType == MTLTextureType3D) != volume)
        return refuse("the Metal storage does not match the texture description");
    out.storage = storage;
    out.level = level;
    out.slice = volume ? 0 : NSUInteger(first);
    out.plane = volume ? NSUInteger(first) : 0;
    out.width = std::max<NSUInteger>(storage.width >> level, 1);
    out.height = std::max<NSUInteger>(storage.height >> level, 1);
    out.blocks_x = (out.width + 3) / 4;
    out.blocks_y = (out.height + 3) / 4;
    out.pitch = (out.blocks_x * 8 + 255) & ~NSUInteger(255);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Uint
        width:out.blocks_x height:out.blocks_y mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.allowGPUOptimizedContents = NO;
    out.alias = [backend.device() newTextureWithDescriptor:descriptor];
    out.staging = [backend.device() newBufferWithLength:out.pitch * out.blocks_y
                                                options:MTLResourceStorageModePrivate];
    if (!out.alias || !out.staging)
        return fail(ReplayErrorCode::allocation, "BC4 block view alias allocation failed");
    [pending.retained addObjectsFromArray:@[out.alias, out.staging, storage]];
    make_resident(pending, out.alias);
    make_resident(pending, out.staging);
    make_resident(pending, storage);
    return true;
}

// Raw block copies through the staging buffer, in their own copy encoder: the
// storage into each alias before its pass, every alias back afterwards. A
// three-dimensional storage is addressed as slice 0 with the plane in z.
bool Encoder::copy_block_aliases(bool to_alias) {
    if (std::none_of(block_aliases.begin(), block_aliases.end(),
                     [](const BlockAlias& block) { return block.alias != nil; }))
        return true;
    const auto blocks = block_aliases;
    return encode_call(^(EncodeCallContext& context) {
        id<MTL4ComputeCommandEncoder> copy = [context.commands computeCommandEncoder];
        if (!copy) {
            context.error = "BC4 block view copy encoder creation failed";
            return;
        }
        [copy barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
                    visibilityOptions:MTL4VisibilityOptionDevice];
        for (const auto& block : blocks) {
            if (!block.alias) continue;
            if (to_alias)
                [copy copyFromTexture:block.storage sourceSlice:block.slice sourceLevel:block.level
                         sourceOrigin:MTLOriginMake(0, 0, block.plane) sourceSize:MTLSizeMake(block.width, block.height, 1)
                             toBuffer:block.staging destinationOffset:0 destinationBytesPerRow:block.pitch
              destinationBytesPerImage:block.pitch * block.blocks_y];
            else
                [copy copyFromTexture:block.alias sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(block.blocks_x, block.blocks_y, 1) toBuffer:block.staging
                    destinationOffset:0 destinationBytesPerRow:block.pitch
              destinationBytesPerImage:block.pitch * block.blocks_y];
        }
        [copy barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit
                      visibilityOptions:MTL4VisibilityOptionDevice];
        for (const auto& block : blocks) {
            if (!block.alias) continue;
            if (to_alias)
                [copy copyFromBuffer:block.staging sourceOffset:0 sourceBytesPerRow:block.pitch
                    sourceBytesPerImage:block.pitch * block.blocks_y
                             sourceSize:MTLSizeMake(block.blocks_x, block.blocks_y, 1) toTexture:block.alias
                       destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
            else
                [copy copyFromBuffer:block.staging sourceOffset:0 sourceBytesPerRow:block.pitch
                    sourceBytesPerImage:block.pitch * block.blocks_y
                             sourceSize:MTLSizeMake(block.width, block.height, 1) toTexture:block.storage
                       destinationSlice:block.slice destinationLevel:block.level
                      destinationOrigin:MTLOriginMake(0, 0, block.plane)];
        }
        [copy barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageAll
                    visibilityOptions:MTL4VisibilityOptionDevice];
        [copy endEncoding];
    }, ReplayErrorCode::allocation);
}

// A sampled view with a non-identity swizzle reports only shader-read usage,
// even when its storage was created or upgraded with shader write. An image
// has no channel mapping, so the write binds that storage, or an unswizzled
// view of the same levels and slices when the image is not the whole texture.
static id<MTLTexture> writable_image_texture(id<MTLTexture> texture, id<MTLTexture> upgraded) {
    if (!texture) return nil;
    if (texture.usage & MTLTextureUsageShaderWrite) return texture;
    id<MTLTexture> base = texture;
    NSUInteger level = 0, slice = 0;
    while (id<MTLTexture> parent = base.parentTexture) {
        level += base.parentRelativeLevel;
        slice += base.parentRelativeSlice;
        base = parent;
    }
    if (!(base.usage & MTLTextureUsageShaderWrite)) {
        if (!upgraded || !(upgraded.usage & MTLTextureUsageShaderWrite)) return nil;
        base = upgraded;
    }
    const bool whole = level == 0 && slice == 0 &&
                       texture.mipmapLevelCount == base.mipmapLevelCount &&
                       texture.pixelFormat == base.pixelFormat &&
                       texture.textureType == base.textureType &&
                       texture.width == base.width && texture.height == base.height &&
                       texture.depth == base.depth && texture.arrayLength == base.arrayLength &&
                       texture.sampleCount == base.sampleCount;
    if (whole) return base;
    const auto type = texture.textureType;
    const bool cube = type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray;
    NSUInteger count = type == MTLTextureType3D ? 1
        : std::max<NSUInteger>(texture.arrayLength, 1) * (cube ? 6 : 1);
    if (type == MTLTextureType2D || type == MTLTextureType1D || type == MTLTextureTypeTextureBuffer)
        count = 1;
    return [base newTextureViewWithPixelFormat:texture.pixelFormat
                                   textureType:type
                                        levels:NSMakeRange(level, texture.mipmapLevelCount)
                                        slices:NSMakeRange(slice, count)];
}

// An R32 view of the texture's root covering the descriptor view's levels and
// slices. The root itself when that view is the whole two-dimensional texture
// and the root is already the requested format.
static id<MTLTexture> atomic_image_view(id<MTLTexture> texture, MTLPixelFormat format) {
    if (!texture) return nil;
    id<MTLTexture> cursor = texture;
    NSUInteger level = 0, slice = 0;
    while (id<MTLTexture> parent = cursor.parentTexture) {
        level += cursor.parentRelativeLevel;
        slice += cursor.parentRelativeSlice;
        cursor = parent;
    }
    const NSUInteger levels = std::max<NSUInteger>(texture.mipmapLevelCount, 1);
    const bool whole = level == 0 && slice == 0 && levels == cursor.mipmapLevelCount &&
                       cursor.pixelFormat == format && cursor.textureType == MTLTextureType2D &&
                       cursor.sampleCount == 1 && cursor.arrayLength <= 1;
    if (whole) return cursor;
    if (cursor.textureType != MTLTextureType2D && cursor.textureType != MTLTextureType2DArray)
        return nil;
    if (level >= cursor.mipmapLevelCount || levels > cursor.mipmapLevelCount - level)
        return nil;
    return [cursor newTextureViewWithPixelFormat:format
                                     textureType:MTLTextureType2D
                                          levels:NSMakeRange(level, levels)
                                          slices:NSMakeRange(slice, 1)];
}

// One stand-in per signedness. A texture Metal cannot run the atomic on binds
// this instead, so the dispatch still has a matching image to write.
static id<MTLTexture> atomic_sink(id<MTLDevice> device, bool is_signed) {
    struct Sink {
        __unsafe_unretained id<MTLDevice> device = nil;
        id<MTLTexture> textures[2] = {};
    };
    static Sink sink;
    if (sink.device != device) {
        sink.textures[0] = sink.textures[1] = nil;
        sink.device = device;
    }
    const int index = is_signed ? 1 : 0;
    if (sink.textures[index]) return sink.textures[index];
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                            is_signed ? MTLPixelFormatR32Sint : MTLPixelFormatR32Uint
                                                                          width:1
                                                                         height:1
                                                                      mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
                       MTLTextureUsageShaderAtomic;
    sink.textures[index] = [device newTextureWithDescriptor:descriptor];
    return sink.textures[index];
}

// Whether this storage image slot (stage and resource) is reported for the
// first time: one warning per slot for the life of the process.
static bool first_image_sink_report(std::uint64_t key) {
    static std::mutex mutex;
    static std::set<std::uint64_t> reported;
    const std::lock_guard lock(mutex);
    return reported.insert(key).second;
}

// The texture a storage image slot binds when its handle names no texture: a
// 1x1 write sink of the slot's declared type, in a format of its component
// class, so the stage still runs and its stores go nowhere. Nil for types a
// sink cannot have (buffer images).
static id<MTLTexture> image_sink(id<MTLDevice> device, MTLTextureType type,
                                 platform::CompiledShaderTextureScalar scalar) {
    struct Sinks {
        __unsafe_unretained id<MTLDevice> device = nil;
        std::map<std::pair<unsigned, unsigned>, id<MTLTexture>> textures;
    };
    static std::mutex mutex;
    static Sinks sinks;
    const std::lock_guard lock(mutex);
    if (sinks.device != device) {
        sinks.textures.clear();
        sinks.device = device;
    }
    const auto key = std::make_pair(unsigned(type), unsigned(scalar));
    if (const auto found = sinks.textures.find(key); found != sinks.textures.end()) return found->second;
    if (type != MTLTextureType1D && type != MTLTextureType1DArray && type != MTLTextureType2D &&
        type != MTLTextureType2DArray && type != MTLTextureType3D)
        return nil;
    const bool sint = scalar == platform::CompiledShaderTextureScalar::s32;
    const bool uint = scalar == platform::CompiledShaderTextureScalar::u32;
    auto* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = type;
    descriptor.pixelFormat = sint ? MTLPixelFormatR32Sint : uint ? MTLPixelFormatR32Uint : MTLPixelFormatR32Float;
    descriptor.width = descriptor.height = descriptor.depth = 1;
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
        (sint || uint ? MTLTextureUsageShaderAtomic : MTLTextureUsageUnknown);
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    sinks.textures[key] = texture;
    return texture;
}

bool Encoder::note_image_write(std::int32_t texture_id, __strong id<MTLTexture>& texture, bool compute) {
    if (!compute)
        return fail(ReplayErrorCode::unsupported, "storage image writes from draw stages are unsupported");
    // The lookup that resolved the texture left its stamp: its serial and the
    // guest storage it reads.
    const auto index = static_cast<std::uint32_t>(texture_id) & (resolved_textures.size() - 1);
    const auto& cached = resolved_textures[index];
    if (cached.texture_id != texture_id || !cached.stamp.serial)
        return fail(ReplayErrorCode::unsupported, "storage image texture has no known serial for its writes");
    const auto serial = cached.stamp.serial;
    const auto backing_begin = cached.stamp.backing_begin;
    const auto backing_size = cached.stamp.backing_size;
    if (!(texture.usage & MTLTextureUsageShaderWrite)) {
        // A texture registered as an image without the IMAGE flag: its
        // storage becomes shader-writable on its first write, with the old
        // contents copied over after all earlier work.
        id<MTLTexture> previous = nil;
        id<MTLTexture> upgraded = backend.resources().upgrade_for_image_write(serial, previous);
        if (!upgraded) {
            char detail[96];
            std::snprintf(detail, sizeof detail, "storage image texture %llu cannot be made shader-writable",
                          static_cast<unsigned long long>(serial));
            return fail(ReplayErrorCode::unsupported, detail);
        }
        if (previous) {
            if (!copy_replaced_image(previous, upgraded)) return false;
            // The encoder cached the pre-upgrade texture for this submission.
            // A descriptor stamp that does not move would hand that object
            // back; drop every entry of this storage and resolve it again.
            for (auto& entry : resolved_textures)
                if (entry.stamp.serial == serial) entry = {};
            texture = sampled_texture(texture_id);
        }
        if (texture && !(texture.usage & MTLTextureUsageShaderWrite))
            texture = writable_image_texture(texture, upgraded);
        if (!texture || !(texture.usage & MTLTextureUsageShaderWrite)) {
            char detail[96];
            std::snprintf(detail, sizeof detail, "storage image texture %llu is not shader-writable",
                          static_cast<unsigned long long>(serial));
            return fail(ReplayErrorCode::unsupported, detail);
        }
    }
    TextureState written{};
    written.serial = serial;
    written.memory = reinterpret_cast<std::uint8_t*>(backing_begin);
    written.size = backing_size;
    note_texture_write(written);
    // Texture accesses are not range-tracked: later dispatches wait.
    compute_untracked = true;
    return true;
}

bool Encoder::copy_replaced_image(id<MTLTexture> previous, id<MTLTexture> upgraded) {
    if (!close_transfer()) return false;
    [pending.retained addObjectsFromArray:@[previous, upgraded]];
    make_resident(pending, previous);
    make_resident(pending, upgraded);
    const bool same_buffer = previous.buffer && previous.buffer == upgraded.buffer;
    return encode_call(^(EncodeCallContext& context) {
        id<MTL4ComputeCommandEncoder> copy = [context.commands computeCommandEncoder];
        if (!copy) {
            context.error = "storage image upgrade copy encoder creation failed";
            return;
        }
        [copy barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
                    visibilityOptions:MTL4VisibilityOptionDevice];
        if (!same_buffer)
            [copy copyFromTexture:previous toTexture:upgraded];
        [copy barrierAfterStages:MTLStageBlit beforeQueueStages:MTLStageAll
                    visibilityOptions:MTL4VisibilityOptionDevice];
        [copy endEncoding];
    }, ReplayErrorCode::allocation);
}

struct AtomicImageKind {
    bool integer = false;
    bool signed_image = false;
};

static AtomicImageKind atomic_image_kind(const platform::CompiledShaderResource& resource) {
    AtomicImageKind kind;
    kind.signed_image = resource.texture_descriptor &&
        resource.texture_descriptor->scalar == platform::CompiledShaderTextureScalar::s32;
    kind.integer = kind.signed_image || (resource.texture_descriptor &&
        resource.texture_descriptor->scalar == platform::CompiledShaderTextureScalar::u32);
    return kind;
}

void Encoder::forget_replaced_image(std::uint64_t serial, id<MTLTexture> previous) {
    std::vector<std::uint64_t> retired;
    const auto remember = [&](id<MTLTexture> texture) {
        if (!texture) return;
        const auto identifier = texture.gpuResourceID._impl;
        if (std::find(retired.begin(), retired.end(), identifier) == retired.end())
            retired.push_back(identifier);
    };
    remember(previous);
    for (auto& entry : resolved_textures) {
        if (entry.stamp.serial != serial) continue;
        remember(entry.texture);
        entry = {};
    }
    for (auto& stage : bindings)
        for (auto& slot : stage.textures)
            if (slot.present &&
                std::find(retired.begin(), retired.end(), slot.value) != retired.end())
                slot = {};
    invalidate_args_all();
}

void Encoder::note_image_atomic_replacement(std::uint64_t serial, id<MTLTexture> previous, bool late) {
    if (late) {
        // Preparation is supposed to replace this storage before binding.
        // Counting survives release builds, where the assert is compiled out.
        ++backend.mutable_statistics().image_atomic_late_upgrades;
        assert(false && "image atomic storage replaced during bind");
    }
    const bool moved = previous.pixelFormat != MTLPixelFormatR32Uint &&
                       previous.pixelFormat != MTLPixelFormatR32Sint;
    if (moved) {
        ++backend.mutable_statistics().image_atomic_upgrades;
        armigo::diag::write(armigo::diag::Level::info,
            "nvn_replay: texture moved to R32 storage for image atomics (serial %llu, format %lu)\n",
            static_cast<unsigned long long>(serial),
            static_cast<unsigned long>(previous.pixelFormat));
    } else {
        armigo::diag::write(armigo::diag::Level::info,
            "nvn_replay: R32 texture remade with shader-atomic usage (serial %llu)\n",
            static_cast<unsigned long long>(serial));
    }
    forget_replaced_image(serial, previous);
}

// A sampled texture of another type than the shader declares, with no view
// in that type: the first ones are named with the stage and the texture, as
// Metal then reads it in the wrong type.
void Encoder::report_type_mismatch(const platform::CompiledGraphicsStage& shader,
                                   const platform::CompiledShaderResource& resource, std::int32_t texture_id,
                                   id<MTLTexture> texture, MTLTextureType want) {
    static std::atomic<unsigned> shown{0};
    if (shown.fetch_add(1, std::memory_order_relaxed) >= 16) return;
    armigo::diag::write(armigo::diag::Level::warning,
        "Armigo NVN: stage %s samples resource %u as Metal texture type %lu but texture ID %d is type %lu "
        "(format %lu, %lux%lux%lu, %lu layers, %lu levels); it is read in the wrong type\n",
        shader.identity.c_str(), resource.resource_id, static_cast<unsigned long>(want), texture_id,
        static_cast<unsigned long>(texture.textureType), static_cast<unsigned long>(texture.pixelFormat),
        static_cast<unsigned long>(texture.width), static_cast<unsigned long>(texture.height),
        static_cast<unsigned long>(texture.depth), static_cast<unsigned long>(texture.arrayLength),
        static_cast<unsigned long>(texture.mipmapLevelCount));
}

bool Encoder::note_image_atomic(std::size_t stage_index, const platform::CompiledShaderResource& resource,
                                std::uint32_t element, std::int32_t texture_id,
                                __strong id<MTLTexture>& texture, bool compute) {
    const auto index = static_cast<std::uint32_t>(texture_id) & (resolved_textures.size() - 1);
    const auto& cached = resolved_textures[index];
    if (cached.texture_id != texture_id || !cached.stamp.serial)
        return fail(ReplayErrorCode::unsupported, "storage image texture has no known serial for its writes");
    const auto serial = cached.stamp.serial;
    const auto backing_begin = cached.stamp.backing_begin;
    const auto backing_size = cached.stamp.backing_size;
    const auto kind = atomic_image_kind(resource);
    const auto want = kind.signed_image ? MTLPixelFormatR32Sint : MTLPixelFormatR32Uint;
    id<MTLTexture> previous = nil;
    id<MTLTexture> storage = nil;
    if (kind.integer)
        storage = backend.resources().upgrade_for_image_atomic(serial, kind.signed_image, previous);
    if (previous) {
        // Closing a pass that is already closed would drop the arguments this
        // bind is writing. The copy still has to follow the pass that used the
        // old storage.
        if (render && !close_render(PassBreak::dispatch)) return false;
        if (!copy_replaced_image(previous, storage)) return false;
        note_image_atomic_replacement(serial, previous, true);
        texture = sampled_texture(texture_id);
        if (!texture)
            return fail(ReplayErrorCode::missing_resource, "resolved storage image is unavailable");
    }
    id<MTLTexture> bound = kind.integer && storage ? atomic_image_view(texture, want) : nil;
    const bool ready = bound && bound.pixelFormat == want && bound.sampleCount == 1 &&
                       bound.textureType == MTLTextureType2D &&
                       backend.resources().image_atomic_capable(serial);
    // The first atomic binds name what each one runs on.
    static std::atomic<unsigned> shown{0};
    if (shown.fetch_add(1, std::memory_order_relaxed) < 16) {
        id<MTLTexture> root = texture;
        while (root.parentTexture) root = root.parentTexture;
        armigo::diag::write(armigo::diag::Level::info,
            "nvn_replay: image atomic bind serial %llu texture ID %d: format %lu %lux%lu of base format %lu "
            "(base usage 0x%lx), bound format %lu, %s%s\n",
            static_cast<unsigned long long>(serial), texture_id, static_cast<unsigned long>(texture.pixelFormat),
            static_cast<unsigned long>(texture.width), static_cast<unsigned long>(texture.height),
            static_cast<unsigned long>(root.pixelFormat), static_cast<unsigned long>(root.usage),
            static_cast<unsigned long>(bound.pixelFormat), ready ? "ready" : "sink",
            previous ? " after a late replacement" : "");
    }
    if (!ready) {
        bound = atomic_sink(backend.device(), kind.signed_image);
        if (!bound)
            return fail(ReplayErrorCode::allocation, "image atomic sink allocation failed");
        ++backend.mutable_statistics().image_atomic_sinks;
        const auto slot = resource.resource_id + element;
        const auto key = std::make_pair(static_cast<std::uint32_t>(stage_index), slot);
        if (std::find(image_atomic_warnings.begin(), image_atomic_warnings.end(), key) ==
            image_atomic_warnings.end()) {
            image_atomic_warnings.push_back(key);
            id<MTLTexture> root = texture;
            while (root.parentTexture) root = root.parentTexture;
            armigo::diag::write(armigo::diag::Level::warning,
                "nvn_replay: image atomics on a texture Metal cannot run them on "
                "(stage %u slot %u texture ID %d: Metal format %lu of base format %lu, type %lu)\n",
                static_cast<unsigned>(stage_index), slot, texture_id,
                static_cast<unsigned long>(texture.pixelFormat),
                static_cast<unsigned long>(root.pixelFormat),
                static_cast<unsigned long>(texture.textureType));
        }
    } else {
        // The stand-in is what the dispatch writes. Publishing the real
        // texture would make a later atomic treat an uploaded image as
        // GPU-written and move it.
        TextureState written{};
        written.serial = serial;
        written.memory = reinterpret_cast<std::uint8_t*>(backing_begin);
        written.size = backing_size;
        note_texture_write(written);
    }
    if (compute) compute_untracked = true;
    // The bound view's root (through its parents) and the colour view.
    make_resident(pending, texture);
    if (storage) make_resident(pending, storage);
    make_resident(pending, bound);
    texture = bound;
    return true;
}

bool Encoder::prepare_images(
    std::initializer_list<std::pair<std::size_t, const platform::CompiledGraphicsStage*>> stages) {
    for (const auto& [stage_index, shader] : stages) {
        if (!shader || stage_index >= stage_count) continue;
        std::vector<std::optional<std::uint64_t>> words(shader->handle_programs.size());
        for (const auto& resource : shader->resources) {
            if (resource.kind != platform::CompiledShaderResourceKind::StorageImage ||
                (!resource.atomic && resource.access == platform::CompiledShaderResourceAccess::Read)) continue;
            if (!resource.array_length || resource.location.count != resource.array_length)
                return fail(ReplayErrorCode::unsupported, "storage image array contract is invalid");
            const auto kind = atomic_image_kind(resource);
            const bool atomic = resource.atomic && kind.integer && image_atomics_enabled();
            for (std::uint32_t element = 0; element != resource.array_length; ++element) {
                const auto handle = read_stage_handle(stage_index, *shader, resource, element, words);
                if (failed) return false;
                auto texture_id = static_cast<std::int32_t>(handle_texture_id(handle));
                if (!texture_id) continue;
                ReplayResources::DescriptorStamp stamp;
                id<MTLTexture> base = backend.resources().image_base_for_id(texture_id, stamp);
                if (!base) {
                    if (backend.resources().texture_id_empty(texture_id)) continue;
                    return fail(ReplayErrorCode::missing_resource, "storage image descriptor backing is unavailable");
                }
                while (base.parentTexture) base = base.parentTexture;
                const bool upgrade = atomic ? backend.resources().image_atomic_needs_upgrade(stamp.serial, kind.signed_image)
                    : !(base.usage & MTLTextureUsageShaderWrite);
                if (!upgrade) continue;
                // A carried clear can still target the old allocation with
                // no encoder open. Land it before replacing that backing.
                if (!close_render(PassBreak::dispatch)) return false;
                id<MTLTexture> previous = nil;
                id<MTLTexture> storage = atomic
                    ? backend.resources().upgrade_for_image_atomic(stamp.serial, kind.signed_image, previous)
                    : backend.resources().upgrade_for_image_write(stamp.serial, previous);
                if (!storage) return fail(ReplayErrorCode::unsupported, "storage image backing cannot acquire required usage");
                if (!previous) continue;
                if (!copy_replaced_image(previous, storage)) return false;
                if (atomic) note_image_atomic_replacement(stamp.serial, previous, false);
                else for (auto& entry : resolved_textures)
                    if (entry.stamp.serial == stamp.serial) entry = {};
            }
        }
    }
    return true;
}

void Encoder::note_texture_write(const TextureState& texture) {
    pending.texture_writes.push_back(texture);
    static const std::uint64_t watched = [] {
        const char* value = std::getenv("ARMIGO_NVN_TEXTURE_PROVENANCE");
        return value ? std::strtoull(value, nullptr, 0) : 0ull;
    }();
    static std::atomic<std::uint64_t> producer_reports{0};
    if (watched && texture.serial == watched && producer_reports.fetch_add(1) < 64) {
        const auto label = [&](std::size_t slot) {
            const auto& program = state.stage_programs[slot];
            return stage_label(program ? program->stages[slot].get() : nullptr);
        };
        armigo::diag::info("nvn_texture_provenance_producer frame=%llu draw=%llu serial=%llu address=%#llx bytes=%llu "
            "bound_vs=%s bound_ts=%s bound_ms=%s bound_fs=%s bound_cs=%s\n",
            static_cast<unsigned long long>(backend.pool_present_count()),
            static_cast<unsigned long long>(pending.counters.draws), static_cast<unsigned long long>(texture.serial),
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(texture.memory)),
            static_cast<unsigned long long>(texture.size), label(NVN_SHADER_STAGE_VERTEX).c_str(),
            label(NVN_SHADER_STAGE_TESS_CONTROL).c_str(), label(NVN_SHADER_STAGE_TESS_EVALUATION).c_str(),
            label(NVN_SHADER_STAGE_FRAGMENT).c_str(), label(NVN_SHADER_STAGE_COMPUTE).c_str());
    }
    if (texture.memory && texture.size && backend.resources().texture_is_in_place(texture)) {
        const auto cpu = reinterpret_cast<std::uintptr_t>(texture.memory);
        other_buffer_writes.emplace_back(cpu, texture.size);
        discard_uniform_captures(cpu, 0, texture.size);
        // A report record can live in a texture's storage.
        if (reports) reports->gpu_write(cpu, texture.size);
    }
}

bool Encoder::open_transfer_with(std::span<const GuestAccess> accesses, bool declared) {
    // A pending conditional predicate reads its counters before this transfer
    // work can change them, as at the conditional itself.
    if (predicate.job && !dispatch_predicate(std::exchange(predicate.job, 0))) return false;
    // Bounds the pairwise check on long chains of independent operations.
    constexpr std::size_t tracked_limit = 256;
    if (transfer) {
        bool conflict = transfer_untracked || !declared ||
            transfer_accesses.size() + accesses.size() > tracked_limit;
        if (!conflict)
            for (const auto& next : accesses)
                for (const auto& done : transfer_accesses)
                    conflict |= (next.write || done.write) && next.begin < done.end && done.begin < next.end;
        if (conflict) {
            encoder_barrier(transfer, MTLStageBlit | MTLStageDispatch, MTLStageBlit | MTLStageDispatch);
            transfer_accesses.clear();
            transfer_untracked = false;
        } else {
            ++backend.mutable_statistics().transfer_barriers_skipped;
        }
    } else {
        note_transfer_open(transfer_site, render != nil);
        close_render(PassBreak::transfer);
        transfer = begin_transfer();
        ++backend.mutable_statistics().compute_passes;
        if (!transfer)
            return fail(ReplayErrorCode::allocation, "Metal replay transfer encoder creation failed");
        transfer_accesses.clear();
        transfer_untracked = false;
    }
    if (declared) transfer_accesses.insert(transfer_accesses.end(), accesses.begin(), accesses.end());
    else transfer_untracked = true;
    return true;
}

// A texture that shares guest bytes can observe a guest write that the
// compute accesses do not name. A private texture is written only by
// operations that are never tracked (copies, clears, storage image writes):
// they take the barrier and leave the encoder untracked.
void Encoder::note_compute_texture(__unsafe_unretained id<MTLTexture> texture) {
    __unsafe_unretained id<MTLTexture> root = texture;
    while (__unsafe_unretained id<MTLTexture> parent = root.parentTexture) root = parent;
    if (root.buffer) compute_untracked = true;
}


id<MTLTexture> Encoder::feedback_texture(id<MTLTexture> texture, bool image,
                                        std::optional<std::uint32_t> pinned_level) {
    if (!texture || !render_descriptor) return texture;
    const std::uint64_t original = texture.gpuResourceID._impl;
    for (const auto& use : feedback_uses)
        if (use.original == original && use.image == image) {
            feedback_rebind = true;
            return use.texture;
        }
    struct Subresource { id<MTLTexture> root; NSUInteger level = 0, slice = 0; };
    const auto root_of = [](id<MTLTexture> view) {
        Subresource sub{view};
        while (sub.root.parentTexture) {
            sub.level += sub.root.parentRelativeLevel;
            sub.slice += sub.root.parentRelativeSlice;
            sub.root = sub.root.parentTexture;
        }
        return sub;
    };
    const auto sampled = root_of(texture);
    const NSUInteger layers = texture.arrayLength *
        ((texture.textureType == MTLTextureTypeCube || texture.textureType == MTLTextureTypeCubeArray) ? 6 : 1);
    for (unsigned index = 0; index != 9; ++index) {
        const bool depth = index == 8;
        const MTLRenderPassAttachmentDescriptor* attachment = depth
            ? render_descriptor.depthAttachment : render_descriptor.colorAttachments[index];
        if (!attachment.texture) continue;
        const auto attached = root_of(attachment.texture);
        const NSUInteger level = attached.level + attachment.level;
        const NSUInteger slice = attached.slice + attachment.slice;
        if (sampled.root != attached.root || level < sampled.level ||
            level - sampled.level >= texture.mipmapLevelCount || slice < sampled.slice ||
            slice - sampled.slice >= layers) continue;
        // Image reinterpretations retain their original pixel-class access.
        if (!depth && image && texture.pixelFormat != attachment.texture.pixelFormat &&
            !raw_image_format_twins(texture.pixelFormat, attachment.texture.pixelFormat)) continue;
        if (depth) {
            // A read after a depth write must leave that pass before sampling,
            // including when the pass has no color attachments.
            if (!feedback_draw_depth_write) {
                if (render && feedback_depth_written) {
                    if (!close_render(PassBreak::depth_sampled)) return nil;
                    feedback_reopen = true;
                }
                feedback_depth_direct_reads = true;
                return texture;
            }
            if (image) {
                fail(ReplayErrorCode::unsupported_resource,
                     "depth attachment image feedback while writing requires an image-compatible snapshot");
                return nil;
            }
        } else {
            const auto bit = std::uint8_t(1u << index);
            if (!((feedback_draw_writes | feedback_written) & bit)) {
                feedback_direct_reads |= bit;
                continue;
            }
        }
        // A mip-chain draw must not expose its attached level to the texture
        // unit, even when its sampling weight for that level would be zero.
        NSUInteger first_level = sampled.level, level_count = texture.mipmapLevelCount;
        bool narrow = false;
        if (!depth && !image && pinned_level && texture.mipmapLevelCount > 1) {
            const auto pin = std::min<NSUInteger>(*pinned_level, texture.mipmapLevelCount - 1);
            if (sampled.level + pin != level) {
                first_level = sampled.level + pin;
                level_count = 1;
                narrow = true;
            }
        }
        if (!depth && !narrow && texture.mipmapLevelCount > 1 && level > sampled.level) {
            level_count = level - sampled.level;
            narrow = true;
        }
        if (narrow) {
            id<MTLTexture> use = [sampled.root newTextureViewWithPixelFormat:texture.pixelFormat
                textureType:texture.textureType levels:NSMakeRange(first_level, level_count)
                slices:NSMakeRange(sampled.slice, layers) swizzle:texture.swizzle];
            if (!use) {
                fail(ReplayErrorCode::allocation, "attachment feedback mip view creation failed");
                return nil;
            }
            feedback_uses.push_back({original, use, image});
            feedback_rebind = true;
            ++backend.mutable_statistics().feedback_mip_view_narrows;
            return use;
        }
        const auto type = texture.textureType;
        if (texture.sampleCount != 1 || (depth && (texture.mipmapLevelCount != 1 || layers != 1)) ||
            (type != MTLTextureType2D && type != MTLTextureType2DArray &&
             type != MTLTextureTypeCube && type != MTLTextureTypeCubeArray) ||
            sampled.root.textureType == MTLTextureType3D) {
            fail(ReplayErrorCode::unsupported_resource,
                 std::string("attachment feedback snapshot does not support ") + (depth ? "depth" : "color") +
                 " texture type " + std::to_string(type) + " with " + std::to_string(texture.sampleCount) +
                 " samples, " + std::to_string(texture.mipmapLevelCount) + " levels and " +
                 std::to_string(layers) + " layers");
            return nil;
        }
        // A shader can address every mip and layer exposed by its view. Keep
        // that shape and its relative coordinates, even when only one of its
        // subresources is attached. Copying only that attachment would change
        // the other reads, while leaving the original view bound aliases it.
        const NSUInteger width = std::max<NSUInteger>(1, sampled.root.width >> sampled.level);
        const NSUInteger height = std::max<NSUInteger>(1, sampled.root.height >> sampled.level);
        bool fresh = false;
        id<MTLTexture> snapshot = backend.feedback_snapshot(sampled.root.gpuResourceID._impl,
            sampled.level, sampled.slice, sampled.root.pixelFormat, width, height, type,
            texture.mipmapLevelCount, texture.arrayLength, depth, pending.sequence, feedback_epoch, fresh);
        if (!snapshot) {
            fail(ReplayErrorCode::allocation, "attachment feedback snapshot allocation failed");
            return nil;
        }
        if (fresh) {
            if (!close_render(PassBreak::attachment_feedback) || !open_transfer()) return nil;
            make_resident(pending, sampled.root);
            make_resident(pending, snapshot);
            for (NSUInteger mip = 0; mip != texture.mipmapLevelCount; ++mip) {
                const auto extent = MTLSizeMake(std::max<NSUInteger>(1, width >> mip),
                                               std::max<NSUInteger>(1, height >> mip), 1);
                for (NSUInteger layer = 0; layer != layers; ++layer)
                    copy_texture_range(sampled.root, static_cast<std::uint32_t>(sampled.slice + layer),
                        static_cast<std::uint32_t>(sampled.level + mip), MTLOriginMake(0, 0, 0), extent,
                        snapshot, static_cast<std::uint32_t>(layer), static_cast<std::uint32_t>(mip),
                        MTLOriginMake(0, 0, 0));
            }
            if (!close_transfer()) return nil;
            feedback_reopen = true;
            const auto count = ++backend.mutable_statistics().feedback_snapshots;
            if (count <= 4) {
                const auto shader = stage_label(bindings[NVN_SHADER_STAGE_FRAGMENT].shader);
                armigo::diag::write(armigo::diag::Level::info,
                    "Armigo NVN: attachment feedback snapshot %llu frame=%llu draw=%llu attachment=%u level=%lu layer=%lu shader=%s\n",
                    static_cast<unsigned long long>(count), static_cast<unsigned long long>(backend.pool_present_count()),
                    static_cast<unsigned long long>(pending.counters.draws), index,
                    static_cast<unsigned long>(level), static_cast<unsigned long>(slice), shader.c_str());
            }
        } else {
            ++backend.mutable_statistics().feedback_snapshot_reuses;
        }
        const auto swizzle = texture.swizzle;
        const bool identity = swizzle.red == MTLTextureSwizzleRed && swizzle.green == MTLTextureSwizzleGreen &&
            swizzle.blue == MTLTextureSwizzleBlue && swizzle.alpha == MTLTextureSwizzleAlpha;
        id<MTLTexture> use = snapshot;
        if (texture.pixelFormat != snapshot.pixelFormat || !identity)
            use = [snapshot newTextureViewWithPixelFormat:texture.pixelFormat textureType:texture.textureType
                levels:NSMakeRange(0, texture.mipmapLevelCount) slices:NSMakeRange(0, layers) swizzle:swizzle];
        if (!use) {
            fail(ReplayErrorCode::allocation, "attachment feedback snapshot view creation failed");
            return nil;
        }
        feedback_uses.push_back({original, use, image});
        feedback_rebind = true;
        return use;
    }
    return texture;
}

bool Encoder::open_render(std::int32_t primitive) {
    if (feedback_rebind) {
        dirty.arguments[NVN_SHADER_STAGE_FRAGMENT] = true;
        feedback_rebind = false;
    }
    close_transfer();
    const auto attachmentless_dimensions = [&] {
        const auto& viewport = state.viewports[0];
        const auto& scissor = state.scissors[0];
        const double x = state.viewport_count
            ? std::max(viewport.originX, viewport.originX + viewport.width)
            : state.scissor_count ? double(scissor.x) + scissor.width : 1.0;
        const double y = state.viewport_count
            ? std::max(viewport.originY, viewport.originY + viewport.height)
            : state.scissor_count ? double(scissor.y) + scissor.height : 1.0;
        return std::pair{NSUInteger(std::fmin(16384.0, std::fmax(1.0, std::ceil(x)))),
                         NSUInteger(std::fmin(16384.0, std::fmax(1.0, std::ceil(y))))};
    };
    if (render && attachmentless_pass) {
        const auto [width, height] = attachmentless_dimensions();
        if ((width != attachment_width || height != attachment_height) &&
            !close_render(PassBreak::render_targets)) return false;
    }
    const auto& fragment_program = state.stage_programs[NVN_SHADER_STAGE_FRAGMENT];
    const auto* fragment_shader = fragment_program
        ? fragment_program->stages[NVN_SHADER_STAGE_FRAGMENT].get() : nullptr;
    const bool color_draw = fragment_shader &&
        (state.channel_mask.mask & fragment_shader->interface.color_output_mask);
    if (render && color_draw && backend.pass_dump_enabled() &&
        backend.pass_dump_wants(backend.pool_present_count()) &&
        pending.counters.draws > pass_draws_at_open && render_descriptor) {
        const auto* attachment = render_descriptor.colorAttachments[0];
        id<MTLTexture> texture = attachment.texture;
        auto& dump = backend.pass_dump();
        const char* selection = std::getenv("ARMIGO_NVN_PASS_DUMP_INPUT_SHADER");
        const std::string_view filters = selection ? selection : "";
        const bool selected = dump.inputs() && !dump.metadata_only() && !filters.empty() &&
            (dump_input_shader_matches(fragment_shader, filters) ||
             dump_input_shader_matches(bindings[NVN_SHADER_STAGE_FRAGMENT].shader, filters));
        if (texture && (selected || dump.splits_draws(
                std::max<NSUInteger>(1, texture.width >> attachment.level),
                std::max<NSUInteger>(1, texture.height >> attachment.level)))) {
            // Keep selected draws isolated, or split color-writing draws of
            // the configured size. Depth-only prefixes keep their image budget.
            if (!close_render(PassBreak::draw_dump)) return false;
        }
    }
    // A relaxed pass ordered only fragment work after the previous pass: a
    // draw with earlier-stage writes or a mesh draw reopens it fully ordered.
    if (render && pass_relaxed && pre_fragment_hazard()) {
        barrier_requested = true;
        if (!close_render(PassBreak::relaxed_hazard)) return false;
    }
    const auto uses_depth_stencil = [&] {
        return nvn_driver::uses_depth_stencil({state.depth_stencil.depth_test, state.depth_stencil.depth_write,
            state.depth_stencil.stencil_test, fragment_shader && fragment_shader->writes_fragment_depth});
    };
    // The pass opened without depth and stencil meets a draw that uses them:
    // split here and remember the kind.
    if (render && depth_omitted && uses_depth_stencil()) {
        backend.note_depth_needed(depth_omitted_key);
        if (!close_render(PassBreak::depth_needed)) return false;
    }
    // A depth-only prefix has no shader storage writes to order. A later draw
    // that accesses storage buffers ends the continued pass conservatively.
    if (depth_barrier_allocation) {
        bool buffer_access = false;
        for (const auto stage : {NVN_SHADER_STAGE_VERTEX, NVN_SHADER_STAGE_FRAGMENT}) {
            const auto& program = state.stage_programs[stage];
            if (program && program->stages[stage])
                buffer_access |= !program->stages[stage]->storage_buffers.empty();
        }
        if (buffer_access && !close_render(PassBreak::storage_after_depth_barrier)) return false;
    }
    // Color slots this draw writes: the fragment stage's exports under the
    // channel mask.
    std::uint8_t color_writers = 0;
    if (fragment_shader && !state.rasterizer_discard) {
        const auto channels = state.channel_mask.mask & fragment_shader->interface.color_output_mask;
        for (unsigned i = 0; i < 8; ++i)
            if ((channels >> (i * 4)) & 15) color_writers |= 1u << i;
    }
    feedback_draw_writes = color_writers;
    const auto stencil_writes = [&](bool front) {
        return (front ? state.stencil_front_write : state.stencil_back_write) &&
            ((front ? state.depth_stencil.front_fail : state.depth_stencil.back_fail) != NVN_STENCIL_OP_KEEP ||
             (front ? state.depth_stencil.front_depth_fail : state.depth_stencil.back_depth_fail) != NVN_STENCIL_OP_KEEP ||
             (front ? state.depth_stencil.front_pass : state.depth_stencil.back_pass) != NVN_STENCIL_OP_KEEP);
    };
    feedback_draw_depth_write = state.targets.depth && !state.rasterizer_discard &&
        ((state.depth_stencil.depth_test && state.depth_stencil.depth_write) ||
         (state.depth_stencil.stencil_test && (stencil_writes(true) || stencil_writes(false))));
    if (render && ((color_writers & feedback_direct_reads) ||
                   (feedback_draw_depth_write && feedback_depth_direct_reads))) {
        ++backend.mutable_statistics().feedback_read_pass_splits;
        if (!close_render(PassBreak::attachment_feedback)) return false;
    }
    if (render && color_aliases.any && color_aliases.needs_reopen(color_writers, attached_colors) &&
        !close_render(PassBreak::color_alias))
        return false;
    // The open pass's descriptor is held by render_descriptor; a new one by
    // `created` until render_descriptor takes it.
    __unsafe_unretained MTL4RenderPassDescriptor* pass = render_descriptor;
    __attribute__((objc_precise_lifetime)) MTL4RenderPassDescriptor* created = nil;
    if (!render) {
        dirty.begin_pass();
        created = [MTL4RenderPassDescriptor new];
        pass = created;
        attachmentless_pass = false;
        attachment_width = attachment_height = attachment_samples = 0;
        attachment_formats = {};
        for (std::int32_t index = 0; index < std::clamp(state.targets.count, 0, 8); ++index) {
            const auto* texture_state = state.targets.color[index];
            if (!texture_state) continue;
            const auto* view = state.targets.has_color_view[index]
                ? &state.targets.color_views[index] : nullptr;
            auto& block = block_aliases[index];
            if (!block_alias(*texture_state, view, block)) return false;
            id<MTLTexture> texture = block.alias ? block.alias
                : texture_for_use(*texture_state, view);
            if (!texture)
                return fail(ReplayErrorCode::missing_resource,
                            "render target texture is unavailable");
            pass.colorAttachments[index].texture = texture;
            // A view's layer in a three-dimensional target is a depth plane.
            if (texture.textureType == MTLTextureType3D && view && view->sets_layers)
                pass.colorAttachments[index].depthPlane = NSUInteger(view->min_layer);
            pass.colorAttachments[index].loadAction = measurement_dontcare_loads()
                ? MTLLoadActionDontCare : MTLLoadActionLoad;
            // ARMIGO_NVN_ATT_NOSTORE=1: measurement only. Bounds what the
            // per-pass store of every attachment costs the GPU.
            static const bool store_attachments = [] {
                const char* value = std::getenv("ARMIGO_NVN_ATT_NOSTORE");
                return !(value && *value == '1');
            }();
            pass.colorAttachments[index].storeAction = store_attachments
                ? MTLStoreActionStore : MTLStoreActionDontCare;
            attachment_formats[index] = texture.pixelFormat;
            if (attachment_samples && attachment_samples != texture.sampleCount)
                return fail(ReplayErrorCode::unsupported,
                            "render targets use inconsistent sample counts");
            attachment_samples = texture.sampleCount;
            attachment_width = attachment_width ? std::min(attachment_width, texture.width) : texture.width;
            attachment_height = attachment_height ? std::min(attachment_height, texture.height) : texture.height;
            make_resident(pending, texture);
        }
        if (state.targets.depth) {
            const auto* view = state.targets.has_depth_view ? &state.targets.depth_view : nullptr;
            id<MTLTexture> texture = texture_for_use(*state.targets.depth, view);
            if (!texture)
                return fail(ReplayErrorCode::missing_resource,
                            "depth target texture is unavailable");
            static const bool omission = [] {
                const char* value = std::getenv("ARMIGO_NVN_OMIT_UNUSED_DEPTH");
                return !(value && *value == '0');
            }();
            const auto* vertex_stage = state.stage_programs[NVN_SHADER_STAGE_VERTEX]
                ? state.stage_programs[NVN_SHADER_STAGE_VERTEX]->stages[NVN_SHADER_STAGE_VERTEX].get() : nullptr;
            __unsafe_unretained id<MTLTexture> depth_root = texture;
            while (__unsafe_unretained id<MTLTexture> parent = depth_root.parentTexture) depth_root = parent;
            depth_omitted_key = {vertex_stage, fragment_shader, (__bridge const void*)depth_root};
            bool clears_depth = false;
            for (const auto& clear : deferred_clears)
                clears_depth |= clear.texture == texture;
            // The pass keeps an attachment only through a colour target the
            // draw writes.
            unsigned bound_writers = 0;
            for (unsigned i = 0; i < 8; ++i)
                if (pass.colorAttachments[i].texture && (color_writers & (1u << i))) bound_writers |= 1u << i;
            if (nvn_driver::may_omit_depth_stencil(omission, bound_writers, clears_depth,
                    {state.depth_stencil.depth_test, state.depth_stencil.depth_write,
                     state.depth_stencil.stencil_test,
                     fragment_shader && fragment_shader->writes_fragment_depth},
                    backend.depth_needed(depth_omitted_key))) {
                depth_omitted = true;
            } else {
                pass.depthAttachment.texture = texture;
                pass.depthAttachment.loadAction = measurement_dontcare_loads()
                    ? MTLLoadActionDontCare : MTLLoadActionLoad;
                pass.depthAttachment.storeAction = MTLStoreActionStore;
                if (texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8
#if !TARGET_OS_IPHONE
                    || texture.pixelFormat == MTLPixelFormatDepth24Unorm_Stencil8
#endif
                ) {
                    pass.stencilAttachment.texture = texture;
                    pass.stencilAttachment.loadAction = MTLLoadActionLoad;
                    pass.stencilAttachment.storeAction = MTLStoreActionStore;
                }
                attachment_width = attachment_width ? std::min(attachment_width, texture.width) : texture.width;
                attachment_height = attachment_height ? std::min(attachment_height, texture.height) : texture.height;
                if (attachment_samples && attachment_samples != texture.sampleCount)
                    return fail(ReplayErrorCode::unsupported,
                                "depth and color targets use inconsistent sample counts");
                attachment_samples = texture.sampleCount;
                make_resident(pending, texture);
            }
        }
        if (!attachment_width || !attachment_height) {
            bool bound_attachment = state.targets.depth != nullptr;
            for (std::int32_t index = 0; index < std::clamp(state.targets.count, 0, 8); ++index)
                bound_attachment |= state.targets.color[index] != nullptr;
            if (bound_attachment)
                return fail(ReplayErrorCode::unsupported,
                            "bound render attachments have no raster extent");
            const auto [width, height] = attachmentless_dimensions();
            attachment_width = pass.renderTargetWidth = width;
            attachment_height = pass.renderTargetHeight = height;
            attachment_samples = pass.defaultRasterSampleCount = 1;
            attachmentless_pass = true;
        }
        color_aliases = RenderColorAliases(pass);
        attached_colors = color_aliases.attachments(color_writers);
        for (unsigned i = 0; i < 8; ++i) {
            if (attached_colors & (1u << i)) continue;
            pass.colorAttachments[i].texture = nil;
            attachment_formats[i] = MTLPixelFormatInvalid;
        }
        render_descriptor = pass;
        attachment_root_count = 0;
        const auto add_root = [&](__unsafe_unretained id<MTLTexture> view) {
            if (!view) return;
            while (__unsafe_unretained id<MTLTexture> parent = view.parentTexture) view = parent;
            attachment_roots[attachment_root_count++] = view;
        };
        for (unsigned index = 0; index != 8; ++index) add_root(pass.colorAttachments[index].texture);
        add_root(pass.depthAttachment.texture);
        add_root(pass.stencilAttachment.texture);
    }
    const auto width = attachment_width, height = attachment_height;
    const auto& vertex_program = state.stage_programs[NVN_SHADER_STAGE_VERTEX];
    // A stage that failed to translate is not dropped: without its fragment
    // stage a draw would write depth only.
    if (state.failed_stages & graphics_stage_bits)
        return fail(ReplayErrorCode::missing_program, "program stage failed to translate");
    // A program without a fragment stage rasterizes for depth and stencil
    // only and writes no color.
    if (!vertex_program && !mesh_draw)
        return fail(ReplayErrorCode::missing_program, "draw has no bound vertex program");
    const auto* vertex_shader = mesh_draw ? nullptr : vertex_program->stages[NVN_SHADER_STAGE_VERTEX].get();
    if (mesh_draw && (!last_pipeline || dirty.pipeline || last_pipeline_primitive != primitive)) {
        MeshPipelineKey key;
        if (const auto why = fill_mesh_pipeline_key(state, attachment_formats, attachment_samples, key))
            return fail(ReplayErrorCode::unsupported, why);
        ++pending.counters.pipeline_hashes;
        const auto stage_of = [&](std::size_t slot) {
            const auto& program = state.stage_programs[slot];
            return program ? program->stages[slot] : nullptr;
        };
        auto* pipeline = backend.mesh_pipeline(key, stage_of(NVN_SHADER_STAGE_TESS_CONTROL),
            stage_of(NVN_SHADER_STAGE_TESS_EVALUATION), stage_of(NVN_SHADER_STAGE_FRAGMENT), pipeline_error_);
        if (!pipeline) return fail(ReplayErrorCode::unsupported, pipeline_error_);
        if (pipeline != last_pipeline) dirty.vertices = true;
        last_pipeline = pipeline;
        last_pipeline_primitive = primitive;
        dirty.pipeline = false;
    } else if (!last_pipeline || dirty.pipeline || last_pipeline_primitive != primitive) {
        DrawPipelineKey key;
        if (const auto why = fill_pipeline_key(state, primitive, attachment_formats, attachment_samples, key))
            return fail(ReplayErrorCode::unsupported, why);
        ++pending.counters.pipeline_hashes;
        auto* pipeline = backend.pipeline(key, vertex_program->stages[NVN_SHADER_STAGE_VERTEX],
            fragment_program ? fragment_program->stages[NVN_SHADER_STAGE_FRAGMENT] : nullptr,
            pipeline_error_);
        if (!pipeline) return fail(ReplayErrorCode::unsupported, pipeline_error_);
        // The first pipeline this submission has to compile: start the ones
        // its later draws will need now, so they compile in parallel.
        static const bool look_ahead = [] {
            const char* value = std::getenv("ARMIGO_NVN_PSO_LOOKAHEAD");
            return !(value && *value == '0');
        }();
        if (look_ahead && !pipelines_predicted && backend.created_pipeline()) {
            pipelines_predicted = true;
            predict_pipelines();
        }
        if (pipeline != last_pipeline) dirty.vertices = true;
        last_pipeline = pipeline;
        last_pipeline_primitive = primitive;
        dirty.pipeline = false;
    } else {
        ++pending.counters.pipeline_hits;
    }
    auto* pipeline = last_pipeline;

    if (!vertex_table || !fragment_table) {
        auto* table_descriptor = [MTL4ArgumentTableDescriptor new];
        table_descriptor.maxBufferBindCount = 31;
        table_descriptor.maxTextureBindCount = 128;
        table_descriptor.maxSamplerStateBindCount = 16;
        table_descriptor.initializeBindings = YES;
        table_descriptor.supportAttributeStrides = YES;
        NSError* table_error = nil;
        const bool new_vertex_table = !vertex_table;
        if (new_vertex_table)
            vertex_table = [backend.device() newArgumentTableWithDescriptor:table_descriptor
                                                                      error:&table_error];
        const bool new_fragment_table = !fragment_table;
        if (new_fragment_table)
            fragment_table = [backend.device() newArgumentTableWithDescriptor:table_descriptor
                                                                        error:&table_error];
        if (!vertex_table || !fragment_table)
            return fail(ReplayErrorCode::allocation,
                        "Metal replay graphics argument table creation failed");
        if (new_vertex_table)
            [pending.retained addObject:vertex_table];
        if (new_fragment_table)
            [pending.retained addObject:fragment_table];

    }
    if (mesh_draw && (!object_table || !mesh_table)) {
        auto* table_descriptor = [MTL4ArgumentTableDescriptor new];
        table_descriptor.maxBufferBindCount = 31;
        table_descriptor.maxTextureBindCount = 128;
        table_descriptor.maxSamplerStateBindCount = 16;
        table_descriptor.initializeBindings = YES;
        NSError* table_error = nil;
        if (!object_table) {
            object_table = [backend.device() newArgumentTableWithDescriptor:table_descriptor error:&table_error];
            if (object_table) [pending.retained addObject:object_table];
        }
        if (!mesh_table) {
            mesh_table = [backend.device() newArgumentTableWithDescriptor:table_descriptor error:&table_error];
            if (mesh_table) [pending.retained addObject:mesh_table];
        }
        if (!object_table || !mesh_table)
            return fail(ReplayErrorCode::allocation, "Metal replay mesh argument table creation failed");
    }
    // A handle program reads guest memory that can change between draws. Reusing
    // the argument table would keep the previous draw's handle. A title with
    // current handle reads keeps a stage's handles, like its other bindings,
    // until a binding, an inline update it reads, a registration or a CPU
    // write the replay tracks changes them (sync_draw_resources).
    const bool current_handles = nvn::title_policy().handle_reads == nvn::TitlePolicy::HandleReads::current;
    // ... an inline update, a buffer write or a storing guest-memory stage
    // since the handles were read also rereads them.
    const auto handles_stale = [&](std::size_t stage, const platform::CompiledGraphicsStage* shader) {
        if (!shader || shader->handle_programs.empty()) return false;
        if (!current_handles) return true;
        const auto& bound = bindings[stage];
        return bound.handle_updates != pending.counters.uniform_updates ||
               bound.handle_writes != other_buffer_writes.size() || bound.handle_stores != guest_write_serial;
    };
    if (handles_stale(NVN_SHADER_STAGE_VERTEX, vertex_shader)) dirty.arguments[NVN_SHADER_STAGE_VERTEX] = true;
    if (handles_stale(NVN_SHADER_STAGE_FRAGMENT, fragment_shader)) dirty.arguments[NVN_SHADER_STAGE_FRAGMENT] = true;
    if (sync_draw_resources()) {
        for (auto& stage : bindings) stage.windows.fill({});
    }
    if (mesh_draw) {
        const auto stage_shader = [&](std::size_t slot) -> const platform::CompiledGraphicsStage* {
            const auto& program = state.stage_programs[slot];
            return program ? program->stages[slot].get() : nullptr;
        };
        if (!prepare_images({
                {NVN_SHADER_STAGE_TESS_CONTROL, stage_shader(NVN_SHADER_STAGE_TESS_CONTROL)},
                {NVN_SHADER_STAGE_TESS_EVALUATION, stage_shader(NVN_SHADER_STAGE_TESS_EVALUATION)},
                {NVN_SHADER_STAGE_FRAGMENT, fragment_shader}}))
            return false;
        // The task and mesh stages bind to their own tables; there are no
        // vertex streams.
        for (const auto& [slot, table] : {std::pair{std::size_t(NVN_SHADER_STAGE_TESS_CONTROL), object_table},
                                          std::pair{std::size_t(NVN_SHADER_STAGE_TESS_EVALUATION), mesh_table}}) {
            const auto& program = state.stage_programs[slot];
            const auto* shader = program ? program->stages[slot].get() : nullptr;
            if (!shader) continue;
            if (stage_has_pool(shader) && !refresh_pools(slot, *shader)) return false;
            // Under current handle reads an unchanged stage keeps its table,
            // as the vertex and fragment stages do.
            if (current_handles && !dirty.arguments[slot] && !dirty.uniforms[slot] &&
                bindings[slot].shader == shader && !handles_stale(slot, shader)) {
                ++backend.mutable_statistics().stage_reuses;
                ++pending.counters.stage_reuses;
                continue;
            }
            if (!bind_stage(slot, *shader, table, true, false)) return false;
            dirty.arguments[slot] = false;
            dirty.uniforms[slot] = false;
            bindings[slot].shader = shader;
        }
        dirty.vertices = false;
    } else {
        if (vertex_shader && stage_has_pool(vertex_shader) &&
            !refresh_pools(NVN_SHADER_STAGE_VERTEX, *vertex_shader))
            return false;
        if (!prepare_images({
                {NVN_SHADER_STAGE_VERTEX, vertex_shader},
                {NVN_SHADER_STAGE_FRAGMENT, fragment_shader}}))
            return false;
        if (dirty.arguments[NVN_SHADER_STAGE_VERTEX] || dirty.uniforms[NVN_SHADER_STAGE_VERTEX] ||
            bindings[NVN_SHADER_STAGE_VERTEX].shader != vertex_shader) {
            const bool uniforms_only = !dirty.arguments[NVN_SHADER_STAGE_VERTEX] &&
                bindings[NVN_SHADER_STAGE_VERTEX].shader == vertex_shader;
            if (!bind_stage(NVN_SHADER_STAGE_VERTEX, *vertex_shader, vertex_table, true, uniforms_only))
                return false;
            dirty.arguments[NVN_SHADER_STAGE_VERTEX] = false;
            dirty.uniforms[NVN_SHADER_STAGE_VERTEX] = false;
            bindings[NVN_SHADER_STAGE_VERTEX].shader = vertex_shader;
        } else {
            ++backend.mutable_statistics().stage_reuses;
            ++pending.counters.stage_reuses;
        }
    }
    if (dirty.vertices && !mesh_draw) {
        for (std::size_t index = 0; index != state.vertex_buffers.size(); ++index) {
            const auto slot = pipeline->vertex_slots[index];
            if (slot < 0) continue;
            const auto& range = state.vertex_buffers[index];
            const auto span = buffer_address(range.address, range.size);
            if (!range.address || !range.size || !span.buffer)
                return fail(ReplayErrorCode::missing_resource,
                            "required vertex buffer range is unavailable");
            if (write_vertex_slot(vertex_table, NVN_SHADER_STAGE_VERTEX, slot, span.gpu,
                                  state.streams[index].stride))
                make_resident(pending, span.buffer);
        }
        if (pipeline->zero_attribute_slot >= 0) {
            id<MTLBuffer> zeros = backend.zero_buffer();
            if (!zeros)
                return fail(ReplayErrorCode::allocation, "zero vertex stream is unavailable");
            write_vertex_slot(vertex_table, NVN_SHADER_STAGE_VERTEX,
                              pipeline->zero_attribute_slot, zeros.gpuAddress, 16);
            make_resident(pending, zeros);
        }
        if (vertex_shader->vertex_fetch_buffer_index && !bind_vertex_fetch(*vertex_shader)) return false;
        dirty.vertices = false;
    }
    if (fragment_shader && stage_has_pool(fragment_shader) &&
        !refresh_pools(NVN_SHADER_STAGE_FRAGMENT, *fragment_shader))
        return false;
    if (fragment_shader && (dirty.arguments[NVN_SHADER_STAGE_FRAGMENT] ||
                         dirty.uniforms[NVN_SHADER_STAGE_FRAGMENT] ||
                         bindings[NVN_SHADER_STAGE_FRAGMENT].shader != fragment_shader)) {
        const bool uniforms_only = !dirty.arguments[NVN_SHADER_STAGE_FRAGMENT] &&
            bindings[NVN_SHADER_STAGE_FRAGMENT].shader == fragment_shader;
        if (!bind_stage(NVN_SHADER_STAGE_FRAGMENT, *fragment_shader, fragment_table, false,
                        uniforms_only))
            return false;
        dirty.arguments[NVN_SHADER_STAGE_FRAGMENT] = 0;
        dirty.uniforms[NVN_SHADER_STAGE_FRAGMENT] = false;
        bindings[NVN_SHADER_STAGE_FRAGMENT].shader = fragment_shader;
    } else if (fragment_shader) {
        ++backend.mutable_statistics().stage_reuses;
        ++pending.counters.stage_reuses;
    }

    if (feedback_reopen) {
        feedback_reopen = false;
        return open_render(primitive);
    }
    if (depth_barrier_reads) {
        if (!close_render(PassBreak::depth_sampled)) return false;
        return open_render(primitive);
    }

    if (state.samples_active) {
        if ((visibility_cursor + 1) * sizeof(std::uint64_t) > pending.visibility.length)
            return fail(ReplayErrorCode::allocation,
                        "samples-passed visibility storage is exhausted");
        pass.visibilityResultBuffer = pending.visibility;
        if (!render) pass_visibility_buffer = true;
        pass_visibility = visibility_cursor++;
    }
    if (!render && state.multisample.sample_locations_enable) {
        if (state.multisample.sample_locations_grid || attachment_samples > 16 ||
            !attachment_samples)
            return fail(ReplayErrorCode::unsupported,
                        "grid-varying programmable sample positions are unsupported");
        const auto required_mask = std::uint16_t((1u << attachment_samples) - 1u);
        if ((state.multisample.sample_location_mask & required_mask) != required_mask)
            return fail(ReplayErrorCode::unsupported,
                        "programmable sample positions are incomplete for the sample count");
        std::array<MTLSamplePosition, 16> positions{};
        for (NSUInteger index = 0; index != attachment_samples; ++index) {
            const auto packed = state.multisample.sample_locations[index];
            positions[index] = {float(packed & 0xfu) / 16.0f,
                                float(packed >> 4) / 16.0f};
        }
        [pass setSamplePositions:positions.data() count:attachment_samples];
    }
    if (!render) {
        // A pending conditional block's predicate runs first in the pass, or
        // as the kernel before it when no in-pass pipeline is ready.
        id<MTLRenderPipelineState> predicate_draw = nil;
        if (predicate.job) {
            predicate_draw = predicate_draw_pipeline(pass);
            if (!predicate_draw && (!dispatch_predicate(std::exchange(predicate.job, 0)) || !close_transfer()))
                return false;
        }
        if (!apply_clears(pass)) return false;
        if (pass.stencilAttachment.texture && pass.stencilAttachment.loadAction == MTLLoadActionClear)
            pass_depth_only = false;
        for (unsigned index = 0; index != 8; ++index)
            if (pass.colorAttachments[index].texture &&
                pass.colorAttachments[index].loadAction == MTLLoadActionClear)
                pass_depth_only = false;
        if (!copy_block_aliases(true)) return false;
        // A pass reopened after a close inside this draw reuses the
        // descriptor: whether it omits depth comes from the descriptor.
        depth_omitted = state.targets.depth && !pass.depthAttachment.texture;
        render = begin_render(pass, vertex_table, fragment_table);
        if (!render)
            return fail(ReplayErrorCode::allocation, "Metal replay render encoder creation failed");
        ++backend.mutable_statistics().render_passes;
        ++pending.counters.passes;
        pass_draws_at_open = pending.counters.draws;
        // A pending conditional block's predicate runs first in its pass.
        if (predicate_draw) encode_predicate_draw(predicate_draw);
        {
            // Guest bindings and final attachment descriptors for each draw
            // pass. Descriptor rows reflect clear folding and omitted targets.
            static const bool geometry_line = [] {
                const char* value = std::getenv("ARMIGO_NVN_PASS_GEOMETRY");
                return value && *value == '1';
            }();
            if (geometry_line)
                {
                    int in_place = 0, targets = 0;
                    for (std::int32_t index = 0; index < std::clamp(state.targets.count, 0, 8); ++index)
                        if (const auto* target = state.targets.color[index]) {
                            ++targets;
                            in_place += backend.resources().texture_is_in_place(*target) ? 1 : 0;
                        }
                    if (const auto* depth = state.targets.depth)
                        in_place += backend.resources().texture_is_in_place(*depth) ? 1 : 0;
                    // The pass's first fragment stage: its package identity, or
                    // the digest of a stage translated at run time.
                    const std::string fragment_name = !fragment_shader ? "-"
                        : !fragment_shader->identity.empty() ? fragment_shader->identity
                        : fragment_shader->msl_source_digest.empty() ? "?"
                        : fragment_shader->msl_source_digest.substr(0, 16);
                    armigo::diag::opt_in(
                                 "nvn_pass frame=%llu %lux%lu samples=%u targets=%d depth=%d in_place=%d "
                                 "pass=%llu fragment=%u channels=%u depth_test=%u depth_write=%u "
                                 "stencil_test=%u rasterizer_discard=%u samples_active=%u fs=%s\n",
                                 static_cast<unsigned long long>(pending.fence),
                                 static_cast<unsigned long>(attachment_width),
                                 static_cast<unsigned long>(attachment_height), unsigned(attachment_samples),
                                 targets, state.targets.depth != nullptr, in_place,
                                 static_cast<unsigned long long>(pending.counters.passes),
                                 unsigned(fragment_shader != nullptr),
                                 fragment_shader ? state.channel_mask.mask & fragment_shader->interface.color_output_mask : 0,
                                 unsigned(state.depth_stencil.depth_test), unsigned(state.depth_stencil.depth_write),
                                 unsigned(state.depth_stencil.stencil_test), unsigned(state.rasterizer_discard),
                                 unsigned(state.samples_active), fragment_name.c_str());
                    const auto describe_attachment = [&](const char* aspect, unsigned index,
                                                         MTLRenderPassAttachmentDescriptor* attachment) {
                        id<MTLTexture> texture = attachment.texture;
                        if (!texture) return;
                        armigo::diag::opt_in(
                            "nvn_attachment frame=%llu pass=%llu aspect=%s index=%u width=%lu height=%lu "
                            "samples=%lu format=%lu level=%lu slice=%lu depth_plane=%lu load=%lu store=%lu "
                            "storage=%lu usage=%lu\n",
                            static_cast<unsigned long long>(pending.fence),
                            static_cast<unsigned long long>(pending.counters.passes), aspect, index,
                            static_cast<unsigned long>(std::max<NSUInteger>(1, texture.width >> attachment.level)),
                            static_cast<unsigned long>(std::max<NSUInteger>(1, texture.height >> attachment.level)),
                            static_cast<unsigned long>(texture.sampleCount),
                            static_cast<unsigned long>(texture.pixelFormat),
                            static_cast<unsigned long>(attachment.level), static_cast<unsigned long>(attachment.slice),
                            static_cast<unsigned long>(attachment.depthPlane),
                            static_cast<unsigned long>(attachment.loadAction),
                            static_cast<unsigned long>(attachment.storeAction),
                            static_cast<unsigned long>(texture.storageMode), static_cast<unsigned long>(texture.usage));
                    };
                    for (unsigned index = 0; index != 8; ++index)
                        describe_attachment("color", index, pass.colorAttachments[index]);
                    describe_attachment("depth", 0, pass.depthAttachment);
                    describe_attachment("stencil", 0, pass.stencilAttachment);
                }
        }
        for (unsigned i = 0; i < 8; ++i) pass.colorAttachments[i].loadAction = MTLLoadActionLoad;
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
    }
    if (encoded_pipeline != pipeline) {
        if (!set_render_pipeline(*pipeline)) return false;
        encoded_pipeline = pipeline;
    }
    render_primitive = primitive;

    if (dirty.fixed & 1u) {
        DepthStateKey depth_values{};
        static_assert(sizeof state.depth_stencil <= 2 * sizeof(std::uint32_t));
        std::memcpy(depth_values.data(), &state.depth_stencil, sizeof state.depth_stencil);
        depth_values[2] = state.stencil_front_read;
        depth_values[3] = state.stencil_back_read;
        depth_values[4] = state.stencil_front_write;
        depth_values[5] = state.stencil_back_write;
        if (fixed_changed(0, fixed.depth, depth_values)) {
            id<MTLDepthStencilState> depth_state = backend.depth_state(depth_values);
            if (!depth_state) {
                auto *depth_descriptor = [MTLDepthStencilDescriptor new];
                const auto depth_compare = compare_function(state.depth_stencil.depth_function);
                if (!depth_compare)
                    return fail(ReplayErrorCode::unsupported,
                                "draw uses an unsupported depth comparison function");
                depth_descriptor.depthCompareFunction =
                    state.depth_stencil.depth_test ? *depth_compare : MTLCompareFunctionAlways;
                depth_descriptor.depthWriteEnabled =
                    state.depth_stencil.depth_test && state.depth_stencil.depth_write;
                const auto face = [&](bool front) -> MTLStencilDescriptor * {
                    auto *result = [MTLStencilDescriptor new];
                    const auto comparison =
                        compare_function(front ? state.depth_stencil.front_function
                                               : state.depth_stencil.back_function);
                    const auto fail_op = stencil_operation(front ? state.depth_stencil.front_fail
                                                                 : state.depth_stencil.back_fail);
                    const auto depth_fail =
                        stencil_operation(front ? state.depth_stencil.front_depth_fail
                                                : state.depth_stencil.back_depth_fail);
                    const auto pass_op = stencil_operation(front ? state.depth_stencil.front_pass
                                                                 : state.depth_stencil.back_pass);
                    if (!comparison || !fail_op || !depth_fail || !pass_op)
                        return nil;
                    result.stencilCompareFunction =
                        state.depth_stencil.stencil_test ? *comparison : MTLCompareFunctionAlways;
                    result.stencilFailureOperation =
                        state.depth_stencil.stencil_test ? *fail_op : MTLStencilOperationKeep;
                    result.depthFailureOperation =
                        state.depth_stencil.stencil_test ? *depth_fail : MTLStencilOperationKeep;
                    result.depthStencilPassOperation =
                        state.depth_stencil.stencil_test ? *pass_op : MTLStencilOperationKeep;
                    result.readMask = front ? state.stencil_front_read : state.stencil_back_read;
                    result.writeMask =
                        state.depth_stencil.stencil_test
                            ? (front ? state.stencil_front_write : state.stencil_back_write)
                            : 0;
                    return result;
                };
                depth_descriptor.frontFaceStencil = face(true);
                depth_descriptor.backFaceStencil = face(false);
                if (!depth_descriptor.frontFaceStencil || !depth_descriptor.backFaceStencil)
                    return fail(ReplayErrorCode::unsupported,
                                "draw uses an unsupported stencil function or operation");
                depth_state =
                    [backend.device() newDepthStencilStateWithDescriptor:depth_descriptor];
                if (!depth_state)
                    return fail(ReplayErrorCode::allocation,
                                "Metal replay depth-stencil state creation failed");
                backend.remember_depth_state(depth_values, depth_state);
            }
            retain_once(pending, depth_state);
            set_depth_stencil(depth_state);
        }
    }
    if ((dirty.fixed & (1u << 1)) &&
        fixed_changed(1, fixed.stencil,
                      std::array{state.stencil_front_ref, state.stencil_back_ref}))
        set_stencil_reference(state.stencil_front_ref, state.stencil_back_ref);

    if (dirty.fixed & ((1u << 2) | (1u << 3) | (1u << 4))) {
        MTLCullMode cull = MTLCullModeNone;
        if (state.polygon.cull == NVN_FACE_FRONT)
            cull = MTLCullModeFront;
        if (state.polygon.cull == NVN_FACE_BACK)
            cull = MTLCullModeBack;
        if (state.polygon.cull == NVN_FACE_FRONT_AND_BACK)
            return fail(ReplayErrorCode::unsupported,
                        "front-and-back culling cannot be encoded by Metal");
        if (fixed_changed(2, fixed.cull, cull))
            set_cull(cull);
        if (fixed_changed(3, fixed.winding, std::uint32_t(state.polygon.front_face)))
            set_winding(state.polygon.front_face == NVN_FRONT_FACE_CW ? MTLWindingClockwise
                                                                      : MTLWindingCounterClockwise);
        if (state.polygon.mode == NVN_POLYGON_MODE_POINT)
            return fail(ReplayErrorCode::unsupported, "polygon point fill mode is unsupported");
        if (fixed_changed(4, fixed.fill, std::uint32_t(state.polygon.mode)))
            set_fill(state.polygon.mode == NVN_POLYGON_MODE_LINE ? MTLTriangleFillModeLines
                                                                 : MTLTriangleFillModeFill);
    }
    if ((dirty.fixed & (1u << 5)) && fixed_changed(5, fixed.depth_clamp, state.depth_clamp))
        set_depth_clip(state.depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip);
    if (dirty.fixed & (1u << 6)) {
        const auto offset_bit =
            state.polygon.mode == NVN_POLYGON_MODE_LINE    ? NVN_POLYGON_OFFSET_ENABLE_LINE
            : state.polygon.mode == NVN_POLYGON_MODE_POINT ? NVN_POLYGON_OFFSET_ENABLE_POINT
                                                           : NVN_POLYGON_OFFSET_ENABLE_FILL;
        const bool offset_enabled = state.polygon.offset_enables & offset_bit;
        const std::array bias{offset_enabled ? state.polygon_units : 0.0f,
                              offset_enabled ? state.polygon_factor : 0.0f,
                              offset_enabled ? state.polygon_clamp : 0.0f};
        if (fixed_changed(6, fixed.bias, bias))
            set_depth_bias(bias[0], bias[1], bias[2]);
    }
    if ((dirty.fixed & (1u << 7)) && fixed_changed(7, fixed.blend, state.blend_color))
        set_blend_color(state.blend_color);
    if (dirty.fixed & (1u << 8)) {
        auto viewports = state.viewports;
        const auto viewport_count = std::max<std::size_t>(state.viewport_count, 1);
        if (!state.viewport_count)
            viewports[0] = {0, 0, double(width), double(height), state.depth_near, state.depth_far};
        if (fixed_changed(8, fixed.viewport, std::pair{viewports, viewport_count}))
            set_viewports(viewports.data(), viewport_count);
    }
    if (dirty.fixed & (1u << 9)) {
        std::array<MTLScissorRect, 16> clipped{};
        const auto scissor_count = std::max<std::size_t>(state.scissor_count, 1);
        for (std::size_t index = 0; index != state.scissor_count; ++index)
            clipped[index] = clip_scissor(state.scissors[index], width, height);
        if (!state.scissor_count)
            clipped[0] = {0, 0, width, height};
        if (fixed_changed(9, fixed.scissor, std::pair{clipped, scissor_count}))
            set_scissors(clipped.data(), scissor_count);
    }
    dirty.fixed = 0;
    if (state.samples_active)
        set_visibility(MTLVisibilityResultModeCounting, pass_visibility * sizeof(std::uint64_t));
    if (backend.pass_dump_enabled() && backend.pass_dump_wants(backend.pool_present_count())) {
        const char* selection = std::getenv("ARMIGO_NVN_PASS_DUMP_INPUT_SHADER");
        const std::string_view filters = selection ? selection : "";
        const bool selected_inputs = dump_input_shader_matches(fragment_shader, filters);
        const auto take_reads = [&](std::size_t slot) {
            const auto& reads = bindings[slot].dump_reads;
            render_dump_reads.insert(reads.begin(), reads.end());
            auto& dump = backend.pass_dump();
            if (dump.metadata_only() || (dump.inputs() && selected_inputs)) {
                const auto metadata = dump_binding_metadata(slot);
                render_dump_reads.insert(metadata.begin(), metadata.end());
            }
            id<MTLTexture> color = render_descriptor.colorAttachments[0].texture;
            if (dump.inputs() && selected_inputs && color_draw && color &&
                ((!dump.metadata_only() && !filters.empty()) || dump.splits_draws(
                    std::max<NSUInteger>(1, color.width >> render_descriptor.colorAttachments[0].level),
                    std::max<NSUInteger>(1, color.height >> render_descriptor.colorAttachments[0].level)))) {
                const auto& textures = bindings[slot].dump_textures;
                render_dump_textures.insert(render_dump_textures.end(), textures.begin(), textures.end());
                const auto* shader = bindings[slot].shader;
                const auto snapshot = [&](std::uint32_t index, std::uint64_t size, std::string name) {
                    if (index >= bindings[slot].buffers.size()) return;
                    const auto& buffer = bindings[slot].buffers[index];
                    if (!buffer.present) return;
                    const auto prefix = std::move(name) + "-d" + std::to_string(pending.counters.draws);
                    constexpr std::uint64_t chunk_bytes = 1024 * 1024;
                    if (size <= chunk_bytes) {
                        render_dump_buffers.push_back(dump.prepare_buffer(prefix, buffer.value, size));
                        return;
                    }
                    std::uint64_t available = 0, allocation_gpu = 0;
                    for (id object : pending.carry->retained) {
                        if (![object conformsToProtocol:@protocol(MTLBuffer)]) continue;
                        id<MTLBuffer> allocation = object;
                        const auto base = allocation.gpuAddress;
                        if (buffer.value < base || buffer.value - base >= allocation.length) continue;
                        allocation_gpu = base;
                        available = allocation.length - (buffer.value - base);
                        break;
                    }
                    const auto captured = std::min({size, available, 16 * chunk_bytes});
                    render_dump_reads.insert("input_buffer stage=" + std::to_string(slot) +
                        " index=" + std::to_string(index) + " gpu=" + std::to_string(buffer.value) +
                        " allocation_gpu=" + std::to_string(allocation_gpu) + " requested=" + std::to_string(size) +
                        " available=" + std::to_string(available) + " captured=" + std::to_string(captured));
                    if (!captured) {
                        render_dump_buffers.push_back(dump.prepare_buffer(prefix, buffer.value, 0));
                        return;
                    }
                    for (std::uint64_t offset = 0; offset < captured && !dump.stopped(); offset += chunk_bytes)
                        render_dump_buffers.push_back(dump.prepare_buffer(prefix + "-o" + std::to_string(offset),
                            buffer.value + offset, std::min(chunk_bytes, captured - offset)));
                };
                if (shader) {
                    for (const auto& buffer : shader->constant_buffers) {
                        const auto guest = hardware_buffer(slot, buffer.guest_binding);
                        const bool driver = buffer.guest_binding == 0 || buffer.guest_binding == 2 ||
                            (state.push_bank_at(slot) && buffer.guest_binding == state.push_bank_at(slot));
                        const auto size = driver ? std::uint64_t(compute_driver_data_bytes)
                            : std::min<std::uint64_t>(guest.size,
                                buffer.minimum_size ? buffer.minimum_size : guest.size);
                        snapshot(buffer.metal_buffer_index, size, "cb" + std::to_string(slot) +
                            "-hw" + std::to_string(buffer.guest_binding));
                    }
                    for (const auto& buffer : shader->constant_buffer_windows)
                        snapshot(buffer.metal_buffer_index, buffer.window_bytes, "cw" + std::to_string(slot) +
                            "-hw" + std::to_string(buffer.guest_binding));
                    for (const auto& buffer : shader->storage_buffers)
                        if (buffer.metal_buffer_index < bindings[slot].dump_buffer_sizes.size())
                            snapshot(buffer.metal_buffer_index, bindings[slot].dump_buffer_sizes[buffer.metal_buffer_index],
                                "sb" + std::to_string(slot) + "-" + std::to_string(buffer.metal_buffer_index));
                    if (shader->guest_memory && pending.guest_table)
                        snapshot(shader->guest_memory->buffer_index, pending.guest_table.length,
                                 "gm" + std::to_string(slot));
                }
            }
        };
        if (mesh_draw) {
            for (const auto slot : {NVN_SHADER_STAGE_TESS_CONTROL, NVN_SHADER_STAGE_TESS_EVALUATION}) {
                const auto& program = state.stage_programs[slot];
                if (program && program->stages[slot]) take_reads(slot);
            }
        } else {
            take_reads(NVN_SHADER_STAGE_VERTEX);
        }
        if (fragment_shader) take_reads(NVN_SHADER_STAGE_FRAGMENT);
        const auto label = [&](std::size_t slot) {
            return stage_label(bindings[slot].shader);
        };
        render_dump_draws.push_back("ordinal=" + std::to_string(pending.counters.draws - pass_draws_at_open) +
            " submission_draw=" + std::to_string(pending.counters.draws) +
            " vs=" + (mesh_draw ? std::string("-") : label(NVN_SHADER_STAGE_VERTEX)) +
            " ts=" + (mesh_draw ? label(NVN_SHADER_STAGE_TESS_CONTROL) : std::string("-")) +
            " ms=" + (mesh_draw ? label(NVN_SHADER_STAGE_TESS_EVALUATION) : std::string("-")) +
            " fs=" + (fragment_shader ? label(NVN_SHADER_STAGE_FRAGMENT) : std::string("-")));
    }
    feedback_written |= color_writers;
    feedback_depth_written |= feedback_draw_depth_write;
    feedback_uses.clear();
    return true;
}

BoundBuffer Encoder::hardware_buffer(std::size_t stage, std::uint32_t slot) const {
    if (slot >= 3 && slot == state.push_bank_at(stage)) return {};
    if (slot >= 3) return slot - 3 < buffer_slots ? state.uniform_at(stage)[slot - 3] : BoundBuffer{};
    if (slot == 1 && state.stage_programs[stage]) {
        const auto& constants = state.stage_programs[stage]->constants[stage];
        return {constants.address, constants.size};
    }
    return {};
}

// Driver data (slot 0): words 0..7 hold 0..7 and every other word all ones,
// with each stage's storage-buffer table (address, size per 16-byte row) at
// its base (vertex 0x110, compute 0x310, fragment 0x510, task 0x670, mesh
// 0x770), the compute stage's uniform-buffer table at 0x210 and a graphics
// stage's draw base instance at 0x644. The
// bound-texture handle table (slot 2) holds the 32-bit handle of texture
// slot n at 0x20 + 8 n.
void Encoder::note_base_instance(std::uint32_t base_instance) {
    if (base_instance == draw_base_instance) return;
    draw_base_instance = base_instance;
    // Driver data bound whole may be read anywhere; a window, up to its size.
    const auto reads_word = [](const platform::CompiledGraphicsStage& shader) {
        return std::ranges::any_of(shader.constant_buffers, [](const auto& binding) {
                   return binding.guest_binding == 0;
               }) ||
               std::ranges::any_of(shader.constant_buffer_windows, [](const auto& window) {
                   return window.guest_binding == 0 && window.window_bytes > graphics_base_instance_offset;
               });
    };
    for (std::size_t stage = 0; stage < stage_count; ++stage) {
        if (stage == NVN_SHADER_STAGE_COMPUTE) continue;
        const auto& program = state.stage_programs[stage];
        const auto* shader = program ? program->stages[stage].get() : nullptr;
        if (!shader || !reads_word(*shader)) continue;
        invalidate_args(stage);
        ++bindings[stage].driver_generation[0];
        bindings[stage].windows[0] = {};
    }
}

std::array<std::byte, compute_driver_data_bytes> Encoder::driver_constants(std::size_t stage, std::uint32_t slot) const {
    std::array<std::byte, compute_driver_data_bytes> bytes{};
    bytes.fill(std::byte{0});
    // The push block: past it the constant buffer reads zero.
    if (slot >= 3 && slot == state.push_bank_at(stage)) {
        std::memcpy(bytes.data(), state.push_at(stage).data(), push_constant_bytes);
        return bytes;
    }
    const auto put = [&](std::size_t offset, const void* value, std::size_t size) {
        if (offset + size <= bytes.size()) std::memcpy(bytes.data() + offset, value, size);
    };
    const auto* layout = stage == NVN_SHADER_STAGE_COMPUTE && slot == 0 ? compute_layout() : nullptr;
    if (layout) {
        put(4, dispatch_groups.data(), sizeof dispatch_groups);
        put(0x10, state.push_at(stage).data(), std::min<std::size_t>(layout->push_bytes, push_constant_bytes));
        const auto put_buffers = [&](std::uint32_t base, std::uint32_t count, const auto& buffers) {
            for (std::size_t row = 0; row < std::min<std::size_t>(count, buffer_slots); ++row) {
                const auto size = static_cast<std::uint32_t>(std::min<std::uint64_t>(buffers[row].size, UINT32_MAX));
                put(base + 16 * row, &buffers[row].address, 8);
                put(base + 16 * row + 8, &size, 4);
            }
        };
        put_buffers(layout->uniform_base(), layout->uniforms, state.uniform_at(stage));
        put_buffers(layout->storage_base(), layout->storages, state.storage_at(stage));
        const auto put_handles = [&](std::uint32_t base, std::uint32_t count, const auto& handles,
                                     std::size_t first = 0) {
            for (std::size_t index = 0; index < std::min<std::size_t>(count, handles.size() - first); ++index)
                put(base + 8 * index, &handles[first + index], 8);
        };
        put_handles(layout->texture_base(), layout->textures, state.textures_at(stage));
        put_handles(layout->separate_texture_base(), layout->separate_textures, state.textures_at(stage), separate_texture_slot_base);
        put_handles(layout->separate_sampler_base(), layout->separate_samplers, state.samplers_at(stage));
        put_handles(layout->image_base(), layout->images, state.images_at(stage));
        return bytes;
    }
    if (slot == 0) {
        std::memset(bytes.data(), 0xff, bytes.size());
        for (std::uint32_t word = 0; word < 8; ++word) put(4 * word, &word, 4);
        // API 231 task and mesh stages take the tessellation slots, with
        // tables of their own.
        const auto& program = state.stage_programs[stage];
        const auto* bound = program ? program->stages[stage].get() : nullptr;
        const int base = stage == NVN_SHADER_STAGE_VERTEX ? 0x110
            : stage == NVN_SHADER_STAGE_COMPUTE ? 0x310
            : stage == NVN_SHADER_STAGE_FRAGMENT ? 0x510
            : stage == NVN_SHADER_STAGE_TESS_CONTROL && bound && bound->task ? 0x670
            : stage == NVN_SHADER_STAGE_TESS_EVALUATION && bound && bound->mesh ? 0x770 : -1;
        if (base >= 0)
            for (std::size_t row = 0; row < buffer_slots; ++row) {
                const auto& buffer = state.storage_at(stage)[row];
                const auto size = static_cast<std::uint32_t>(std::min<std::uint64_t>(buffer.size, UINT32_MAX));
                put(base + 16 * row, &buffer.address, 8);
                put(base + 16 * row + 8, &size, 4);
            }
        if (stage != NVN_SHADER_STAGE_COMPUTE)
            put(graphics_base_instance_offset, &draw_base_instance, sizeof draw_base_instance);
        if (stage == NVN_SHADER_STAGE_COMPUTE)
            for (std::size_t row = 0; row < buffer_slots; ++row) {
                const auto& buffer = state.uniform_at(stage)[row];
                const auto size = static_cast<std::uint32_t>(std::min<std::uint64_t>(buffer.size, UINT32_MAX));
                put(0x210 + 16 * row, &buffer.address, 8);
                put(0x210 + 16 * row + 8, &size, 4);
            }
    } else if (slot == 2) {
        put(0x18, &state.prefix_at(stage), sizeof(std::uint64_t));
        for (std::size_t offset = 0x20; offset < 0x968; offset += 8) {
            const auto handle = static_cast<std::uint32_t>(bound_c2_handle(stage, offset));
            put(offset, &handle, 4);
        }
    }
    return bytes;
}

bool Encoder::current_buffer_write_overlap(std::uintptr_t cpu, std::uint64_t size) {
    if (!size || cpu_writes_completed == other_buffer_writes.size()) return false;
    if (cpu_addressed_buffers && cpu) return buffer_writes_overlap(cpu, size, cpu_writes_completed);
    const auto registration = backend.resources().registration_generation();
    if (registration != cpu_writes_registration) {
        cpu_writes_registration = registration;
        cpu_writes_indexed = cpu_writes_completed;
        current_cpu_writes.clear();
        cpu_writes_unknown = false;
    }
    while (cpu_writes_indexed < other_buffer_writes.size()) {
        const auto [address, length] = other_buffer_writes[cpu_writes_indexed++];
        if (!length) continue;
        const auto synchronizations = gpu_synchronizations;
        const auto range = backend.resources().buffer_address(address, length);
        if (failed) return true;
        if (gpu_synchronizations != synchronizations)
            return current_buffer_write_overlap(cpu, size);
        if (!range.buffer) continue;  // Unbacked shader stores target private scratch.
        if (range.cpu) current_cpu_writes.add(reinterpret_cast<std::uintptr_t>(range.cpu), length);
        else cpu_writes_unknown = true;
    }
    return !cpu || cpu_writes_unknown || current_cpu_writes.may_intersect(cpu, size);
}

// Whether [address, address + size) meets one of a guest-memory table's
// sorted disjoint pool ranges.
bool guest_ranges_overlap(const std::vector<std::pair<std::uint64_t, std::uint64_t>>* ranges,
                          std::uint64_t address, std::uint64_t size) {
    if (!ranges || !size) return false;
    const auto after = std::ranges::upper_bound(*ranges, address,
        {}, [](const auto& range) { return range.first; });
    if (after != ranges->begin() && std::prev(after)->second > address) return true;
    return after != ranges->end() && after->first < address + size;
}

// A guest-memory stage since the last GPU wait may have written the range.
bool Encoder::guest_written(std::uint64_t address, std::uint64_t size) const {
    return guest_writes_pending && guest_ranges_overlap(pending.guest_ranges.get(), address, size);
}

// The first storing stage since the last GPU wait. Captured uniform bytes in
// the pools it may write are discarded: the learned pools once stores are
// predicted, and otherwise every pool of the table.
void Encoder::note_guest_writes() {
    ++guest_write_serial;
    pending.guest_writes = true;
    if (std::exchange(guest_writes_pending, true)) return;
    if (backend.apply_guest_write_prediction(pending)) {
        if (const auto* ranges = pending.guest_ranges.get())
            for (const auto& [begin, end] : *ranges)
                if (end > begin) discard_gpu_uniform_captures(begin, end - begin);
        return;
    }
    // No prediction for this store (not settled, switched off, or over the
    // cap): it may write anywhere in the table, also when an earlier segment
    // of this submission was encoded from a prediction.
    if (auto ranges = backend.guest_table_ranges()) pending.guest_ranges = std::move(ranges);
    if (const auto* pools = backend.guest_table_pools())
        for (const auto& [begin, length] : *pools) discard_gpu_uniform_captures(begin, length);
}

bool Encoder::order_cpu_read(std::uint64_t address, std::uint64_t size, bool overlay,
                             const std::byte* cpu, bool current) {
    const SiteScope site_scope(*this, TransferSite::cpu_read, true);
    for (const auto& previous : previous_submissions) {
        if (previous->fence <= retired_fence) continue;
        // Captured uniform bytes can be overlaid without waiting; ordinary
        // GPU writers must finish before their backing is read on the CPU.
        const auto& writes = overlay ? previous->other_buffer_writes : previous->buffer_writes;
        // A guest-memory stage that stores may have written the pools recorded for it.
        const bool guest = previous->guest_writes &&
            guest_ranges_overlap(previous->guest_ranges.get(), address, size);
        const bool buffer = std::ranges::any_of(writes, [&](const auto& write) {
            return address < write.first + write.second && write.first < address + size;
        });
        if (guest || buffer) {
            backend.wait_retired(*previous);
            retired_fence = backend.completed_fence();
        } else if (previous->guest_writes && previous->guest_writes_predicted) {
            // The learned pools excluded this read, so the wait was skipped.
            // The current submission may store nothing of its own.
            backend.note_predicted_guest_read(*previous, address, size);
        }
    }
    if (current) {
        if (!order_completion_read(nil, 0, cpu, size)) return false;
        const auto pointer = reinterpret_cast<std::uintptr_t>(cpu);
        const bool version_alias = !cpu_addressed_buffers && pointer && std::ranges::any_of(uniform_versions, [&](const auto& version) {
            if (!version.source_cpu) return false;
            const auto [first, last] = overlap_offsets(version.source_cpu, version.end - version.begin, pointer, size);
            return first != last && version.begin + first != address + (version.source_cpu + first - pointer);
        });
        if (version_alias && !flush_uniforms()) return false;
        const bool copied_alias = !cpu_addressed_buffers &&
            (uniform_copy_unknown || mapping_alias(uniform_copy_mappings, pointer, address, size));
        const bool written = copied_alias || guest_written(address, size) || current_buffer_write_overlap(pointer, size);
        if (failed || (written && !synchronize_gpu())) return false;
        if (!written && pending.guest_writes_predicted && guest_writes_pending)
            backend.note_predicted_guest_read(pending, address, size);
    }
    return !failed;
}

void Encoder::record_uniform_update(std::uint64_t address, std::uint64_t size,
                                    const std::byte* bytes, const BufferSpan& destination) {
    ++pending.counters.uniform_updates;
    ++update_generation;
    const auto* contents = cpu_addressed_buffers ? reinterpret_cast<const std::byte*>(address) :
        static_cast<const std::byte*>(destination.buffer.contents);
    const auto cpu = cpu_addressed_buffers ? address :
        contents ? reinterpret_cast<std::uintptr_t>(contents + destination.offset) : 0;
    const auto gpu = cpu_addressed_buffers ? 0 : destination.buffer.gpuAddress + destination.offset;
    // Keep the latest disjoint pieces. An outer rewrite must not discard an
    // older tail, and normalization bounds fragmentation by captured bytes.
    if (!uniform_updates.empty() && captured_guest_bytes.may_intersect(address, size)) {
        // Captures are disjoint in guest address space. An exact replacement
        // changes no range or backing index and needs no vector allocation.
        for (auto& old : uniform_updates)
            if (old.address == address && old.size == size && old.cpu == cpu && old.gpu == gpu) {
                old.bytes = bytes;
                return;
            }
        std::vector<UniformUpdate> kept;
        kept.reserve(uniform_updates.size() + 1);
        for (const auto& old : uniform_updates) {
            const auto [first, last] = overlap_offsets(old.address, old.size, address, size);
            if (first == last) kept.push_back(old);
            else {
                if (first) kept.push_back(uniform_slice(old, 0, first));
                if (last < old.size) kept.push_back(uniform_slice(old, last, old.size - last));
            }
        }
        uniform_updates = std::move(kept);
    }
    uniform_updates.push_back({address, size, bytes, cpu, gpu});
    capture_retirement_needed |= uniform_updates.size() >= 4096;
    captured_guest_bytes.add(address, size);
    const auto prefix = address & 3;
    const auto covered = (prefix + size + 3) & ~std::uint64_t{3};
    if (cpu) captured_cpu_words.add(cpu - prefix, covered);
    if (gpu) captured_gpu_words.add(gpu - prefix, covered);
    note_update_pages(address, size);
}

void Encoder::refresh_uniform_words(std::uint64_t address, std::uint64_t size, const std::byte* bytes) {
    const auto first = address & ~std::uint64_t{3};
    const auto last = (address + size - 1) & ~std::uint64_t{3};
    const auto words = (last - first) / 4 + 1;
    const auto refresh = [&](auto found) {
        const auto word = found->first;
        const auto [begin, end] = overlap_offsets(word, 4, address, size);
        if (begin != end) {
            if (begin || end != 4) return uniform_words.erase(found);
            std::memcpy(&found->second, bytes + word - address, 4);
        }
        return ++found;
    };
    // Update only existing cache entries, with work bounded by the smaller
    // of the touched words and the populated cache. Unqueried data costs none.
    if (words < uniform_words.size()) {
        for (std::uint64_t index = 0; index < words; ++index)
            if (const auto found = uniform_words.find(first + index * 4); found != uniform_words.end())
                refresh(found);
    } else {
        for (auto found = uniform_words.begin(); found != uniform_words.end();) found = refresh(found);
    }
}

void Encoder::discard_uniform_captures(std::uintptr_t cpu, std::uint64_t gpu, std::uint64_t size) {
    if (!size || uniform_updates.empty() || capture_retirement_needed) return;
    if ((!cpu || !captured_cpu_words.may_intersect(cpu, size)) &&
        (!gpu || !captured_gpu_words.may_intersect(gpu, size))) return;
    const auto overlap = [&](const UniformUpdate& update, bool words) {
        const auto prefix = words ? update.address & 3 : 0;
        const auto length = words ? (prefix + update.size + 3) & ~std::uint64_t{3} : update.size;
        if (cpu && update.cpu) {
            const auto hit = overlap_offsets(update.cpu - prefix, length, cpu, size);
            if (hit.first != hit.second) return hit;
        }
        if (gpu && update.gpu) return overlap_offsets(update.gpu - prefix, length, gpu, size);
        return std::pair<std::uint64_t, std::uint64_t>{};
    };
    bool changed = false;
    for (const auto& update : uniform_updates) {
        const auto [first, last] = overlap(update, true);
        if (first == last) continue;
        changed = true;
        const auto base = update.address & ~std::uint64_t{3};
        for (auto word = (base + first) & ~std::uint64_t{3}; word < base + last; word += 4)
            uniform_words.erase(word);
    }
    if (!changed) return;
    std::vector<UniformUpdate> kept;
    kept.reserve(uniform_updates.size() + 1);
    for (const auto& update : uniform_updates) {
        const auto [first, last] = overlap(update, false);
        if (first == last) kept.push_back(update);
        else {
            if (first) kept.push_back(uniform_slice(update, 0, first));
            if (last < update.size) kept.push_back(uniform_slice(update, last, update.size - last));
        }
    }
    uniform_updates = std::move(kept);
    ++update_generation;
    rebuild_capture_ranges();
    capture_retirement_needed |= uniform_updates.size() >= 4096;
}

bool Encoder::retire_captures_if_full() {
    if (!capture_retirement_needed && uniform_updates.size() < 4096 && uniform_copy_count < 4096) return true;
    // Retire only at a command boundary: the previous writer has been fully
    // emitted. Completion makes every retained byte available in its backing.
    if (!synchronize_gpu()) return false;
    uniform_updates.clear();
    uniform_words.clear();
    update_pages = {};
    rebuild_capture_ranges();
    capture_retirement_needed = false;
    ++update_generation;
    invalidate_args_all();
    return true;
}

void Encoder::rebuild_capture_ranges() {
    captured_guest_bytes.clear();
    captured_cpu_words.clear();
    captured_gpu_words.clear();
    for (const auto& update : uniform_updates) {
        captured_guest_bytes.add(update.address, update.size);
        const auto prefix = update.address & 3;
        const auto size = (prefix + update.size + 3) & ~std::uint64_t{3};
        if (update.cpu) captured_cpu_words.add(update.cpu - prefix, size);
        if (update.gpu) captured_gpu_words.add(update.gpu - prefix, size);
    }
}

bool Encoder::uniform_capture_alias(std::uint64_t address, std::uint64_t size,
                                    const BufferSpan& span) const {
    if (cpu_addressed_buffers) return false;
    const auto cpu = span.buffer.contents
        ? reinterpret_cast<std::uintptr_t>(span.buffer.contents) + span.offset : 0;
    const auto gpu = span.buffer.gpuAddress + span.offset;
    if ((!cpu || !captured_cpu_words.may_intersect(cpu, size)) &&
        !captured_gpu_words.may_intersect(gpu, size)) return false;
    for (const auto& update : uniform_updates) {
        const auto prefix = update.address & 3;
        const auto length = (prefix + update.size + 3) & ~std::uint64_t{3};
        const auto different = [&](std::uint64_t old, std::uint64_t next) {
            if (!old || !next) return false;
            const auto [first, last] = overlap_offsets(old - prefix, length, next, size);
            return first != last && update.address - prefix + first != address + (old - prefix + first - next);
        };
        if (different(update.cpu, cpu) || different(update.gpu, gpu)) return true;
    }
    return false;
}

void Encoder::discard_gpu_uniform_captures(std::uint64_t address, std::uint64_t size) {
    if (!size || uniform_updates.empty()) return;
    if (cpu_addressed_buffers && !captured_cpu_words.may_intersect(address, size)) return;
    const auto range = backend.resources().buffer_address(address, size);
    if (failed) return;
    // Unbacked writable bindings use private scratch; they do not overwrite
    // any guest capture. Their absence is not a missing-resource error.
    if (!range.buffer) return;
    discard_uniform_captures(reinterpret_cast<std::uintptr_t>(range.cpu), range.gpu, size);
}

void Encoder::overlay_uniform(std::uint64_t address, std::byte* bytes, std::size_t size,
                               std::uint32_t* covered, bool current_only) const {
    // Updates are appended in command order and one window is updated at one
    // offset about two hundred times a frame: the range is compared before any
    // address arithmetic and a scan stops at the first update past the range
    // when the list has kept its order, so a fill costs the updates it covers.
    ++pending.counters.overlay_calls;
    if (!updates_touch(address, size)) return;
    const auto apply = [&](const auto& updates) {
        // The list is in command order; only a list whose addresses happen to
        // ascend may stop at the first update past the range.
        bool ordered = true;
        for (std::size_t index = 1; index < updates.size(); ++index)
            if (updates[index].address < updates[index - 1].address) {
                ordered = false;
                break;
            }
        for (const auto& update : updates) {
            if (update.address >= address + size) {
                if (ordered) break;
                continue;
            }
            if (update.address + update.size <= address) continue;
            const auto begin = std::max(address, update.address);
            const auto end = std::min(address + size, update.address + update.size);
            pending.counters.overlay_bytes += end - begin;
            std::memcpy(bytes + begin - address, update.bytes + begin - update.address, end - begin);
            // A capture kept past a predicted store can supply bytes that
            // store may still be writing. The submission that predicted is
            // the one a later new pool is checked against.
            if (pending.guest_writes_predicted && pending.guest_writes)
                backend.note_predicted_guest_read(pending, begin, end - begin);
            if (covered)
                for (auto byte = begin - address; byte < std::min<std::uint64_t>(end - address, 32); ++byte)
                    *covered |= 1u << byte;
        }
    };
    if (!current_only)
        for (const auto& previous : previous_submissions)
            if (previous->fence > retired_fence) apply(previous->uniform_updates);
    apply(uniform_updates);
}

bool Encoder::fill_uniform(UniformVersion& version, std::uint64_t end) {
    end = std::min(end, version.end);
    if (end <= version.valid_end) return true;
    const auto address = version.valid_end;
    // Reading the guest bytes may wait for the GPU. The wait writes every
    // listed version back to guest memory and ends it, so a listed version is
    // not touched afterwards; one being flushed is held by the flush itself.
    const bool listed = &version >= uniform_versions.data() &&
                        &version < uniform_versions.data() + uniform_versions.size();
    const auto synchronizations = gpu_synchronizations;
    const auto source = buffer_at(address, end - address);
    if (listed && gpu_synchronizations != synchronizations) return !failed;
    if (!source.buffer.contents)
        return fail(ReplayErrorCode::missing_resource, "uniform version source is unavailable");
    // A live version is flushed before a later overlapping writer. Reading
    // its base must not wait for a writer that has not yet been emitted.
    if (!order_cpu_read(address, end - address, true,
            static_cast<const std::byte*>(source.buffer.contents) + source.offset, false)) return false;
    if (listed && gpu_synchronizations != synchronizations) return !failed;
    auto* bytes = version.slice.cpu + address - version.begin;
    ++pending.counters.fill_calls;
    pending.counters.fill_bytes += end - address;
    std::memcpy(bytes, static_cast<const std::byte*>(source.buffer.contents) + source.offset, end - address);
    // Newly copied bytes may intersect earlier pending writes even when no
    // update was recorded since the preceding prefix was materialized.
    overlay_uniform(address, bytes, end - address);
    version.valid_end = end;
    return true;
}

bool Encoder::uniform_cpu_conflict(std::uintptr_t cpu, std::uint64_t size) const {
    if (!cpu || !size || uniform_versions.empty() || cpu >= uniform_cpu_high ||
        (cpu < uniform_cpu_low && size <= uniform_cpu_low - cpu)) return false;
    return std::ranges::any_of(uniform_versions, [&](const auto& version) {
        const auto hit = overlap_offsets(version.source_cpu, version.end - version.begin, cpu, size);
        return version.source_cpu && hit.first != hit.second;
    });
}

bool Encoder::uniform_conflict(std::uint64_t address, std::uint64_t size) {
    if (!address || !size || uniform_versions.empty()) return false;
    if (std::ranges::any_of(uniform_versions, [&](const auto& version) {
            const auto hit = overlap_offsets(version.begin, version.end - version.begin, address, size);
            return hit.first != hit.second;
        })) return true;
    if (cpu_addressed_buffers) return false;
    const auto synchronizations = gpu_synchronizations;
    const auto range = backend.resources().buffer_address(address, size);
    if (failed) return true;
    if (gpu_synchronizations != synchronizations) return !uniform_versions.empty();
    if (!range.buffer) return true;
    return uniform_cpu_conflict(reinterpret_cast<std::uintptr_t>(range.cpu), size) ||
        std::ranges::any_of(uniform_versions, [&](const auto& version) {
            const auto hit = overlap_offsets(version.source_gpu, version.end - version.begin, range.gpu, size);
            return version.source_gpu && range.gpu && hit.first != hit.second;
        });
}

// Different GPU addresses over the same CPU bytes require alias cache
// visibility, not just execution order or ordinary device visibility.
void Encoder::record_alias_mapping(std::vector<PhysicalMapping>& mappings,
                                   std::uintptr_t cpu, std::uint64_t gpu, std::uint64_t size) {
    if (!cpu || !gpu || !size) return;
    const auto end = cpu + size;
    if (mappings.empty() || mappings.back().cpu + mappings.back().size <= cpu) {
        if (!mappings.empty() && mappings.back().cpu + mappings.back().size == cpu &&
            mappings.back().gpu && mappings.back().gpu + mappings.back().size == gpu)
            mappings.back().size += size;
        else mappings.push_back({cpu, gpu, size});
        return;
    }
    const auto first = std::lower_bound(mappings.begin(), mappings.end(), cpu,
        [](const auto& mapping, auto begin) { return mapping.cpu + mapping.size <= begin; });
    if (first != mappings.end() && first->cpu <= cpu &&
        end <= first->cpu + first->size && (!first->gpu || first->gpu + cpu - first->cpu == gpu)) return;
    std::vector<PhysicalMapping> next;
    next.reserve(mappings.size() + 2);
    const auto append = [&](std::uintptr_t begin, std::uint64_t limit, std::uint64_t native) {
        if (begin >= limit) return;
        if (!next.empty()) {
            auto& prior = next.back();
            if (prior.cpu + prior.size == begin &&
                ((!prior.gpu && !native) || (prior.gpu && native && prior.gpu + prior.size == native))) {
                prior.size += limit - begin;
                return;
            }
        }
        next.push_back({begin, native, limit - begin});
    };
    auto cursor = cpu;
    for (const auto& old : mappings) {
        const auto old_end = old.cpu + old.size;
        if (old_end <= cpu) { append(old.cpu, old_end, old.gpu); continue; }
        if (old.cpu >= end) {
            append(cursor, end, gpu + cursor - cpu); cursor = end;
            append(old.cpu, old_end, old.gpu); continue;
        }
        append(old.cpu, std::min<std::uint64_t>(cpu, old_end), old.gpu);
        append(cursor, std::min<std::uint64_t>(old.cpu, end), gpu + cursor - cpu);
        const auto begin = std::max<std::uint64_t>(cpu, old.cpu);
        const auto limit = std::min<std::uint64_t>(end, old_end);
        const auto native = gpu + begin - cpu;
        append(begin, limit, old.gpu && old.gpu + begin - old.cpu == native ? native : 0);
        cursor = limit;
        append(limit, old_end, old.gpu ? old.gpu + limit - old.cpu : 0);
    }
    append(cursor, end, gpu + cursor - cpu);
    // Each insertion adds at most two endpoints; callers bound their histories.
    mappings = std::move(next);
}

bool Encoder::mapping_alias(const std::vector<PhysicalMapping>& mappings,
                            std::uintptr_t cpu, std::uint64_t gpu, std::uint64_t size) {
    if (!cpu || !gpu || !size) return false;
    auto mapping = std::lower_bound(mappings.begin(), mappings.end(), cpu,
        [](const auto& entry, auto begin) { return entry.cpu + entry.size <= begin; });
    for (; mapping != mappings.end() && mapping->cpu < cpu + size; ++mapping) {
        const auto first = std::max<std::uint64_t>(cpu, mapping->cpu);
        if (!mapping->gpu || mapping->gpu + first - mapping->cpu != gpu + first - cpu) return true;
    }
    return false;
}

bool Encoder::needs_alias_visibility(std::uintptr_t cpu, std::uint64_t gpu, std::uint64_t size) const {
    return mapping_alias(uniform_native_mappings, cpu, gpu, size);
}

void Encoder::record_uniform_copy(std::uint64_t address, std::uint64_t size, const BufferSpan& destination) {
    const auto* contents = static_cast<const std::byte*>(destination.buffer.contents);
    // Keep every actual copy's native mapping, including fallback copies.
    // The disjoint index has at most two endpoints per resident destination
    // allocation; repeated updates to one allocation do not grow it.
    record_alias_mapping(uniform_native_mappings, reinterpret_cast<std::uintptr_t>(contents),
                         destination.buffer.gpuAddress, destination.buffer.length);
    if (cpu_addressed_buffers) return;
    ++uniform_copy_count;
    capture_retirement_needed |= uniform_copy_count >= 4096;
    if (contents) record_alias_mapping(uniform_copy_mappings,
        reinterpret_cast<std::uintptr_t>(contents + destination.offset), address, size);
    else uniform_copy_unknown = true;
}

bool Encoder::prepare_uniform_access(std::uint64_t address, std::uint64_t size, std::uint64_t second) {
    const SiteScope site_scope(*this, TransferSite::uniform_access, true);
    const bool first_conflict = uniform_conflict(address, size);
    const bool second_conflict = second && uniform_conflict(second, size);
    if (failed || ((first_conflict || second_conflict) && !flush_uniforms())) return false;
    if (uniform_native_mappings.empty() || !size) return true;
    const auto aliased = [&](std::uint64_t begin) {
        if (!begin) return false;
        if (cpu_addressed_buffers) {
            const auto low = uniform_native_mappings.front().cpu;
            const auto& last = uniform_native_mappings.back();
            const auto high = last.cpu + last.size;
            if (begin >= high || (begin < low && size <= low - begin)) return false;
        }
        const auto range = backend.resources().buffer_address(begin, size);
        return needs_alias_visibility(reinterpret_cast<std::uintptr_t>(range.cpu), range.gpu, size);
    };
    const bool alias = aliased(address) || (second && aliased(second));
    if (failed) return false;
    if (!alias) return true;
    if (render) {
        encoder_barrier(render, MTLStageVertex, MTLStageVertex | MTLStageFragment,
                        MTL4VisibilityOptionResourceAlias);
    } else {
        if (!transfer && !open_transfer()) return false;
        encoder_barrier(transfer, MTLStageBlit | MTLStageDispatch, MTLStageBlit | MTLStageDispatch,
                        MTL4VisibilityOptionResourceAlias);
    }
    return !failed;
}

bool Encoder::prepare_graphics_uniform_writes() {
    const SiteScope site_scope(*this, TransferSite::graphics_writes);
    if (uniform_versions.empty()) return true;
    constexpr auto graphics_writers = (1u << NVN_SHADER_STAGE_VERTEX) | (1u << NVN_SHADER_STAGE_FRAGMENT) |
        (1u << NVN_SHADER_STAGE_GEOMETRY) | (1u << NVN_SHADER_STAGE_TESS_CONTROL) |
        (1u << NVN_SHADER_STAGE_TESS_EVALUATION);
    if (state.write_stages & graphics_writers) return flush_uniforms();
    // Target snapshots change only at a target bind. Reject disjoint CPU
    // bounds before inspecting each attachment on every draw.
    if (uniform_cpu_high <= attachment_cpu_low || attachment_cpu_high <= uniform_cpu_low) return true;
    const auto overlaps = [&](const TextureState* target) {
        return target && target->memory && uniform_cpu_conflict(
            reinterpret_cast<std::uintptr_t>(target->memory), target->size) &&
            backend.resources().texture_is_in_place(*target);
    };
    for (const auto* target : state.targets.color)
        if (overlaps(target)) return flush_uniforms();
    return !overlaps(state.targets.depth) || flush_uniforms();
}

bool Encoder::version_uniform(std::uint64_t address, std::uint64_t size, const std::byte* bytes,
                               const BufferSpan& destination) {
    std::uint64_t begin = 0, end = 0;
    for (const auto& stage : state.uniform) {
        for (const auto& binding : stage)
            if (binding.address && binding.address <= address &&
                address + size <= binding.address + binding.size) {
                begin = binding.address;
                end = begin + binding.size;
                break;
            }
        if (end) break;
    }
    if (!end || end - begin > 64 * 1024 || pending.counters.uniform_updates >= 4096) return false;
    const auto* contents = cpu_addressed_buffers ? reinterpret_cast<const std::byte*>(address) :
        static_cast<const std::byte*>(destination.buffer.contents);
    const auto delta = address - begin;
    if (!contents || delta > destination.offset) return false;
    const auto source_cpu = cpu_addressed_buffers ? std::uintptr_t(begin) :
        reinterpret_cast<std::uintptr_t>(contents + destination.offset - delta);
    const auto source_gpu = cpu_addressed_buffers ? 0 : destination.buffer.gpuAddress + destination.offset - delta;
    if (!cpu_addressed_buffers && (uniform_copy_unknown ||
        mapping_alias(uniform_copy_mappings, source_cpu, begin, end - begin) ||
        current_buffer_write_overlap(source_cpu, end - begin))) return false;
    const auto overlaps = [&](std::uint64_t address, std::uint64_t size) {
        return address && size && address < end && begin < address + size;
    };
    // Inactive logical banks remain observable after a geometry-mode change.
    // A version must cover every overlapping binding, not only active slots.
    for (const auto& stage : state.uniform)
        for (const auto& binding : stage)
            if (overlaps(binding.address, binding.size) &&
                (binding.address < begin || binding.address + binding.size > end)) return false;
    for (const auto& stage : state.storage)
        for (const auto& binding : stage)
            if (overlaps(binding.address, binding.size)) return false;
    for (const auto& binding : state.vertex_buffers)
        if (overlaps(binding.address, binding.size)) return false;
    for (const auto& [address, size] : other_buffer_writes)
        if (overlaps(address, size)) return false;
    // Bytes a guest-memory stage may still store are not copied into a version.
    if (guest_written(begin, end - begin)) return false;
    UniformVersion* version = nullptr;
    for (auto& candidate : uniform_versions)
        if (overlaps(candidate.begin, candidate.end - candidate.begin)) {
            if (candidate.begin != begin || candidate.end != end) return false;
            version = &candidate;
        }
    if (!version) {
        if (uniform_versions.size() >= 64) return false;
        ++pending.counters.version_creations;
        uniform_versions.push_back({begin, end, begin, address, address + size, {}, source_cpu, source_gpu});
        uniform_cpu_low = std::min(uniform_cpu_low, source_cpu);
        uniform_cpu_high = std::max(uniform_cpu_high, source_cpu + std::uintptr_t(end - begin));
        version = &uniform_versions.back();
    }
    version->dirty_begin = std::min(version->dirty_begin, address);
    version->dirty_end = std::max(version->dirty_end, address + size);
    record_uniform_update(address, size, bytes, destination);
    // One upload slice carries the whole window for the submission's lifetime
    // and is written in place. Reallocating a slice per extension copied the
    // window prefix again each time: the game updates one 64 KB window about
    // 220 times a frame, mostly 16 bytes at a time.
    if (!version->slice) {
        const auto slice = backend.allocate_upload(pending, end - begin);
        if (!slice)
            return fail(ReplayErrorCode::allocation, "uniform version allocation failed");
        version->slice = slice;
        // A new slice carries a new GPU address, which the stage tables encode.
        invalidate_uniforms_all();
        ++backend.mutable_statistics().uniform_versions;
        return true;
    }
    if (address < version->valid_end) {
        // Any overlap with the readable prefix needs a new version, including
        // an update that straddles its end. Earlier segments may be executing.
        // The prefix and the update are copied on the encode thread, in order
        // with every later copy and with the records that lead to GPU reads
        // of the version. The prefix grows over a straddling update, so a
        // later fill on this thread never writes the bytes those copies own.
        const auto next = backend.allocate_upload(pending, end - begin);
        if (!next)
            return fail(ReplayErrorCode::allocation, "uniform version allocation failed");
        copy_memory(next.cpu, version->slice.cpu, version->valid_end - begin);
        backend.mutable_statistics().uniform_version_bytes += version->valid_end - begin;
        copy_memory(next.cpu + (address - begin), bytes, size);
        version->valid_end = std::max(version->valid_end, address + size);
        version->slice = next;
        invalidate_uniforms_all();
        ++backend.mutable_statistics().uniform_versions;
        return true;
    }
    // The update lies beyond the readable prefix and will be overlaid on the
    // copied guest bytes when the prefix grows.
    return true;
}

std::uint64_t Encoder::uniform_address(const BoundBuffer& binding, std::uint64_t gpu,
                                      std::uint64_t extent) {
    for (auto& version : uniform_versions)
        if (binding.address >= version.begin && binding.address + binding.size <= version.end) {
            const auto synchronizations = gpu_synchronizations;
            if (!fill_uniform(version, binding.address + extent)) return 0;
            // A GPU wait wrote the version back to the guest buffer.
            if (gpu_synchronizations != synchronizations) return gpu;
            return version.slice.gpu + binding.address - version.begin;
        }
    return gpu;
}

bool Encoder::flush_uniforms() {
    if (failed) return false;
    auto versions = std::move(uniform_versions);
    uniform_versions.clear();
    uniform_cpu_low = UINTPTR_MAX;
    uniform_cpu_high = 0;
    for (auto& version : versions) {
        if (!fill_uniform(version, version.dirty_end)) return false;
        const auto size = version.dirty_end - version.dirty_begin;
        const auto destination = buffer_at(version.dirty_begin, size);
        if (!destination.buffer)
            return fail(ReplayErrorCode::missing_resource, "uniform writeback destination is unavailable");
        // The copy reads a slice only the CPU writes and writes the guest range.
        const GuestAccess writeback{version.dirty_begin, version.dirty_begin + size, true};
        if (!open_transfer_with({&writeback, 1}, true)) return false;
        make_resident(pending, destination.buffer);
        const auto cpu = reinterpret_cast<std::uintptr_t>(destination.buffer.contents);
        const auto gpu = destination.buffer.gpuAddress;
        if (needs_alias_visibility(cpu ? cpu + destination.offset : 0,
                                   gpu + destination.offset, size))
            encoder_barrier(transfer, MTLStageBlit | MTLStageDispatch, MTLStageBlit | MTLStageDispatch,
                            MTL4VisibilityOptionResourceAlias);
        copy_buffer_range(version.slice.buffer, version.slice.offset + version.dirty_begin - version.begin,
                          destination.buffer, destination.offset, size);
        pending.buffer_writes.emplace_back(version.dirty_begin, size);
        record_uniform_copy(version.dirty_begin, size, destination);
        buffer_write_order.write();
        ++backend.mutable_statistics().copies;
    }
    if (!versions.empty()) invalidate_args_all();
    return true;
}

UploadSlice Encoder::upload(std::size_t size, const void* data) {
    auto slice = backend.allocate_upload(pending, size);
    if (!slice) {
        fail(ReplayErrorCode::allocation, "replay upload arena allocation failed");
        return {};
    }
    if (data) std::memcpy(slice.cpu, data, size);
    else std::memset(slice.cpu, 0, size);
    return slice;
}

// An in-stream update to [address, address + size). The draw stages rebind
// their uniform buffers; the rest of a stage's bindings need resolving again
// only when the update lands on a word the stage's shader reads a texture
// handle or storage descriptor from. Other stages rebind in full.
void Encoder::invalidate_uniforms(std::uint64_t address, std::uint64_t size, std::uintptr_t cpu) {
    if (!uniform_rebind_enabled()) return invalidate_args_all();
    for (std::size_t stage = 0; stage < stage_count; ++stage) {
        bindings[stage].uniform_slots = UINT32_MAX;
        if (stage != NVN_SHADER_STAGE_VERTEX && stage != NVN_SHADER_STAGE_FRAGMENT) {
            dirty.arguments[stage] = true;
            continue;
        }
        bool contents = false;
        for (const auto& read : bindings[stage].content_reads) {
            const auto bound = hardware_buffer(stage, read.slot);
            const auto begin = bound.address + read.offset;
            contents = bound.address && address < begin + read.length && begin < address + size;
            if (!contents && !cpu_addressed_buffers && cpu && bound.address) {
                const auto range = backend.resources().buffer_address(begin, read.length);
                const auto [first, last] = overlap_offsets(cpu, size,
                    reinterpret_cast<std::uintptr_t>(range.cpu), range.cpu ? read.length : 0);
                contents = first != last;
            }
            if (contents) break;
        }
        if (contents) dirty.arguments[stage] = true;
        else dirty.uniforms[stage] = true;
    }
}

bool Encoder::buffer_writes_overlap(std::uint64_t address, std::uint64_t size, std::size_t first) {
    if (buffer_writes_indexed > other_buffer_writes.size()) {
        buffer_writes_indexed = 0;
        buffer_write_ranges.clear();
    }
    while (buffer_writes_indexed < other_buffer_writes.size()) {
        const auto& range = other_buffer_writes[buffer_writes_indexed++];
        buffer_write_ranges.add(range.first, range.second);
    }
    if (const auto overlap = buffer_write_ranges.intersection(address, size)) {
        if (!*overlap || first == 0) return *overlap;
    }
    // A full-history hit may belong before a cached window's snapshot. Keep
    // its precise suffix check, and the legacy rules for ambiguous ranges.
    return std::any_of(other_buffer_writes.begin() + std::ptrdiff_t(first),
                       other_buffer_writes.end(), [&](const auto& range) {
        return address < range.first + range.second && range.first < address + size;
    });
}

namespace {

PoolTableKey pool_table_key(const platform::CompiledShaderResource& resource) {
    PoolTableKey key;
    key.sampler = resource.kind == platform::CompiledShaderResourceKind::SamplerPool;
    if (key.sampler) {
        key.normalized_coordinates = true;
        key.comparison = false;
        return key;
    }
    key.normalized_coordinates = false;
    key.comparison = false;
    if (!resource.texture_descriptor) return key;
    const auto& descriptor = *resource.texture_descriptor;
    const bool array = descriptor.array;
    const bool multisample = descriptor.multisample;
    using Dimension = platform::CompiledShaderTextureDimension;
    switch (descriptor.dimension) {
    case Dimension::d1:
        key.type = array ? MTLTextureType1DArray : MTLTextureType1D;
        break;
    case Dimension::d2:
        key.type = multisample ? (array ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample)
                               : (array ? MTLTextureType2DArray : MTLTextureType2D);
        break;
    case Dimension::d3:
        key.type = MTLTextureType3D;
        break;
    case Dimension::cube:
        key.type = array ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
        break;
    case Dimension::buffer:
        key.type = MTLTextureTypeTextureBuffer;
        break;
    }
    using Format = platform::TextureComponentFormat;
    switch (descriptor.component_format) {
    case Format::signed_integer: key.component = PoolComponent::Sint; break;
    case Format::unsigned_integer: key.component = PoolComponent::Uint; break;
    case Format::depth: key.component = PoolComponent::Depth; break;
    default: key.component = PoolComponent::Float; break;
    }
    return key;
}

} // namespace

bool Encoder::stage_has_pool(const platform::CompiledGraphicsStage* shader) const {
    if (!shader) return false;
    for (const auto& resource : shader->resources)
        if (resource.kind == platform::CompiledShaderResourceKind::TexturePool ||
            resource.kind == platform::CompiledShaderResourceKind::SamplerPool)
            return true;
    return false;
}

bool Encoder::refresh_pools(std::size_t stage, const platform::CompiledGraphicsStage& shader) {
    const auto bound = bindings[stage].pool_generation;
    for (const auto& resource : shader.resources) {
        if (resource.kind != platform::CompiledShaderResourceKind::TexturePool &&
            resource.kind != platform::CompiledShaderResourceKind::SamplerPool)
            continue;
        const auto found = backend.pool_tables().lookup(pool_table_key(resource), backend.resources(),
                                                        pending.fence, backend.completed_fence(),
                                                        backend.pool_present_count());
        backend.adopt_pool_stats();
        if (!found.ok)
            return fail(ReplayErrorCode::allocation, "texture pool table is unavailable");
    }
    if (backend.pool_tables().generation() != bound) dirty.arguments[stage] = true;
    return true;
}

void Encoder::retain_pool_version(const std::shared_ptr<PoolTableVersion>& version) {
    if (!version) return;
    auto current = version->retire_fence.load(std::memory_order_relaxed);
    while (current < pending.fence &&
           !version->retire_fence.compare_exchange_weak(current, pending.fence, std::memory_order_relaxed)) {
    }
    // One strong ref for the submission. Each command buffer that binds the
    // version still needs its residency set; a later segment does not keep
    // the previous buffer's sets.
    bool held = false;
    for (const auto& existing : pending.pool_versions) {
        if (existing.get() == version.get()) {
            held = true;
            break;
        }
    }
    if (!held)
        pending.pool_versions.push_back(version);
    if (!version->residency) return;
    for (const auto* attached : pool_residency_attached)
        if (attached == version.get()) return;
    bool attached = false;
    if (pending.encode_thread) {
        pending.encode_thread->put(EncodeOp::residency_set, 0,
                                   ResidentRecord{&pending, (__bridge void*)version->residency});
        attached = true;
    } else if (pending.commands) {
        use_residency_set(pending, pending.commands, version->residency);
        attached = true;
    }
    if (attached)
        pool_residency_attached.push_back(version.get());
}

bool Encoder::bind_pool_resource(std::size_t stage, const platform::CompiledShaderResource& resource,
                                 __unsafe_unretained id<MTL4ArgumentTable> table, bool fragment) {
    if (resource.kind == platform::CompiledShaderResourceKind::TexturePool && !resource.texture_descriptor)
        return fail(ReplayErrorCode::unsupported, "texture pool resource has no element type");
    const auto& location = resource.location;
    if (location.kind != platform::CompiledShaderResourceLocationKind::Direct ||
        location.resource_namespace != platform::CompiledShaderResourceNamespace::Buffer ||
        location.first >= 31)
        return fail(ReplayErrorCode::unsupported, "texture pool table binding is outside the buffer namespace");
    const auto found = backend.pool_tables().lookup(pool_table_key(resource), backend.resources(), pending.fence,
                                                    backend.completed_fence(), backend.pool_present_count());
    backend.adopt_pool_stats();
    if (!found.diagnostic.empty())
        return fail(ReplayErrorCode::unsupported_resource, found.diagnostic);
    if (!found.ok || !found.version)
        return fail(ReplayErrorCode::allocation, "texture pool table is unavailable");
    write_buffer_slot(table, stage, location.first, found.version->gpu);
    make_resident(pending, found.version->buffer);
    if (found.bitmap) make_resident(pending, found.bitmap);
    retain_pool_version(found.version);
    bindings[stage].pool_generation = backend.pool_tables().generation();
    if (fragment && render && resource.kind == platform::CompiledShaderResourceKind::TexturePool &&
        state.targets.depth) {
        const auto& depth = *state.targets.depth;
        const auto storage = pool_storage_key(reinterpret_cast<std::uint64_t>(depth.memory), depth.size,
                                              &depth.description, sizeof depth.description);
        if (backend.resources().pool_depth_read(depth.serial, storage)) depth_barrier_reads = true;
    }
    return true;
}

std::uint64_t Encoder::bound_c2_handle(std::size_t stage_index, std::uint64_t offset) const {
    if (offset == 0x18) return static_cast<std::uint32_t>(state.prefix_at(stage_index));
    if (offset < 0x20 || (offset - 0x20) % 8) return 0;
    if (offset < 0x168) {
        const auto slot = (offset - 0x20) / 8;
        if (slot >= 32 && slot < 40 && state.images_at(stage_index)[slot - 32])
            return state.images_at(stage_index)[slot - 32];
        return state.textures_at(stage_index)[slot];
    }
    if (offset < 0x568)
        return state.textures_at(stage_index)[separate_texture_slot_base + (offset - 0x168) / 8];
    if (offset < 0x968) return state.samplers_at(stage_index)[(offset - 0x568) / 8];
    return 0;
}

std::int32_t Encoder::bound_texture_id_at(std::size_t stage_index, std::size_t slot, bool image) const {
    if (slot >= (image ? image_slots : texture_slots)) return 0;
    const auto handle = image ? state.images_at(stage_index)[slot]
                              : state.textures_at(stage_index)[slot];
    return static_cast<std::int32_t>(handle_texture_id(handle));
}

// The constant-buffer word a bound handle, or a handle program's cbuf
// operation, reads. `offset` is the byte offset in that slot.
std::uint64_t Encoder::read_stage_cbuf_word(std::size_t stage_index, std::uint32_t slot,
                                            std::uint64_t offset) {
    const bool compute = stage_index == NVN_SHADER_STAGE_COMPUTE;
    if (slot == 2) {
        if (offset == 0x1c) return state.prefix_at(stage_index) >> 32;
        return static_cast<std::uint32_t>(bound_c2_handle(stage_index, offset));
    }
    if (slot == 0 && compute) {
        // An API 231 compute stage's handle runs, 8 bytes a slot.
        const auto* layout = compute_layout();
        if (!layout) return 0;
        const auto entry = [&](std::uint32_t base, std::uint32_t count, const auto& handles,
                               std::size_t first = 0) -> std::optional<std::uint64_t> {
            if (offset < base || offset >= base + 8ull * count) return std::nullopt;
            const auto index = (offset - base) / 8;
            return (offset - base) % 8 || first + index >= handles.size() ? 0 : handles[first + index];
        };
        for (const auto& found : {entry(layout->texture_base(), layout->textures, state.textures_at(stage_index)),
                entry(layout->separate_texture_base(), layout->separate_textures, state.textures_at(stage_index), separate_texture_slot_base),
                entry(layout->separate_sampler_base(), layout->separate_samplers, state.samplers_at(stage_index)),
                entry(layout->image_base(), layout->images, state.images_at(stage_index))})
            if (found) return *found;
    }
    if (slot == 0 || (slot >= 3 && slot == state.push_bank_at(stage_index))) {
        // Handle programs read the same generated words as the GPU stage,
        // including the addresses in the graphics storage-buffer tables.
        const auto driver = driver_constants(stage_index, slot);
        if (offset > driver.size() || sizeof(std::uint32_t) > driver.size() - offset) return 0;
        std::uint32_t word = 0;
        std::memcpy(&word, driver.data() + offset, sizeof word);
        return word;
    }
    if (slot >= buffer_slots + 3) return 0;
    const auto binding = hardware_buffer(stage_index, slot);
    if (!binding.address || offset > binding.size || sizeof(std::uint32_t) > binding.size - offset)
        return 0;
    const auto address = binding.address + offset;
    if ((address & 3) == 0)
        if (const auto found = uniform_words.find(address); found != uniform_words.end())
            return found->second;
    std::uint32_t handle = 0, covered = 0;
    if (!uniform_updates.empty() && captured_guest_bytes.may_intersect(address, sizeof handle)) {
        // A complete current capture supersedes earlier writes without a
        // backing read. Partial captures and prior submissions cannot be
        // cached: their remaining bytes still need normal read ordering.
        overlay_uniform(address, reinterpret_cast<std::byte*>(&handle), sizeof handle, &covered, true);
        if (covered == 0xf) {
            if (!(address & 3)) uniform_words.emplace(address, handle);
            return handle;
        }
        handle = covered = 0;
    }
    const auto span = buffer_at(address, sizeof(std::uint32_t));
    const auto* contents = static_cast<const std::byte*>(span.buffer.contents);
    if (contents && span.offset <= span.buffer.length && sizeof handle <= span.buffer.length - span.offset) {
        // With the title's current handle reads, GPU writers of this
        // submission are not waited for: the handle is memory as it is now,
        // under the captured uniform updates applied below.
        const bool current = nvn::title_policy().handle_reads == nvn::TitlePolicy::HandleReads::ordered;
        if (!order_cpu_read(address, sizeof handle, true, contents + span.offset, current)) return 0;
        std::memcpy(&handle, contents + span.offset, sizeof handle);
        covered = 0xf;
    }
    // A later writer may invalidate one cached word while leaving earlier
    // captured bytes in the rest of it. Apply only those surviving bytes.
    overlay_uniform(address, reinterpret_cast<std::byte*>(&handle), sizeof handle, &covered);
    if (covered != 0xf) {
        fail(ReplayErrorCode::missing_resource, "texture handle backing is not CPU-visible");
        return 0;
    }
    return handle;
}

// Naturally aligned guest word. Zero where nothing mapped covers it.
// Guest-memory stores and ordinary GPU buffer writes that may overlap the
// word finish before the load. An in-place texture write is not one of
// those buffer writes: it is recorded on the texture and as a CPU range,
// so this load waits for that too. Deferred uniform captures are applied
// after the backing read, including captures an earlier submission still
// holds because a lower fence has not retired.
std::uint32_t Encoder::read_stage_guest_word(std::uint64_t address) {
    address &= ~std::uint64_t{3};
    const auto span = buffer_at(address, sizeof(std::uint32_t));
    const auto* contents = static_cast<const std::byte*>(span.buffer.contents);
    const bool mapped = contents && span.offset <= span.buffer.length &&
        sizeof(std::uint32_t) <= span.buffer.length - span.offset;
    const auto* cpu = mapped ? contents + span.offset : nullptr;
    // An address in no memory pool reads 0, under the submission's captured
    // uniform updates: no GPU work writes there, so nothing orders the read.
    if (!mapped) {
        std::uint32_t word = 0, covered = 0;
        overlay_uniform(address, reinterpret_cast<std::byte*>(&word), sizeof word, &covered);
        return covered == 0xf ? word : 0;
    }
    const auto synchronizations = gpu_synchronizations;
    // A title with current handle reads does not wait for this submission's
    // GPU writers here either; earlier submissions' writers still finish.
    const bool current = nvn::title_policy().handle_reads == nvn::TitlePolicy::HandleReads::ordered;
    if (!order_cpu_read(address, sizeof(std::uint32_t), false, cpu, current)) return 0;
    const auto overlaps = [&](std::uint64_t begin, std::uint64_t length) {
        if (!length) return false;
        const auto meets = [](std::uint64_t left, std::uint64_t left_size, std::uint64_t right) {
            return left < right + sizeof(std::uint32_t) && right < left + left_size;
        };
        if (meets(begin, length, address)) return true;
        const auto pointer = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(cpu));
        return cpu && meets(begin, length, pointer);
    };
    const auto texture_overlaps = [&](const TextureState& texture) {
        const auto backing = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texture.memory));
        return texture.memory && texture.size && backend.resources().texture_is_in_place(texture) &&
            overlaps(backing, texture.size);
    };
    for (const auto& previous : previous_submissions) {
        if (previous->fence <= retired_fence) continue;
        const bool written = std::ranges::any_of(previous->other_buffer_writes, [&](const auto& write) {
                return overlaps(write.first, write.second);
            }) || std::ranges::any_of(previous->texture_writes, texture_overlaps);
        if (!written) continue;
        backend.wait_retired(*previous);
        retired_fence = backend.completed_fence();
    }
    // order_cpu_read already drained this submission when it synchronized.
    // A later in-place write is still only in the uncompleted suffix.
    if (current && gpu_synchronizations == synchronizations) {
        bool pending_write = false;
        for (std::size_t index = cpu_writes_completed; index < other_buffer_writes.size(); ++index)
            pending_write = pending_write || overlaps(other_buffer_writes[index].first,
                                                      other_buffer_writes[index].second);
        if (!pending_write)
            for (const auto& texture : pending.texture_writes) {
                if (!texture_overlaps(texture)) continue;
                const auto backing = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(texture.memory));
                for (std::size_t index = cpu_writes_completed; index < other_buffer_writes.size(); ++index)
                    pending_write = pending_write || (other_buffer_writes[index].first == backing &&
                                                      other_buffer_writes[index].second == texture.size);
            }
        if (pending_write && !synchronize_gpu()) return 0;
    }
    std::uint32_t word = 0, covered = 0;
    if (mapped) {
        std::memcpy(&word, cpu, sizeof word);
        covered = 0xf;
    }
    overlay_uniform(address, reinterpret_cast<std::byte*>(&word), sizeof word, &covered);
    if (!mapped && covered != 0xf) return 0;
    return word;
}

std::uint64_t Encoder::read_stage_handle(
    std::size_t stage_index, const platform::CompiledGraphicsStage& shader,
    const platform::CompiledShaderResource& resource, std::uint32_t element,
    std::vector<std::optional<std::uint64_t>>& handle_program_words) {
    if (resource.cbuf_slot == armigo::shader::pipeline::handle_program_slot) {
        const auto byte_offset = std::uint64_t(resource.handle_byte_offset) + element * 4ull;
        if ((byte_offset & 3u) || byte_offset / 4 >= shader.handle_programs.size()) {
            fail(ReplayErrorCode::unsupported, "handle program is malformed");
            return 0;
        }
        const auto index = static_cast<std::size_t>(byte_offset / 4);
        if (handle_program_words[index]) return *handle_program_words[index];
        const auto value = armigo::shader::pipeline::evaluate_handle_program(
            std::span<const std::uint32_t>{shader.handle_programs[index]},
            [&](std::uint32_t slot, std::uint32_t byte_offset) -> std::uint32_t {
                return static_cast<std::uint32_t>(read_stage_cbuf_word(stage_index, slot, byte_offset & ~3u));
            },
            [&](std::uint64_t address) { return read_stage_guest_word(address); });
        if (failed) return 0;
        if (!value) {
            fail(ReplayErrorCode::unsupported, "handle program is malformed");
            return 0;
        }
        handle_program_words[index] = *value;
        return *value;
    }
    return read_stage_cbuf_word(stage_index, resource.cbuf_slot,
                          std::uint64_t(resource.handle_byte_offset) + element * 4ull);
}

bool Encoder::bind_stage(std::size_t stage_index,
                         const platform::CompiledGraphicsStage& shader,
                         __unsafe_unretained id<MTL4ArgumentTable> table, bool vertex,
                         bool uniforms_only) {
    // New writes can affect descriptor-only slots absent from constant-buffer
    // reflection. Rebuild the whole stage to order those resource reads too.
    if (uniforms_only && stage_index < stage_count &&
        (bindings[stage_index].uniform_effects != pending.effects.size() ||
         bindings[stage_index].uniform_writes != other_buffer_writes.size()))
        uniforms_only = false;
    ++backend.mutable_statistics().stage_binds;
    ++pending.counters.stage_binds;
    if (uniforms_only) {
        ++backend.mutable_statistics().stage_uniform_binds;
        ++pending.counters.stage_uniform_binds;
    }
    if (stage_index >= stage_count)
        return fail(ReplayErrorCode::invalid_command, "shader stage index exceeds the replay limit");
    if (!uniforms_only) {
        auto& bound = bindings[stage_index];
        bound.handle_updates = pending.counters.uniform_updates;
        bound.handle_writes = other_buffer_writes.size();
        bound.handle_stores = guest_write_serial;
    }
    // Stages that address guest memory read the submission's pool table.
    if (shader.guest_memory) {
        if (shader.guest_memory->buffer_index >= 31)
            return fail(ReplayErrorCode::unsupported,
                        "guest-memory table index exceeds the Metal argument-table limit");
        if (!pending.guest_table)
            if (const auto error = backend.prepare_guest_table(pending))
                return fail(ReplayErrorCode::unsupported, error);
        write_buffer_slot(table, stage_index, shader.guest_memory->buffer_index, pending.guest_table.gpuAddress);
        if (shader.guest_memory->writes) note_guest_writes();
    }
    if (shader.requires_rasterizer_discard && !state.rasterizer_discard)
        return fail(ReplayErrorCode::unsupported, "a vertex stage without outputs draws only with rasterizer discard");
    if (stage_index == NVN_SHADER_STAGE_COMPUTE &&
        sync_draw_resources())
        for (auto& bound : bindings) bound.windows.fill({});
    auto& stage = bindings[stage_index];
    // An unchanged write history allows a uniform-only rebind to retain the
    // other constant addresses.
    const auto uniform_slots = uniforms_only ? stage.uniform_slots : UINT32_MAX;
    const auto uniform_synchronizations = gpu_synchronizations;
    const auto uniform_version_count = uniform_versions.size();
    const auto finish_uniforms = [&] {
        stage.uniform_slots = 0;
        stage.uniform_effects = pending.effects.size();
        stage.uniform_writes = other_buffer_writes.size();
    };
    if (stage.shader != &shader) {
        for (const auto& window : shader.constant_buffer_windows)
            if (window.zero_padding_bytes != 0 && window.zero_padding_bytes != 16)
                return fail(ReplayErrorCode::unsupported, "constant-buffer padding reflection is invalid");
        stage.window_extents.fill(0);
        for (const auto& window : shader.constant_buffer_windows)
            if (window.guest_binding < stage.window_extents.size())
                stage.window_extents[window.guest_binding] = window.window_bytes;
        // A stage that stores through the guest-memory table may store
        // anywhere in its pools: it writes like a storage-buffer writer.
        stage.writes_buffers = (shader.guest_memory && shader.guest_memory->writes) ||
            std::ranges::any_of(shader.storage_buffers, [](const auto& binding) {
            return binding.access != platform::CompiledShaderResourceAccess::Read;
        });
        const auto reads_push_bank = [](const auto& binding) { return binding.guest_binding == push_constant_bank; };
        stage.reads_push_bank = std::ranges::any_of(shader.constant_buffers, reads_push_bank) ||
                                std::ranges::any_of(shader.constant_buffer_windows, reads_push_bank);
        stage.content_reads.clear();
        for (const auto& resource : shader.resources)
            if (resource.cbuf_slot != 0 && resource.cbuf_slot != 2 &&
                resource.cbuf_slot < buffer_slots + 3)
                for (std::uint32_t element = 0; element != resource.array_length; ++element)
                    stage.content_reads.push_back(
                        {resource.cbuf_slot, resource.handle_byte_offset + element * 4u, 4});
        for (const auto& reflected : shader.storage_buffers)
            if (reflected.cbuf_slot >= 3 && reflected.cbuf_slot < buffer_slots + 3)
                stage.content_reads.push_back(
                    {reflected.cbuf_slot, reflected.descriptor_byte_offset, 12});
        stage.shader = &shader;
    }
    const auto bind_buffer = [&](const BoundBuffer& binding, std::uint32_t index,
                                 std::uint64_t minimum, bool uniform = false,
                                 bool writable = false) -> bool {
        if (!uniform) {
            if (!prepare_uniform_access(binding.address, binding.size)) return false;
            if (stage_index != NVN_SHADER_STAGE_COMPUTE && !close_transfer()) return false;
        }
        if (index >= 31)
            return fail(ReplayErrorCode::unsupported,
                        "shader buffer index exceeds the Metal argument-table limit");
        if (!binding.address) {
            const auto scratch = writable ? upload(64 * 1024) : UploadSlice{};
            id<MTLBuffer> zeros = writable ? scratch.buffer : backend.zero_buffer();
            if (!zeros)
                return fail(ReplayErrorCode::allocation, "empty binding buffer is unavailable");
            write_buffer_slot(table, stage_index, index, writable ? scratch.gpu : zeros.gpuAddress);
            make_resident(pending, zeros);
            return true;
        }
        const auto span = buffer_address(binding.address, std::max(binding.size, minimum));
        if (!span.buffer || span.offset > span.length) {
            // The range is not inside any registered memory pool. The hardware
            // reads zero from a range whose storage is not backed, so the
            // binding reads zeros instead of dropping the draw. A writable
            // fallback gets its own fence-retired slice so shader stores cannot
            // corrupt the immutable defaults or another submission's storage.
            const auto scratch = writable ? upload(64 * 1024) : UploadSlice{};
            id<MTLBuffer> zeros = writable ? scratch.buffer : backend.zero_buffer();
            if (!zeros)
                return fail(ReplayErrorCode::allocation, "zero binding buffer is unavailable");
            write_buffer_slot(table, stage_index, index, writable ? scratch.gpu : zeros.gpuAddress);
            make_resident(pending, zeros);
            return true;
        }
        // A uniform version only has to carry the bytes the shader reads, not
        // the whole range the game bound: the reflected contract is the proven
        // upper bound of the window's reads, and every version copies it.
        const auto extent = uniform && minimum
            ? std::min<std::uint64_t>(binding.size, minimum)
            : binding.size;
        const auto address = uniform ? uniform_address(binding, span.gpu, extent) : span.gpu;
        if (failed) return false;
        if (write_buffer_slot(table, stage_index, index, address))
            make_resident(pending, span.buffer);
        return true;
    };
    // Blocks the driver writes rather than the game: driver data, the
    // bound-texture table and the push block.
    const auto push_bank = state.push_bank_at(stage_index);
    const auto driver_block = [&](std::uint32_t slot) {
        return slot == 0 || slot == 2 || (push_bank && slot == push_bank);
    };
    if (stage.reads_push_bank && selected_api_version() && selected_api_version()->major == 231)
        note_push_bank_reads(stage_index, push_bank == push_constant_bank,
                             state.uniform_at(stage_index)[push_constant_bank - 3].size != 0);
    const auto bind_driver = [&](std::uint32_t slot, std::uint32_t index) -> bool {
        if (index >= 31)
            return fail(ReplayErrorCode::unsupported,
                        "shader buffer index exceeds the Metal argument-table limit");
        auto& cached = stage.driver[slot < 3 ? slot : 3];
        const auto generation = stage.driver_generation[slot < 3 ? slot : 3];
        if (cached.generation == generation && cached.gpu) {
            write_buffer_slot(table, stage_index, index, cached.gpu);
            return true;
        }
        const auto bytes = driver_constants(stage_index, slot);
        const auto buffer = upload(bytes.size(), bytes.data());
        if (!buffer) return false;
        cached = {generation, buffer.gpu};
        if (stage_index == NVN_SHADER_STAGE_COMPUTE && slot == 0) compute_driver_slice = buffer;
        write_buffer_slot(table, stage_index, index, buffer.gpu);
        return true;
    };
    // A window binding's proven read extent: a uniform version has to carry
    // only the bytes the shader can read, not the range the game bound.
    const auto& window_extents = stage.window_extents;
    for (const auto& binding : shader.constant_buffers) {
        if (binding.guest_binding >= buffer_slots + 3)
            return fail(ReplayErrorCode::unsupported, "constant buffer slot exceeds the replay limit");
        if (!(uniform_slots & (1u << binding.guest_binding))) continue;
        if (driver_block(binding.guest_binding)) {
            if (!bind_driver(binding.guest_binding, binding.metal_buffer_index)) return false;
            continue;
        }
        const auto window = window_extents[binding.guest_binding];
        const auto extent = window ? window
            : binding.minimum_size ? binding.minimum_size : 0u;
        if (!bind_buffer(hardware_buffer(stage_index, binding.guest_binding),
                         binding.metal_buffer_index,
                         extent ? extent : binding.minimum_size, true)) return false;
    }
    for (const auto& binding : shader.constant_buffer_windows) {
        if (binding.guest_binding >= buffer_slots + 3 || binding.metal_buffer_index >= 31 ||
            !binding.window_bytes)
            return fail(ReplayErrorCode::unsupported,
                        "constant-buffer window reflection is invalid");
        if (!(uniform_slots & (1u << binding.guest_binding))) continue;
        const auto source_binding = hardware_buffer(stage_index, binding.guest_binding);
        // An unpadded window can bind full guest storage in place.
        // Padded windows always own their zero tail. Shaders read
        // every word of the window without checking the bound size, so a
        // smaller buffer gets a window copy that is zero past its bound size.
        // Earlier in-stream updates reach memory by GPU copies ordered before
        // this draw; an active version carries the later ones. Driver data
        // and ranges the GPU wrote in this submission still use a window.
        if (!binding.zero_padding_bytes && !driver_block(binding.guest_binding) &&
            source_binding.address && source_binding.size >= binding.window_bytes) {
            const std::uint64_t extent = binding.window_bytes;
            const bool gpu_written = buffer_writes_overlap(source_binding.address, extent);
            if (!gpu_written) {
                const auto span = buffer_address(source_binding.address, extent);
                if (span.buffer && span.offset <= span.length &&
                    extent <= span.length - span.offset) {
                    const auto address = uniform_address(source_binding, span.gpu, extent);
                    if (failed) return false;
                    if (write_buffer_slot(table, stage_index, binding.metal_buffer_index, address))
                        make_resident(pending, span.buffer);
                    continue;
                }
            }
        }
        auto &cached = bindings[stage_index].windows[binding.guest_binding];
        const auto written_since = [&](const WindowUpload& snapshot) {
            const auto end = snapshot.source + std::min<std::uint64_t>(snapshot.source_size, snapshot.bytes);
            return buffer_writes_overlap(snapshot.source, end - snapshot.source, snapshot.writes) ||
                (snapshot.guest_writes != guest_write_serial &&
                 guest_ranges_overlap(pending.guest_ranges.get(), snapshot.source, end - snapshot.source));
        };
        if (cached.gpu && cached.bytes == binding.window_bytes &&
            cached.zero_padding_bytes == binding.zero_padding_bytes &&
            cached.source == source_binding.address && cached.source_size == source_binding.size &&
            cached.generation == update_generation &&
            cached.cpu && cached.writes <= other_buffer_writes.size() && !written_since(cached)) {
            ++pending.counters.window_reuses;
            write_buffer_slot(table, stage_index, binding.metal_buffer_index, cached.gpu);
            continue;
        }
        const auto window = upload(binding.backing_bytes());
        if (!window) return false;
        ++pending.counters.window_uploads;
        write_buffer_slot(table, stage_index, binding.metal_buffer_index, window.gpu);
        // The upload is zero-filled: words past the copied bytes read zero, as
        // past the hardware's bound size.
        std::uint64_t filled = 0;
        if (driver_block(binding.guest_binding)) {
            const auto bytes = driver_constants(stage_index, binding.guest_binding);
            filled = std::min<std::size_t>(bytes.size(), binding.window_bytes);
            std::memcpy(window.cpu, bytes.data(), filled);
        } else if (source_binding.address && source_binding.size) {
            const auto copied = std::min<std::uint64_t>(source_binding.size,
                                                        binding.window_bytes);
            // A word only partly inside the bound size reads zero.
            filled = copied & ~std::uint64_t{3};
            const auto source = buffer_at(source_binding.address, copied);
            if (!source.buffer || !source.buffer.contents ||
                source.offset > source.buffer.length ||
                copied > source.buffer.length - source.offset)
                return fail(ReplayErrorCode::missing_resource,
                            "constant-buffer zero window source is unavailable");
            const bool gpu_written = buffer_writes_overlap(source_binding.address, copied) ||
                guest_written(source_binding.address, copied);
            if (gpu_written) {
                const SiteScope site_scope(*this, TransferSite::uniform_window);
                if (!flush_uniforms() || !open_transfer()) return false;
                make_resident(pending, source.buffer);
                if (filled)
                    copy_buffer_range(source.buffer, source.offset, window.buffer, window.offset, filled);
                close_transfer();
                // Only the GPU copy reads the source after encoding.
                [pending.retained addObject:source.buffer];
            } else {
                if (!order_cpu_read(source_binding.address, copied, true,
                        static_cast<const std::byte*>(source.buffer.contents) + source.offset)) return false;
                std::memcpy(window.cpu,
                            static_cast<const std::byte*>(source.buffer.contents) + source.offset, filled);
                overlay_uniform(source_binding.address, window.cpu, filled);
            }
        }
        cached = {window.gpu, source_binding.address, source_binding.size, update_generation,
                  binding.window_bytes, binding.zero_padding_bytes, window.cpu,
                  other_buffer_writes.size(), guest_write_serial};
        write_buffer_slot(table, stage_index, binding.metal_buffer_index, window.gpu);
    }

    // A uniform-only rebind ends here: the rest resolves from bindings an
    // in-stream update does not touch (invalidate_uniforms checks that).
    if (uniforms_only) {
        // Resolving the changed slot can drain and flush versions used by
        // earlier slots. Revisit all addresses after that transition.
        if (gpu_synchronizations != uniform_synchronizations ||
            uniform_versions.size() != uniform_version_count)
            return bind_stage(stage_index, shader, table, vertex, false);
        finish_uniforms();
        return true;
    }
    bindings[stage_index].writes.clear();
    if (backend.pass_dump_enabled()) { stage.dump_reads.clear(); stage.dump_textures.clear(); }
    const bool compute = stage_index == NVN_SHADER_STAGE_COMPUTE;
    if (compute) {
        // Guest constant buffers the GPU reads: every bound range the shader
        // names (driver data is CPU-written upload memory).
        compute_accesses.clear();
        // Guest-memory accesses have no declared range.
        compute_untracked = !shader.constant_buffers_reflected || shader.guest_memory.has_value();
        const auto add_constant = [&](std::uint32_t slot) {
            if (slot == 0 || slot == 2) return;
            const auto bound = hardware_buffer(stage_index, slot);
            if (bound.address && bound.size)
                compute_accesses.push_back({bound.address, bound.address + bound.size, false});
        };
        for (const auto& binding : shader.constant_buffers) add_constant(binding.guest_binding);
        for (const auto& binding : shader.constant_buffer_windows) add_constant(binding.guest_binding);
    }
    for (std::size_t index = 0; index != shader.storage_buffers.size(); ++index) {
        const auto& reflected = shader.storage_buffers[index];
        BoundBuffer resolved{};
        // Only the driver-data tables give the dispatch's exact range;
        // descriptors read from guest memory leave its accesses incomplete.
        bool from_table = false;
        bool descriptor_read = false;
        const auto* layout = compute ? compute_layout() : nullptr;
        if (reflected.cbuf_slot == 0 && layout) {
            // An API 231 compute stage's uniform and storage entries.
            const auto offset = reflected.descriptor_byte_offset;
            const auto uniforms = layout->uniform_base(), storages = layout->storage_base();
            if (offset >= uniforms && offset < storages && !((offset - uniforms) % 16) &&
                (offset - uniforms) / 16 < buffer_slots) {
                resolved = state.uniform_at(stage_index)[(offset - uniforms) / 16];
                from_table = true;
            } else if (offset >= storages && offset < layout->texture_base() && !((offset - storages) % 16) &&
                       (offset - storages) / 16 < buffer_slots) {
                resolved = state.storage_at(stage_index)[(offset - storages) / 16];
                from_table = true;
            }
        } else if (reflected.cbuf_slot == 0) {
            // Descriptors in the driver data: the storage and compute uniform tables.
            const auto offset = reflected.descriptor_byte_offset;
            const int base = stage_index == NVN_SHADER_STAGE_VERTEX ? 0x110
                : stage_index == NVN_SHADER_STAGE_COMPUTE ? 0x310
                : stage_index == NVN_SHADER_STAGE_FRAGMENT ? 0x510
                : stage_index == NVN_SHADER_STAGE_TESS_CONTROL && shader.task ? 0x670
                : stage_index == NVN_SHADER_STAGE_TESS_EVALUATION && shader.mesh ? 0x770 : -1;
            if (base >= 0 && offset >= std::uint32_t(base) && offset < base + 16 * buffer_slots && !(offset % 16)) {
                resolved = state.storage_at(stage_index)[(offset - base) / 16];
                from_table = true;
            } else if (stage_index == NVN_SHADER_STAGE_COMPUTE && offset >= 0x210 &&
                       offset < 0x210 + 16 * buffer_slots && !(offset % 16)) {
                resolved = state.uniform_at(stage_index)[(offset - 0x210) / 16];
                from_table = true;
            }
        } else if (reflected.cbuf_slot >= 3 && reflected.cbuf_slot < buffer_slots + 3) {
            const auto descriptor_buffer = hardware_buffer(stage_index, reflected.cbuf_slot);
            constexpr std::size_t descriptor_size = 12;
            if (descriptor_buffer.address && reflected.descriptor_byte_offset <= descriptor_buffer.size &&
                descriptor_size <= descriptor_buffer.size - reflected.descriptor_byte_offset) {
                const auto descriptor = buffer_at(
                    descriptor_buffer.address + reflected.descriptor_byte_offset, descriptor_size);
                if (descriptor.buffer.contents &&
                    descriptor.offset <= descriptor.buffer.length &&
                    descriptor_size <= descriptor.buffer.length - descriptor.offset) {
                    const auto address = descriptor_buffer.address + reflected.descriptor_byte_offset;
                    if (!order_cpu_read(address, descriptor_size, true,
                            static_cast<const std::byte*>(descriptor.buffer.contents) + descriptor.offset)) return false;
                    std::array<std::byte, descriptor_size> descriptor_bytes{};
                    std::memcpy(descriptor_bytes.data(),
                        static_cast<const std::byte*>(descriptor.buffer.contents) + descriptor.offset,
                        descriptor_size);
                    overlay_uniform(address, descriptor_bytes.data(), descriptor_size);
                    const auto* bytes = descriptor_bytes.data();
                    std::memcpy(&resolved.address, bytes, sizeof resolved.address);
                    std::uint32_t size = 0;
                    std::memcpy(&size, bytes + 8, sizeof size);
                    resolved.size = size;
                    descriptor_read = true;
                }
            }
        }
        if (!from_table && !descriptor_read)
            return fail(ReplayErrorCode::missing_resource,
                        "storage buffer descriptor is unavailable");
        if (!bind_buffer(resolved, reflected.metal_buffer_index, reflected.alignment, false,
                         reflected.access != platform::CompiledShaderResourceAccess::Read)) return false;
        if (backend.pass_dump_enabled() && reflected.metal_buffer_index < stage.dump_buffer_sizes.size())
            stage.dump_buffer_sizes[reflected.metal_buffer_index] = resolved.size;
        if (resolved.address &&
            reflected.access != platform::CompiledShaderResourceAccess::Read)
            bindings[stage_index].writes.emplace_back(resolved.address, resolved.size);
        if (compute && !from_table) compute_untracked = true;
        if (compute && resolved.address)
            compute_accesses.push_back({resolved.address,
                resolved.address + std::max<std::uint64_t>(resolved.size, reflected.alignment),
                reflected.access != platform::CompiledShaderResourceAccess::Read});
    }

    if (shader.constant_buffer_support_index) {
        std::array<std::uint32_t, 18> sizes{};
        for (std::uint32_t index = 0; index != sizes.size(); ++index)
            sizes[index] = index == 0 || index == 2 ? 0xa00
                : static_cast<std::uint32_t>(std::min<std::uint64_t>(
                      hardware_buffer(stage_index, index).size, UINT32_MAX));
        const auto support = upload(sizeof sizes, sizes.data());
        if (!support) return false;
        write_buffer_slot(table, stage_index, *shader.constant_buffer_support_index,
                          support.gpu);
    }

    const auto registrations = draw_sampler_registrations();
    const auto bound_texture_id = [&](std::size_t slot, bool image) -> std::int32_t {
        return bound_texture_id_at(stage_index, slot, image);
    };
    const auto bound_texture = [&](std::size_t slot, bool image) -> id<MTLTexture> {
        const auto id = bound_texture_id(slot, image);
        return id ? sampled_texture(id) : nil;
    };
    const auto resolve_sampler = [&](std::int32_t descriptor_id) -> id<MTLSamplerState> {
        if (!descriptor_id) return nil;
        id<MTLSamplerState> sampler = backend.resources().sampler_for_id(descriptor_id, registrations);
        if (!sampler)
            if (const auto detail = backend.resources().take_sampler_failure(); !detail.empty())
                fail(ReplayErrorCode::unsupported_resource, detail + " shader=" + stage_label(&shader));
        return sampler;
    };
    // The sampler bound at `slot`, else the one in the texture handle bound at
    // `texture_slot`.
    const auto bound_sampler = [&](std::size_t slot, std::size_t texture_slot)
        -> id<MTLSamplerState> {
        const std::uint64_t texture_handle =
            texture_slot < texture_slots ? state.textures_at(stage_index)[texture_slot] : 0;
        std::int32_t id = 0;
        if (slot < sampler_slots && state.samplers_at(stage_index)[slot])
            id = static_cast<std::int32_t>(handle_sampler_id(
                state.samplers_at(stage_index)[slot]));
        if (!id) id = static_cast<std::int32_t>(handle_sampler_id(texture_handle));
        return resolve_sampler(id);
    };
    // One result per program for this bind. The next draw evaluates again.
    std::vector<std::optional<std::uint64_t>> handle_program_words;
    if (!shader.handle_programs.empty())
        handle_program_words.resize(shader.handle_programs.size());
    const auto read_handle = [&](const platform::CompiledShaderResource& resource,
                                 std::uint32_t element) -> std::uint64_t {
        return read_stage_handle(stage_index, shader, resource, element, handle_program_words);
    };


    struct BorderContext {
        CoverageSampler sampler;
        bool bound = false;
    };
    std::vector<BorderContext> borders;
    CoverageBorderRecord* border_records = nullptr;
    if (shader.custom_border) {
        const auto& support = *shader.custom_border;
        if (support.buffer_index >= 31 || support.pairs.empty() || support.pairs.size() > 4096)
            return fail(ReplayErrorCode::unsupported, "custom border record layout is invalid");
        const auto records = upload(support.pairs.size() * sizeof(CoverageBorderRecord));
        if (!records) return false;
        border_records = reinterpret_cast<CoverageBorderRecord*>(records.cpu);
        for (std::size_t i = 0; i != support.pairs.size(); ++i) std::construct_at(border_records + i);
        write_buffer_slot(table, stage_index, support.buffer_index, records.gpu);
        borders.resize(support.pairs.size());
        for (std::size_t i = 0; i != support.pairs.size(); ++i) {
            const auto pair = support.pairs[i];
            const auto texture = std::ranges::find(shader.resources, pair.texture_resource,
                                                  &platform::CompiledShaderResource::resource_id);
            const auto sampler = std::ranges::find(shader.resources, pair.sampler_resource,
                                                  &platform::CompiledShaderResource::resource_id);
            if (texture == shader.resources.end() || sampler == shader.resources.end() ||
                texture->kind != platform::CompiledShaderResourceKind::SampledTexture ||
                sampler->kind != platform::CompiledShaderResourceKind::Sampler ||
                texture->array_length != 1 || sampler->array_length != 1 || !texture->texture_descriptor ||
                texture->texture_descriptor->dimension != platform::CompiledShaderTextureDimension::d2 ||
                texture->texture_descriptor->array || texture->texture_descriptor->multisample ||
                texture->texture_descriptor->comparison ||
                texture->texture_descriptor->component_format == platform::TextureComponentFormat::depth ||
                !texture->texture_descriptor->normalized_coordinates ||
                texture->texture_descriptor->scalar != platform::CompiledShaderTextureScalar::f32)
                return fail(ReplayErrorCode::unsupported, "custom border reflection is not a coverage-v1 pair");
            const auto handle = read_handle(*sampler, 0);
            if (failed) return false;
            const auto id = static_cast<std::int32_t>(handle_sampler_id(handle));
            if (!id) continue;
            borders[i].sampler = backend.resources().coverage_sampler_for_id(id, registrations);
            if (!borders[i].sampler.sampler) {
                const auto detail = backend.resources().take_sampler_failure();
                return fail(ReplayErrorCode::unsupported_resource,
                    (detail.empty() ? "custom border sampler state is unavailable" : detail) +
                    " descriptor=" + std::to_string(id) + " stage=" + std::to_string(stage_index) +
                    " shader=" + stage_label(&shader));
            }
        }
    }

    // Common layouts fit the fixed reflected slot table. Larger admitted
    // layouts retain the dynamic fallback without changing their ABI.
    std::array<std::uint64_t, 64> argument_slots;
    std::vector<std::uint64_t> large_argument_slots;
    std::span<std::uint64_t> argument_members;
    if (shader.argument_buffer) {
        const auto& layout = *shader.argument_buffer;
        if (layout.abi_version != 1 || !layout.byte_extent || (layout.byte_extent & 7u) ||
            layout.buffer_index >= 31)
            return fail(ReplayErrorCode::unsupported,
                        "resolved shader argument-buffer layout is invalid");
        const auto count = layout.byte_extent / sizeof(std::uint64_t);
        if (count <= argument_slots.size()) {
            std::fill_n(argument_slots.begin(), count, std::uint64_t{0});
            argument_members = {argument_slots.data(), count};
        }
        else {
            large_argument_slots.resize(count, 0);
            argument_members = large_argument_slots;
        }
    }
    bool resource_changed = true;
    const auto set_resource = [&](const platform::CompiledShaderResource& resource,
                                  std::uint32_t element, MTLResourceID id) -> bool {
        const auto& location = resource.location;
        resource_changed = true;
        if (location.kind == platform::CompiledShaderResourceLocationKind::Direct) {
            const auto index = location.first + element;
            if (resource.kind == platform::CompiledShaderResourceKind::Sampler) {
                if (index >= 16)
                    return fail(ReplayErrorCode::unsupported,
                                "direct sampler binding exceeds the Metal limit");
                write_sampler_slot(table, stage_index, index, id._impl);
            } else {
                if (index >= 128)
                    return fail(ReplayErrorCode::unsupported,
                                "direct texture binding exceeds the Metal limit");
                resource_changed = write_texture_slot(table, stage_index, index, id._impl);
            }
            return true;
        }
        const auto offset = std::uint64_t(location.byte_offset) +
                            std::uint64_t(element) * location.stride;
        if (!shader.argument_buffer || (offset & 7u) ||
            offset >= shader.argument_buffer->byte_extent)
            return fail(ReplayErrorCode::unsupported,
                        "argument-buffer resource location is invalid");
        argument_members[offset / 8] = id._impl;
        return true;
    };

    if (!shader.resources.empty()) {
        for (const auto& resource : shader.resources) {
            if (!resource.array_length || resource.location.count != resource.array_length)
                return fail(ReplayErrorCode::unsupported,
                            "resolved shader resource array contract is invalid");
            if (resource.kind == platform::CompiledShaderResourceKind::TexturePool ||
                resource.kind == platform::CompiledShaderResourceKind::SamplerPool) {
                if (!bind_pool_resource(stage_index, resource, table,
                                        !compute && stage_index == NVN_SHADER_STAGE_FRAGMENT))
                    return false;
                continue;
            }
            for (std::uint32_t element = 0; element != resource.array_length; ++element) {
                const auto handle = read_handle(resource, element);
                if (failed) return false;
                if (resource.kind == platform::CompiledShaderResourceKind::Sampler) {
                    id<MTLSamplerState> sampler = nil;
                    bool covered = false;
                    for (std::size_t i = 0; i != borders.size(); ++i)
                        if (shader.custom_border->pairs[i].sampler_resource == resource.resource_id) {
                            sampler = borders[i].sampler.sampler; covered = true; break;
                        }
                    if (!covered && handle)
                        sampler = resolve_sampler(static_cast<std::int32_t>(handle_sampler_id(handle)));
                    if (failed) return false;
                    if (!sampler) sampler = backend.empty_sampler();
                    if (!sampler)
                        return fail(ReplayErrorCode::allocation, "empty sampler creation failed");
                    if (!set_resource(resource, element, sampler.gpuResourceID)) return false;
                    retain_once(pending, sampler);
                    continue;
                }
                const bool image =
                    resource.kind == platform::CompiledShaderResourceKind::StorageImage;
                id<MTLTexture> texture = nil;
                std::int32_t texture_id = 0;
                constexpr bool direct_substitution = false;
                texture_id = static_cast<std::int32_t>(handle_texture_id(handle));
                if (texture_id) texture = sampled_texture(texture_id);
                const bool empty = !texture_id || (!texture && backend.resources().texture_id_empty(texture_id));
                if (!texture && !image && empty && resource.texture_descriptor) {
                    const auto& descriptor = *resource.texture_descriptor;
                    const auto format = descriptor.comparison || descriptor.component_format == platform::TextureComponentFormat::depth
                        ? MTLPixelFormatDepth32Float : descriptor.scalar == platform::CompiledShaderTextureScalar::u32
                        ? MTLPixelFormatRGBA32Uint : descriptor.scalar == platform::CompiledShaderTextureScalar::s32
                        ? MTLPixelFormatRGBA32Sint : MTLPixelFormatRGBA8Unorm;
                    if (const auto type = declared_texture_type(resource))
                        texture = backend.resources().fitted_texture(0, nil, *type, true, format);
                }
                if (!texture && image && empty) {
                    // A storage image whose handle names no texture writes a
                    // sink, as the guest's driver lets such a slot go
                    // unwritten, instead of dropping the whole dispatch.
                    // Without a reflected descriptor the slot is a 2D float image.
                    const auto type = declared_texture_type(resource).value_or(MTLTextureType2D);
                    const auto scalar = resource.texture_descriptor ? resource.texture_descriptor->scalar
                                                                    : platform::CompiledShaderTextureScalar::f32;
                    id<MTLTexture> sink = image_sink(backend.device(), type, scalar);
                    if (sink) {
                        if (first_image_sink_report((std::uint64_t(stage_index) << 32) | resource.resource_id))
                            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN: storage image slot %u of stage %zu names no texture (handle %#llx); "
                                "it writes a sink\n", unsigned(resource.resource_id + element), stage_index,
                                static_cast<unsigned long long>(handle));
                        ++backend.mutable_statistics().image_sinks;
                        if (!set_resource(resource, element, sink.gpuResourceID)) return false;
                        make_resident(pending, sink);
                        continue;
                    }
                }
                if (!texture)
                    return fail(ReplayErrorCode::missing_resource,
                                image ? "resolved storage image is unavailable"
                                      : "resolved sampled texture is unavailable");
                note_depth_barrier_read(texture);
                note_attachment_sample(texture);
                if (compute) note_compute_texture(texture);
                const auto descriptor_type = texture.textureType;
                // A texture of another type than the shader declares reads
                // undefined values: bind the view in the declared type.
                if (const auto type = declared_texture_type(resource);
                    !image && type && texture.textureType != *type) {
                    // A host-resolved handle program still selects an element
                    // of a typed table, just as a shader-indexed handle array does.
                    const bool typed_handle = resource.texture_descriptor &&
                        resource.texture_descriptor->scalar == platform::CompiledShaderTextureScalar::f32 &&
                        (resource.cbuf_slot == armigo::shader::pipeline::handle_program_slot ||
                         resource.array_length > 1);
                    if (id<MTLTexture> fitted =
                            backend.resources().fitted_texture(texture_id, texture, *type, typed_handle))
                        texture = fitted;
                    else
                        report_type_mismatch(shader, resource, texture_id, texture, *type);
                }
                if (stage_index == NVN_SHADER_STAGE_FRAGMENT &&
                    (!image || (resource.access == platform::CompiledShaderResourceAccess::Read && !resource.atomic))) {
                    std::optional<std::uint32_t> pin;
                    if (!image && texture.mipmapLevelCount > 1 && handle_sampler_id(handle) &&
                        std::ranges::any_of(shader.resources, [&](const auto& sampler) {
                            return sampler.kind == platform::CompiledShaderResourceKind::Sampler &&
                                sampler.cbuf_slot == resource.cbuf_slot &&
                                sampler.handle_byte_offset == resource.handle_byte_offset &&
                                sampler.array_length == resource.array_length;
                        }))
                        pin = backend.resources().sampler_pinned_level(
                            static_cast<std::int32_t>(handle_sampler_id(handle)), registrations);
                    texture = feedback_texture(texture, image, pin);
                    if (failed) return false;
                }
                if (backend.pass_dump_enabled()) {
                    std::uint64_t serial = 0;
                    if (texture_id && !resolved_textures.empty()) {
                        const auto index = static_cast<std::uint32_t>(texture_id) & (resolved_textures.size() - 1);
                        const auto& cached = resolved_textures[index];
                        if (cached.texture_id == texture_id) serial = cached.stamp.serial;
                    }
                    id<MTLTexture> root = texture;
                    while (root.parentTexture) root = root.parentTexture;
                    char detail[448];
                    std::snprintf(detail, sizeof detail,
                        " stage=%zu resource=%u element=%u handle=%#llx direct_substitution=%u image=%u descriptor=%d serial=%llu "
                        "native_id=%#llx root_id=%#llx type=%lu declared_type=%ld bound_type=%lu format=%lu "
                        "size=%lux%lux%lu layers=%lu levels=%lu cbuf=%u+%u",
                        stage_index, resource.resource_id, element, static_cast<unsigned long long>(handle),
                        unsigned(direct_substitution), unsigned(image), texture_id,
                        static_cast<unsigned long long>(serial),
                        static_cast<unsigned long long>(texture.gpuResourceID._impl),
                        static_cast<unsigned long long>(root.gpuResourceID._impl), static_cast<unsigned long>(descriptor_type),
                        declared_texture_type(resource) ? static_cast<long>(*declared_texture_type(resource)) : -1l,
                        static_cast<unsigned long>(texture.textureType), static_cast<unsigned long>(texture.pixelFormat),
                        static_cast<unsigned long>(texture.width), static_cast<unsigned long>(texture.height),
                        static_cast<unsigned long>(texture.depth), static_cast<unsigned long>(texture.arrayLength),
                        static_cast<unsigned long>(texture.mipmapLevelCount), resource.cbuf_slot,
                        resource.handle_byte_offset);
                    stage.dump_reads.push_back("shader=" + stage_label(&shader) + detail);
                    if (!image && backend.pass_dump().inputs())
                        stage.dump_textures.push_back({texture, serial, "r" + std::to_string(stage_index) +
                            "-" + std::to_string(resource.resource_id) + "-" + std::to_string(element)});
                }
                const bool dump_image = compute && image && backend.pass_dump_enabled() &&
                    !backend.pass_dump().metadata_only() &&
                    (resource.atomic || resource.access != platform::CompiledShaderResourceAccess::Read) &&
                    backend.pass_dump_wants(backend.pool_present_count());
                std::uint64_t dump_serial = 0;
                if (dump_image && texture_id && !resolved_textures.empty()) {
                    const auto index = static_cast<std::uint32_t>(texture_id) & (resolved_textures.size() - 1);
                    const auto& cached = resolved_textures[index];
                    if (cached.texture_id == texture_id) dump_serial = cached.stamp.serial;
                }
                if (image && resource.atomic && image_atomics_enabled()) {
                    if (!note_image_atomic(stage_index, resource, element, texture_id, texture, compute))
                        return false;
                } else if (image && resource.access != platform::CompiledShaderResourceAccess::Read &&
                           !note_image_write(texture_id, texture, compute))
                    return false;
                if (image && !resource.atomic && raw_image_view_required(texture)) {
                    texture = backend.resources().raw_image_view(texture_id, texture);
                    if (!texture)
                        return fail(ReplayErrorCode::unsupported_resource,
                                    "storage image format or channel mapping has no raw view");
                }
                if (dump_image && texture)
                    compute_dump_images.push_back(
                        {resource.resource_id + element, dump_serial, texture});
                if (failed) return false;
                id<MTLTexture> border_original = texture;
                for (std::size_t i = 0; i != borders.size(); ++i) {
                    if (shader.custom_border->pairs[i].texture_resource != resource.resource_id) continue;
                    borders[i].bound = true;
                    if (!borders[i].sampler.custom) continue;
                    if (!texture_id || !texture || texture.textureType != MTLTextureType2D || texture.sampleCount != 1)
                        return fail(ReplayErrorCode::unsupported_resource,
                                    "custom border coverage needs its admitted 2D sampled view shader=" + stage_label(&shader));
                    const auto swizzle = border_original.swizzle;
                    const auto channel = [](MTLTextureSwizzle value) -> std::uint32_t {
                        switch (value) {
                        case MTLTextureSwizzleRed: return 0; case MTLTextureSwizzleGreen: return 1;
                        case MTLTextureSwizzleBlue: return 2; case MTLTextureSwizzleAlpha: return 3;
                        case MTLTextureSwizzleZero: return 4; case MTLTextureSwizzleOne: return 5;
                        }
                        return UINT32_MAX;
                    };
                    auto& record = border_records[i];
                    record.color = borders[i].sampler.color;
                    // Sampling preserves native defaults for absent channels,
                    // including alpha one. Border reconstruction only changes
                    // stored components, before applying the original swizzle.
                    const unsigned components = [&] {
                        switch (border_original.pixelFormat) {
                        case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Unorm_sRGB:
                        case MTLPixelFormatR8Snorm: case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint:
                        case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm:
                        case MTLPixelFormatR16Uint: case MTLPixelFormatR16Sint: case MTLPixelFormatR16Float:
                        case MTLPixelFormatR32Uint: case MTLPixelFormatR32Sint: case MTLPixelFormatR32Float:
                        case MTLPixelFormatBC4_RUnorm: case MTLPixelFormatBC4_RSnorm:
                            return 1u;
                        case MTLPixelFormatRG8Unorm: case MTLPixelFormatRG8Unorm_sRGB:
                        case MTLPixelFormatRG8Snorm: case MTLPixelFormatRG8Uint: case MTLPixelFormatRG8Sint:
                        case MTLPixelFormatRG16Unorm: case MTLPixelFormatRG16Snorm:
                        case MTLPixelFormatRG16Uint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRG16Float:
                        case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Sint: case MTLPixelFormatRG32Float:
                        case MTLPixelFormatBC5_RGUnorm: case MTLPixelFormatBC5_RGSnorm:
                            return 2u;
                        case MTLPixelFormatB5G6R5Unorm: case MTLPixelFormatRG11B10Float:
                        case MTLPixelFormatRGB9E5Float: case MTLPixelFormatBC6H_RGBFloat:
                        case MTLPixelFormatBC6H_RGBUfloat: case MTLPixelFormatETC2_RGB8:
                        case MTLPixelFormatETC2_RGB8_sRGB: case MTLPixelFormatBGR10_XR:
                        case MTLPixelFormatBGR10_XR_sRGB:
                            return 3u;
                        default: return 4u;
                        }
                    }();
                    for (unsigned c = components; c != 4; ++c) record.color[c] = 0;
                    record.swizzle = {channel(swizzle.red), channel(swizzle.green), channel(swizzle.blue), channel(swizzle.alpha)};
                    if (std::ranges::any_of(record.swizzle, [](auto value) { return value >= 6; }))
                        return fail(ReplayErrorCode::unsupported_resource, "custom border channel mapping is invalid");
                    id<MTLTexture> coverage = backend.border_coverage(border_original.width, border_original.height, border_original.mipmapLevelCount);
                    texture = backend.border_identity(border_original);
                    if (!coverage || !texture)
                        return fail(ReplayErrorCode::allocation, "custom border coverage texture allocation failed");
                    record.coverage = coverage.gpuResourceID._impl;
                    record.enabled = 1;
                    make_resident(pending, coverage);
                    make_resident(pending, texture);
                }
                if (!set_resource(resource, element, texture.gpuResourceID)) return false;
                if (resource_changed) make_resident(pending, texture);
            }
        }
        for (std::size_t i = 0; i != borders.size(); ++i)
            if (borders[i].sampler.custom && !borders[i].bound)
                return fail(ReplayErrorCode::unsupported, "custom border reflected texture was not bound");
    } else {
        for (const auto& binding : shader.textures) {
            if (binding.texture_index >= texture_slots)
                return fail(ReplayErrorCode::unsupported,
                            "legacy texture binding exceeds the replay limit");
            id<MTLTexture> texture = bound_texture(binding.texture_index, false);
            if (!texture)
                return fail(ReplayErrorCode::missing_resource,
                            "legacy sampled texture is unavailable");
            if (stage_index == NVN_SHADER_STAGE_FRAGMENT) {
                std::optional<std::uint32_t> pin;
                if (texture.mipmapLevelCount > 1 && binding.sampler_index == binding.texture_index &&
                    binding.sampler_index < sampler_slots && !state.samplers_at(stage_index)[binding.sampler_index]) {
                    const auto handle = state.textures_at(stage_index)[binding.texture_index];
                    if (handle_sampler_id(handle)) pin = backend.resources().sampler_pinned_level(
                        static_cast<std::int32_t>(handle_sampler_id(handle)), registrations);
                }
                texture = feedback_texture(texture, false, pin);
                if (failed) return false;
            }
            note_depth_barrier_read(texture);
            note_attachment_sample(texture);
            if (compute) note_compute_texture(texture);
            table_texture(table, binding.texture_index, texture.gpuResourceID._impl);
            make_resident(pending, texture);
            if (binding.sampler_index != shader::graphics_msl::TextureBinding::no_sampler) {
                if (binding.sampler_index >= 16)
                    return fail(ReplayErrorCode::unsupported, "legacy sampler index exceeds the Metal limit");
                id<MTLSamplerState> sampler = bound_sampler(binding.sampler_index, binding.texture_index);
                if (failed) return false;
                if (!sampler) sampler = backend.empty_sampler();
                if (!sampler)
                    return fail(ReplayErrorCode::allocation, "empty sampler creation failed");
                table_sampler(table, binding.sampler_index, sampler.gpuResourceID._impl);
                [pending.retained addObject:sampler];
            }
        }
    }
    if (shader.argument_buffer) {
        const auto arguments = upload(argument_members.size() * sizeof(std::uint64_t),
                                      argument_members.data());
        if (!arguments) return false;
        write_buffer_slot(table, stage_index, shader.argument_buffer->buffer_index,
                          arguments.gpu);
    }
    static_cast<void>(vertex);
    finish_uniforms();
    return true;
}

// The fetch table of a vertex stage that reads its own inputs: the streams
// and attributes of the bound vertex state, as the vertex descriptor of a
// pipeline built for that state describes them. An input whose attribute is
// disabled reads zeros, as through the descriptor's zero stream.
bool Encoder::bind_vertex_fetch(const platform::CompiledGraphicsStage& vertex) {
    shader::VertexFetchTable table{};
    std::array<bool, buffer_slots> used{};
    // The alignment each used stream's address and stride need: the fetch
    // loads a component with one load of the component's size.
    std::array<std::uint32_t, buffer_slots> alignment{};
    for (const auto& input : vertex.vertex_class_inputs) {
        if (input.location >= shader::vertex_fetch_attributes)
            return fail(ReplayErrorCode::unsupported, "vertex input exceeds the attribute limit");
        const auto& attribute = state.attributes[input.location];
        if (attribute.format == NVN_FORMAT_NONE) continue;
        const auto format = vertex_fetch_format(attribute.format);
        if (!format)
            return fail(ReplayErrorCode::unsupported,
                        "vertex attribute format " + std::to_string(attribute.format) + " is unsupported");
        if (attribute.stream >= buffer_slots)
            return fail(ReplayErrorCode::unsupported, "vertex attribute stream index exceeds the replay limit");
        const auto kind = static_cast<std::uint32_t>(format->kind);
        const std::uint32_t size = kind <= 4 || format->kind == shader::VertexFetchKind::unorm8_bgra ? 1
            : kind <= 9 ? 2 : 4;
        if (attribute.offset % size)
            return fail(ReplayErrorCode::unsupported, "vertex attribute offset " + std::to_string(attribute.offset) +
                        " is not aligned to its " + std::to_string(size) + "-byte components");
        table.attribute[input.location] = shader::vertex_fetch_attribute(
            format->kind, format->components, attribute.stream, attribute.offset, format->scaled);
        used[attribute.stream] = true;
        alignment[attribute.stream] = std::max(alignment[attribute.stream], size);
    }
    for (std::size_t index = 0; index != buffer_slots; ++index) {
        if (!used[index]) continue;
        const auto& range = state.vertex_buffers[index];
        const auto span = buffer_address(range.address, range.size);
        if (!range.address || !range.size || !span.buffer)
            return fail(ReplayErrorCode::missing_resource, "required vertex buffer range is unavailable");
        const auto& stream = state.streams[index];
        if (span.gpu % alignment[index] || stream.stride % alignment[index])
            return fail(ReplayErrorCode::unsupported, "vertex stream address or stride is not aligned to its "
                        "attributes' components");
        table.stream[index] = span.gpu;
        // A stride of zero reads the same bytes for every index.
        table.stride[index] = stream.stride;
        table.step[index] = stream.divisor ? stream.divisor : shader::vertex_fetch_step_vertex;
        table.limit[index] = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(range.size, std::numeric_limits<std::uint32_t>::max()));
        make_resident(pending, span.buffer);
    }
    const auto uploaded = upload(sizeof table, &table);
    if (!uploaded) return false;
    write_buffer_slot(vertex_table, NVN_SHADER_STAGE_VERTEX, *vertex.vertex_fetch_buffer_index, uploaded.gpu);
    return true;
}

// One draw-parameter record serves both stages: the vertex stage reads the
// original first instance (word 0), a fragment stage the height of the pass's
// attachments (word 1), each at its own reflected index.
bool Encoder::bind_draw_support(std::uint32_t base_instance) {
    const auto index_of = [&](std::size_t stage) -> std::optional<std::uint8_t> {
        const auto& program = state.stage_programs[stage];
        if (!program || !program->stages[stage]) return std::nullopt;
        return program->stages[stage]->draw_support_buffer_index;
    };
    const auto vertex = index_of(NVN_SHADER_STAGE_VERTEX), fragment = index_of(NVN_SHADER_STAGE_FRAGMENT);
    if (!vertex && !fragment) return true;
    if (fragment && !attachment_height)
        return fail(ReplayErrorCode::unsupported, "render-target height is unknown outside a render pass");
    const std::array<std::uint32_t, 2> record{base_instance, static_cast<std::uint32_t>(attachment_height)};
    const auto support = upload(sizeof record, record.data());
    if (!support) return false;
    if (vertex) write_buffer_slot(vertex_table, NVN_SHADER_STAGE_VERTEX, *vertex, support.gpu);
    if (fragment) write_buffer_slot(fragment_table, NVN_SHADER_STAGE_FRAGMENT, *fragment, support.gpu);
    return true;
}

void Encoder::note_draw_writes() {
    const bool vertex_writes = bindings[NVN_SHADER_STAGE_VERTEX].writes_buffers;
    const bool fragment_writes = state.stage_programs[NVN_SHADER_STAGE_FRAGMENT] && !state.rasterizer_discard &&
        bindings[NVN_SHADER_STAGE_FRAGMENT].writes_buffers;
    pass_storage_writes |= vertex_writes || fragment_writes;
    if (pass_depth_only) {
        pass_depth_only = state.targets.depth && state.depth_stencil.depth_test &&
            state.depth_stencil.depth_write && !state.depth_stencil.stencil_test;
        for (std::int32_t index = 0; index < std::clamp(state.targets.count, 0, 8); ++index)
            if (state.targets.color[index] && ((state.channel_mask.mask >> (index * 4)) & 15))
                pass_depth_only = false;
        for (const auto stage : {NVN_SHADER_STAGE_VERTEX, NVN_SHADER_STAGE_FRAGMENT}) {
            const auto& program = state.stage_programs[stage];
            if (program && program->stages[stage] && !program->stages[stage]->storage_buffers.empty())
                pass_depth_only = false;
        }
    }
    if (!dirty.attachments_written) {
        for (std::int32_t index = 0; index < std::clamp(state.targets.count, 0, 8); ++index)
            if (state.targets.color[index])
                note_texture_write(*state.targets.color[index]);
        if (state.targets.depth) note_texture_write(*state.targets.depth);
        dirty.attachments_written = true;
    }
    note_shader_writes(NVN_SHADER_STAGE_VERTEX);
    if (state.stage_programs[NVN_SHADER_STAGE_FRAGMENT])
        note_shader_writes(NVN_SHADER_STAGE_FRAGMENT);
    note_writes();
    buffer_write_order.draw(vertex_writes, fragment_writes);
}

// Pipeline look-ahead. The first pipeline a submission has to compile is a
// sign that the rest of its draws may need new ones too (a scene entered, a
// pass type used for the first time), and the draws are reached one at a time
// with each compile waited for. This walks the records after the current one
// over a copy of the state, applying only what a pipeline key depends on, and
// starts the pipelines of the draws that would miss (up to 160), so they
// compile in parallel while the replay walks on. A draw still waits only for
// its own pipeline. The keys are the ones the draws will build, except where
// an attachment's storage does not exist yet (no format to key on); those
// draws are left to compile when reached.
void Encoder::predict_pipelines() {
    constexpr std::size_t limit = 160;
    struct Candidate {
        DrawPipelineKey key;
        std::shared_ptr<const platform::CompiledGraphicsStage> vertex, fragment;
    };
    struct Lookahead {
        Encoder& encoder;
        ReplayState shadow;
        std::size_t index = 0;
        std::array<MTLPixelFormat, 8> formats{};
        NSUInteger samples = 0;
        bool formats_known = true;
        std::vector<Candidate> found;
        std::size_t draws = 0, unknown = 0;

        void attach() {
            formats = {};
            samples = 0;
            formats_known = true;
            auto& resources = encoder.backend.resources();
            for (std::int32_t i = 0; i < std::clamp(shadow.targets.count, 0, 8); ++i) {
                const auto* texture = shadow.targets.color[i];
                if (!texture) continue;
                const auto peeked = resources.peek_attachment(
                    *texture, shadow.targets.has_color_view[i] ? &shadow.targets.color_views[i] : nullptr);
                if (peeked.format == MTLPixelFormatInvalid) { formats_known = false; return; }
                formats[i] = peeked.format;
                samples = peeked.samples;
            }
            if (!samples && shadow.targets.depth) {
                const auto peeked = resources.peek_attachment(
                    *shadow.targets.depth, shadow.targets.has_depth_view ? &shadow.targets.depth_view : nullptr);
                if (peeked.format == MTLPixelFormatInvalid) { formats_known = false; return; }
                samples = peeked.samples;
            }
        }

        void draw(std::int32_t primitive) {
            ++draws;
            if (!formats_known) { ++unknown; return; }
            if (found.size() >= limit || !shadow.render_enabled) return;
            Candidate candidate{};
            if (fill_pipeline_key(shadow, primitive, formats, samples, candidate.key)) return;
            if (encoder.backend.has_pipeline(candidate.key)) return;
            for (const auto& other : found)
                if (!std::memcmp(&other.key, &candidate.key, sizeof candidate.key)) return;
            const auto& vertex = shadow.stage_programs[NVN_SHADER_STAGE_VERTEX];
            const auto& fragment = shadow.stage_programs[NVN_SHADER_STAGE_FRAGMENT];
            candidate.vertex = vertex->stages[NVN_SHADER_STAGE_VERTEX];
            if (fragment) candidate.fragment = fragment->stages[NVN_SHADER_STAGE_FRAGMENT];
            found.push_back(std::move(candidate));
        }

        bool visit(const CommandRecordView& record) {
            if (++index <= encoder.records_seen) return true;
            if (found.size() >= limit) return false;
            switch (record.op) {
                case Op::draw:
                case Op::draw_indexed:
                case Op::draw_indirect:
                case Op::draw_indexed_indirect:
                case Op::multi_draw_indirect:
                case Op::multi_draw_indexed_indirect: {
                    // Every draw record opens with its primitive.
                    std::int32_t primitive = 0;
                    if (record.payload_size >= sizeof primitive)
                        std::memcpy(&primitive, record.payload, sizeof primitive);
                    draw(primitive);
                    break;
                }
                case Op::set_render_targets: {
                    shadow.targets = payload<SetRenderTargets>(record);
                    if (shadow.targets.count < 0 || shadow.targets.count > 8) return false;
                    shadow.target_owners = {};
                    for (std::int32_t i = 0; i != shadow.targets.count; ++i) {
                        if (!shadow.targets.color[i]) continue;
                        shadow.target_owners[i] =
                            std::make_shared<const TextureState>(*shadow.targets.color[i]);
                        shadow.targets.color[i] = shadow.target_owners[i].get();
                    }
                    shadow.depth_owner.reset();
                    if (shadow.targets.depth) {
                        shadow.depth_owner = std::make_shared<const TextureState>(*shadow.targets.depth);
                        shadow.targets.depth = shadow.depth_owner.get();
                    }
                    attach();
                    break;
                }
                case Op::bind_program: {
                    const auto command = payload<BindProgram>(record);
                    constexpr std::uint32_t all_stage_bits = (1u << logical_binding_stage_count) - 1;
                    if (!command.stages || command.stages < 0 ||
                        (std::uint32_t(command.stages) & ~all_stage_bits)) break;
                    // An unresolved program may publish stages outside its
                    // requested mask. Stop rather than guess its effects.
                    std::shared_ptr<const ResolvedProgram> program;
                    bool translating = false;
                    if (command.program) program = retain_ready_program(*command.program, translating);
                    if (command.program && !program) return false;
                    shadow.bind(command, program);
                    break;
                }
                case Op::bind_state: {
                    const auto command = payload<BindState>(record);
                    switch (command.kind) {
                        case 0: {
                            BlendStateBits value{};
                            std::memcpy(&value, command.bytes, sizeof value);
                            if (value.target < shadow.blend.size()) shadow.blend[value.target] = value;
                            break;
                        }
                        case 1: std::memcpy(&shadow.color, command.bytes, sizeof shadow.color); break;
                        case 2: std::memcpy(&shadow.channel_mask, command.bytes, sizeof shadow.channel_mask); break;
                        case 3: {
                            MultisampleStateBits value{};
                            std::memcpy(&value, command.bytes, sizeof value);
                            shadow.multisample = value;
                            break;
                        }
                        default: break;
                    }
                    break;
                }
                case Op::bind_vertex_attributes: {
                    const auto command = payload<BindVertexAttributes>(record);
                    if (command.first < 0 || command.count < 0 || command.first > 16 ||
                        command.count > 16 - command.first) break;
                    std::memcpy(shadow.attributes.data() + command.first, command.states,
                                command.count * sizeof(VertexAttributeStateBits));
                    break;
                }
                case Op::bind_vertex_streams: {
                    const auto command = payload<BindVertexStreams>(record);
                    if (command.first < 0 || command.count < 0 || command.first > 16 ||
                        command.count > 16 - command.first) break;
                    std::memcpy(shadow.streams.data() + command.first, command.states,
                                command.count * sizeof(VertexStreamStateBits));
                    break;
                }
                case Op::geometry_processing_mode:
                    shadow.geometry_mode = payload<ScalarInt>(record).value;
                    shadow.select();
                    break;
                case Op::rasterizer_discard:
                    shadow.rasterizer_discard = payload<ScalarInt>(record).value != 0;
                    break;
                default: break;
            }
            return true;
        }
    };
    const auto begin = timeline_now_ns();
    Lookahead lookahead{*this, state};
    lookahead.attach();
    struct RangeWalk {
        Lookahead* lookahead;
        std::size_t index, first, end;
    } range{&lookahead, 0, pending.first_record, pending.end_record};
    const auto status = walk_commands(*pending.captured,
        [](const CommandRecordView& record, void* context) {
            auto& range = *static_cast<RangeWalk*>(context);
            const auto at = range.index++;
            return at < range.first || at >= range.end || range.lookahead->visit(record);
        }, &range);
    (void)status;
    std::string detail;
    for (auto& candidate : lookahead.found) {
        auto* created = backend.pipeline(candidate.key, std::move(candidate.vertex),
                                         std::move(candidate.fragment), detail);
        if (created && backend.created_pipeline()) {
            created->predicted = true;
            ++backend.mutable_statistics().predicted_pipelines;
        }
    }
    timeline_event("pso_predict", lookahead.found.size(), timeline_now_ns() - begin);
    timeline_event("pso_predict_draws", lookahead.draws, lookahead.unknown);
}

bool Encoder::draw(const Draw& command) {
    if (!resolve_carried_predicate()) return false;
    if (predicate_open() && predicate.used == predicate_capacity && !resolve_predicate()) return false;
    if (!state.render_enabled) return true;
    if (expands_to_triangles(command.primitive)) {
        // Quads and fans draw as indexed triangles over the same vertices.
        if (command.first < 0 || command.count < 0 || command.instances < 0 || command.base_instance < 0)
            return fail(ReplayErrorCode::invalid_command, "draw contains a negative range");
        if (command.count < 3 || !command.instances) return true;
        const bool fan = command.primitive == NVN_DRAW_PRIMITIVE_TRIANGLE_FAN;
        const auto capacity = expanded_capacity(fan, static_cast<std::size_t>(command.count));
        if (!capacity) return true;
        if (capacity > max_expanded_indices)
            return fail(ReplayErrorCode::unsupported, "expanded draw exceeds Metal's 32-bit index count");
        if (!prepare_graphics_uniform_writes() || !order_buffer_writes()) return false;
        // The triangles are written straight into this submission's upload arena.
        const auto converted = backend.allocate_upload(pending, capacity * sizeof(std::uint32_t));
        if (!converted) return fail(ReplayErrorCode::allocation, "replay upload arena allocation failed");
        const auto used = expand_triangles(reinterpret_cast<std::uint32_t*>(converted.cpu), nullptr, 4,
                                           static_cast<std::uint32_t>(command.first),
                                           static_cast<std::size_t>(command.count), fan, false, 0);
        if (used == expansion_collision)
            return fail(ReplayErrorCode::unsupported, "index value collides with the native restart sentinel");
        note_base_instance(static_cast<std::uint32_t>(command.base_instance));
        if (!open_render(command.primitive)) return false;
        if (!bind_draw_support(static_cast<std::uint32_t>(command.base_instance))) return false;
        encode_draw_indexed(MTLPrimitiveTypeTriangle, static_cast<std::uint32_t>(used),
                            MTLIndexTypeUInt32, converted.gpu, used * sizeof(std::uint32_t),
                            command.instances, 0, command.base_instance);
        note_draw_writes();
        ++backend.mutable_statistics().draws;
        ++pending.counters.draws;
        return true;
    }
    const auto primitive = primitive_type(command.primitive);
    if (!primitive)
        return fail(ReplayErrorCode::unsupported, "draw primitive is unsupported by Metal replay");
    if (command.first < 0 || command.count < 0 || command.instances < 0 ||
        command.base_instance < 0)
        return fail(ReplayErrorCode::invalid_command, "draw contains a negative range");
    if (!command.count || !command.instances) return true;
    if ((*primitive == MTLPrimitiveTypeLine || *primitive == MTLPrimitiveTypeLineStrip) &&
        state.line_width != 1.0f)
        return fail(ReplayErrorCode::unsupported,
                    "Metal supports only unit-width NVN line draws");
    if (*primitive == MTLPrimitiveTypePoint && state.point_size != 1.0f &&
        (!state.stage_programs[NVN_SHADER_STAGE_VERTEX] ||
         !state.stage_programs[NVN_SHADER_STAGE_VERTEX]->stages[NVN_SHADER_STAGE_VERTEX] ||
         state.stage_programs[NVN_SHADER_STAGE_VERTEX]
             ->stages[NVN_SHADER_STAGE_VERTEX]->fixed_point_size))
        return fail(ReplayErrorCode::unsupported,
                    "point size requires a compatible vertex shader artifact");
    if (!prepare_graphics_uniform_writes() || !order_buffer_writes()) return false;
    note_base_instance(static_cast<std::uint32_t>(command.base_instance));
    if (!open_render(command.primitive)) return false;
    if (!bind_draw_support(static_cast<std::uint32_t>(command.base_instance))) return false;
    encode_draw(*primitive, command.first, command.count, command.instances, command.base_instance);
    note_draw_writes();
    ++backend.mutable_statistics().draws;
    ++pending.counters.draws;
    return true;
}

// A mesh draw dispatches `count` task threadgroups (mesh threadgroups when
// the program has no task stage) with the task and mesh stages' threadgroup
// sizes. Both stages read the first workgroup as word 0 of their
// draw-parameter record. The indirect form's record {count, first} is written
// by the GPU earlier in the submission: its count is copied into a {count, 1,
// 1} threadgroup count before the pass, and the stages read first from the
// record itself (take_mesh_records). Mesh draws are not predicated on the
// GPU: an open conditional block is resolved first.
bool Encoder::draw_mesh(std::uint32_t first, std::uint32_t count, std::uint64_t indirect) {
    const SiteScope site_scope(*this, TransferSite::mesh_indirect);
    if (!resolve_carried_predicate()) return false;
    if (predicate_open() && !resolve_predicate()) return false;
    if (!state.render_enabled) return true;
    if (!indirect && !count) return true;
    std::uint64_t groups = 0;
    BufferAddress record{};
    if (indirect) {
        record = buffer_address(indirect, 8);
        if (!record.buffer || (record.gpu & 3))
            return fail(ReplayErrorCode::missing_resource, "mesh draw's indirect record is unavailable or misaligned");
        auto found = records_seen < mesh_groups_end ? mesh_groups.find(indirect) : mesh_groups.end();
        if (found == mesh_groups.end()) {
            // A pending uniform update of the record lands before it is read.
            if (!prepare_uniform_access(indirect, 8) || !take_mesh_records(indirect)) return false;
            found = mesh_groups.find(indirect);
            if (found == mesh_groups.end())
                return fail(ReplayErrorCode::missing_resource, "mesh draw's indirect record was not copied");
        }
        groups = found->second;
    }
    if (!prepare_graphics_uniform_writes() || !order_buffer_writes()) return false;
    note_base_instance(0);
    mesh_draw = true;
    const bool opened = open_render(mesh_draw_primitive);
    mesh_draw = false;
    if (!opened) return false;
    if (!bind_draw_support(0)) return false;
    // The first workgroup: the record's second word, or a word of its own.
    std::uint64_t first_address = indirect ? record.gpu + sizeof(std::uint32_t) : 0;
    if (!indirect) {
        const auto slice = upload(sizeof first, &first);
        if (!slice) return false;
        first_address = slice.gpu;
    }
    const platform::CompiledGraphicsStage* task = nullptr;
    const platform::CompiledGraphicsStage* mesh = nullptr;
    for (const auto& [slot, table, shader] :
         {std::tuple{std::size_t(NVN_SHADER_STAGE_TESS_CONTROL), object_table, &task},
          std::tuple{std::size_t(NVN_SHADER_STAGE_TESS_EVALUATION), mesh_table, &mesh}}) {
        const auto& program = state.stage_programs[slot];
        *shader = program ? program->stages[slot].get() : nullptr;
        if (*shader && (*shader)->draw_support_buffer_index)
            write_buffer_slot(table, slot, *(*shader)->draw_support_buffer_index, first_address);
    }
    // A task stage's status words follow its draw parameters.
    if (task && task->draw_support_buffer_index) {
        if (!pending.task_status) {
            const auto status = upload(2 * sizeof(std::uint32_t));
            if (!status) return false;
            pending.task_status = reinterpret_cast<const std::uint32_t*>(status.cpu);
            pending.task_status_gpu = status.gpu;
        }
        write_buffer_slot(object_table, NVN_SHADER_STAGE_TESS_CONTROL, *task->draw_support_buffer_index + 1,
                          pending.task_status_gpu);
    }
    const auto size_of = [](const std::optional<platform::CompiledMeshLaunch>& launch) {
        return launch ? launch->local_size : std::array<std::uint32_t, 3>{1, 1, 1};
    };
    DrawMeshRecord draw{};
    draw.groups = count;
    const auto object = size_of(task ? task->task : std::nullopt);
    const auto threads = size_of(mesh->mesh);
    std::copy(object.begin(), object.end(), draw.object);
    std::copy(threads.begin(), threads.end(), draw.mesh);
    draw.indirect = groups;
    if (queue) {
        queue->put(EncodeOp::mesh_tables, 0,
                   MeshTablesRecord{(__bridge void*)object_table, (__bridge void*)mesh_table});
        queue->put(EncodeOp::draw_mesh, 0, draw);
        queue->publish();
    } else {
        [render setArgumentTable:object_table atStages:MTLRenderStageObject];
        [render setArgumentTable:mesh_table atStages:MTLRenderStageMesh];
        const auto object_size = MTLSizeMake(draw.object[0], draw.object[1], draw.object[2]);
        const auto mesh_size = MTLSizeMake(draw.mesh[0], draw.mesh[1], draw.mesh[2]);
        if (draw.indirect)
            [render drawMeshThreadgroupsWithIndirectBuffer:draw.indirect threadsPerObjectThreadgroup:object_size
                                 threadsPerMeshThreadgroup:mesh_size];
        else
            [render drawMeshThreadgroups:MTLSizeMake(count, 1, 1) threadsPerObjectThreadgroup:object_size
                 threadsPerMeshThreadgroup:mesh_size];
    }
    note_draw_writes();
    // Task and mesh stages write storage like a vertex stage does.
    for (const auto& [slot, shader] : {std::pair{std::size_t(NVN_SHADER_STAGE_TESS_CONTROL), task},
                                        std::pair{std::size_t(NVN_SHADER_STAGE_TESS_EVALUATION), mesh}}) {
        if (shader) note_shader_writes(slot);
        if (!shader || shader->storage_buffers.empty()) continue;
        pass_depth_only = false;
        pass_storage_writes |= bindings[slot].writes_buffers;
    }
    note_writes();
    ++backend.mutable_statistics().draws;
    ++pending.counters.draws;
    return true;
}

// A transfer cannot run inside a render pass, so indirect mesh records are
// copied ahead: the first indirect mesh draw with no copy copies its record
// and those of the indirect mesh draws after it in replay order, up to the
// next record that may write one (a dispatch, buffer copy or fill, texture to
// buffer copy, report, event, sync wait, transform feedback, a uniform update
// over a taken record, or a draw whose program stores to memory). Only a draw
// met inside an open render pass splits it. A later record is not taken where
// a write may reach it without such a record: a uniform update it overlaps
// (pending, or met earlier in the scan) lands at a later flush, and texture
// storage over it is written by clears, copies and attachments.
// ARMIGO_NVN_MESH_RECORD_BATCH=0 copies each draw's record alone.
bool Encoder::take_mesh_records(std::uint64_t indirect) {
    static const std::size_t limit = [] {
        const char* value = std::getenv("ARMIGO_NVN_MESH_RECORD_BATCH");
        return value && *value == '0' ? std::size_t(1) : std::size_t(1024);
    }();
    constexpr std::uint32_t render_stages = ((1u << stage_count) - 1) & ~(1u << NVN_SHADER_STAGE_COMPUTE);
    struct Taken {
        std::uint64_t address;
        std::size_t index;
    };
    struct Scan {
        ReplayPrograms programs;
        const Encoder& encoder;
        std::vector<Taken> records;
        // Uniform updates met since the first record.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> updated;
        std::uint64_t low = ~0ull, high = 0;
        std::uint32_t writes = 0;
        std::size_t index = 0, end = std::numeric_limits<std::size_t>::max();

        bool take(std::uint64_t address, std::size_t at) {
            if (std::ranges::any_of(records, [&](const Taken& taken) { return taken.address == address; }))
                return true;
            for (const auto& [begin, last] : updated)
                if (begin < address + 8 && address < last) return false;
            records.push_back({address, at});
            low = std::min(low, address);
            high = std::max(high, address + 8);
            return true;
        }
        bool stop(std::size_t at) {
            end = at;
            return false;
        }
        bool visit(const CommandRecordView& record) {
            if (++index <= encoder.records_seen) return true;
            switch (record.op) {
                case Op::draw:
                case Op::draw_indexed:
                case Op::draw_indirect:
                case Op::draw_indexed_indirect:
                case Op::multi_draw_indirect:
                case Op::multi_draw_indexed_indirect:
                case Op::draw_texture:
                case Op::draw_mesh_tasks:
                    return !writes || stop(index + 1);
                case Op::draw_mesh_tasks_indirect:
                    if (records.size() >= limit || !take(payload<DrawMeshTasksIndirect>(record).address, index))
                        return stop(index);
                    return !writes || stop(index + 1);
                case Op::bind_program: {
                    const auto command = payload<BindProgram>(record);
                    if (!command.stages) return true;
                    std::shared_ptr<const ResolvedProgram> program;
                    bool translating = false;
                    if (command.program) program = retain_ready_program(*command.program, translating);
                    if (command.program && !program) return stop(index);
                    programs.bind(command, program);
                    writes = (programs.write_stages | programs.failed_stages) & render_stages;
                    return true;
                }
                case Op::geometry_processing_mode:
                    programs.geometry_mode = payload<ScalarInt>(record).value;
                    programs.select();
                    writes = (programs.write_stages | programs.failed_stages) & render_stages;
                    return true;
                case Op::update_uniforms: {
                    const auto command = payload<UpdateUniforms>(record);
                    const auto last = command.destination + command.size;
                    if (command.destination < high && low < last)
                        for (const auto& taken : records)
                            if (command.destination < taken.address + 8 && taken.address < last) return stop(index);
                    updated.emplace_back(command.destination, last);
                    return true;
                }
                case Op::dispatch:
                case Op::dispatch_indirect:
                case Op::copy_buffer:
                case Op::clear_buffer:
                case Op::copy_texture_to_buffer:
                case Op::report_counter:
                case Op::report_value:
                case Op::reset_counter:
                case Op::signal_event:
                case Op::wait_event:
                case Op::wait_sync:
                case Op::bind_transform_feedback_buffer:
                case Op::transform_feedback:
                    return stop(index);
                default: return true;
            }
        }
    } scan{static_cast<const ReplayPrograms&>(state), *this, {}};
    scan.writes = (state.write_stages | state.failed_stages) & render_stages;
    scan.take(indirect, records_seen);
    if (scan.writes) {
        scan.end = records_seen + 1;
    } else {
        struct RangeWalk {
            Scan* scan;
            std::size_t index, first, end;
        } range{&scan, 0, pending.first_record, pending.end_record};
        const auto status = walk_commands(*pending.captured,
            [](const CommandRecordView& record, void* context) {
                auto& range = *static_cast<RangeWalk*>(context);
                const auto at = range.index++;
                return at < range.first || at >= range.end || range.scan->visit(record);
            }, &range);
        (void)status;
    }
    for (std::size_t i = 1; i < scan.records.size(); ++i) {
        const auto address = scan.records[i].address;
        if (uniform_conflict(address, 8) || backend.resources().overlaps_texture_backing(address, 8) ||
            nvn::window_texture_overlaps(address, 8)) {
            scan.end = scan.records[i].index;
            scan.records.resize(i);
            break;
        }
    }
    if (failed) return false;
    mesh_groups.clear();
    mesh_groups_end = scan.end;
    // Each copy gets {count, 1, 1}; the transfer writes count.
    struct Copy {
        BufferAddress record;
        std::uint64_t address;
    };
    std::vector<Copy> copies;
    copies.reserve(scan.records.size());
    for (const auto& taken : scan.records) {
        const auto record = buffer_address(taken.address, 8);
        if (record.buffer && !(record.gpu & 3)) copies.push_back({record, taken.address});
    }
    if (copies.empty()) return true;
    constexpr std::size_t stride = 4;
    std::vector<std::uint32_t> ones(copies.size() * stride);
    for (std::size_t i = 0; i != copies.size(); ++i) {
        ones[i * stride + 1] = 1;
        ones[i * stride + 2] = 1;
    }
    const auto groups = upload(ones.size() * sizeof(std::uint32_t), ones.data());
    if (!groups) return false;
    make_resident(pending, groups.buffer);
    if (!close_render(PassBreak::transfer) || !open_transfer()) return false;
    for (std::size_t i = 0; i != copies.size(); ++i) {
        const auto offset = i * stride * sizeof(std::uint32_t);
        copy_buffer_range(copies[i].record.buffer, copies[i].record.offset, groups.buffer, groups.offset + offset,
                          sizeof(std::uint32_t));
        make_resident(pending, copies[i].record.buffer);
        mesh_groups.emplace(copies[i].address, groups.gpu + offset);
    }
    return close_transfer();
}

bool Encoder::draw_indexed(const DrawIndexed& command) {
    if (!resolve_carried_predicate()) return false;
    if (predicate_open() && predicate.used == predicate_capacity && !resolve_predicate()) return false;
    if (!state.render_enabled) return true;
    const bool expanded = expands_to_triangles(command.primitive);
    const auto primitive = expanded ? std::optional<MTLPrimitiveType>{MTLPrimitiveTypeTriangle}
                                 : primitive_type(command.primitive);
    if (!primitive)
        return fail(ReplayErrorCode::unsupported,
                    "indexed draw primitive is unsupported by Metal replay");
    if (command.count < 0 || command.instances < 0 || command.base_instance < 0)
        return fail(ReplayErrorCode::invalid_command,
                    "indexed draw contains a negative range");
    if (!command.count || !command.instances) return true;
    const std::uint64_t element_size = command.index_type == NVN_INDEX_TYPE_UNSIGNED_BYTE ? 1 :
        command.index_type == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 :
        command.index_type == NVN_INDEX_TYPE_UNSIGNED_INT ? 4 : 0;
    if (!element_size)
        return fail(ReplayErrorCode::invalid_command, "indexed draw has an invalid index type");
    if (!prepare_graphics_uniform_writes() || !order_buffer_writes()) return false;
    const auto bytes = std::uint64_t(command.count) * element_size;
    state.index_buffer = {command.index_buffer, bytes};
    const auto source = buffer_address(command.index_buffer, bytes);
    if (!source.cpu || source.offset > source.length || bytes > source.length - source.offset)
        return fail(ReplayErrorCode::missing_resource, "index buffer range is unavailable");
    if (!expanded && element_size >= 2 && (!state.primitive_restart ||
        (state.restart_index & (element_size == 2 ? 0xffffu : UINT32_MAX)) ==
            (element_size == 2 ? 0xffffu : UINT32_MAX))) {
        note_base_instance(static_cast<std::uint32_t>(command.base_instance));
        if (!open_render(command.primitive) ||
            !bind_draw_support(static_cast<std::uint32_t>(command.base_instance))) return false;
        make_resident(pending, source.buffer);
        encode_draw_indexed(*primitive, command.count,
                            element_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32,
                            source.gpu, bytes, command.instances,
                            command.base_vertex, command.base_instance);
    } else {
        // Byte indices and custom restart values require a CPU conversion, and
        // may draw several segments: resolve an open block on the CPU first.
        if (!resolve_predicate()) return false;
        if (!state.render_enabled) return true;
        const bool written = guest_written(command.index_buffer, bytes) ||
            std::any_of(other_buffer_writes.begin(), other_buffer_writes.end(), [&](const auto& range) {
                return command.index_buffer < range.first + range.second &&
                    range.first < command.index_buffer + bytes;
            });
        if (written && !synchronize_gpu()) return false;
        if (!order_cpu_read(command.index_buffer, bytes, false, source.cpu)) return false;
        const auto restart = state.restart_index & (UINT32_MAX >> (32 - element_size * 8));
        if (expanded) {
            // Quads and fans expand straight into this submission's upload
            // arena as one triangle list.
            const bool fan = command.primitive == NVN_DRAW_PRIMITIVE_TRIANGLE_FAN;
            const auto capacity = expanded_capacity(fan, static_cast<std::size_t>(command.count));
            if (!capacity) return true;
            if (capacity > max_expanded_indices)
                return fail(ReplayErrorCode::unsupported, "expanded draw exceeds Metal's 32-bit index count");
            const auto converted = backend.allocate_upload(pending, capacity * sizeof(std::uint32_t));
            if (!converted) return fail(ReplayErrorCode::allocation, "replay upload arena allocation failed");
            const auto used = expand_triangles(reinterpret_cast<std::uint32_t*>(converted.cpu),
                                               reinterpret_cast<const std::uint8_t*>(source.cpu),
                                               static_cast<unsigned>(element_size), 0,
                                               static_cast<std::size_t>(command.count), fan,
                                               state.primitive_restart, restart);
            if (used == expansion_collision)
                return fail(ReplayErrorCode::unsupported, "index value collides with the native restart sentinel");
            if (!used) return true;
            note_base_instance(static_cast<std::uint32_t>(command.base_instance));
            if (!open_render(command.primitive)) return false;
            if (!bind_draw_support(static_cast<std::uint32_t>(command.base_instance))) return false;
            encode_draw_indexed(*primitive, static_cast<std::uint32_t>(used), MTLIndexTypeUInt32, converted.gpu,
                                used * sizeof(std::uint32_t), command.instances, command.base_vertex,
                                command.base_instance);
            note_draw_writes();
            ++backend.mutable_statistics().draws;
            ++pending.counters.draws;
            return true;
        }
        const auto* input = source.cpu;
        std::vector<std::uint32_t> indices(command.count);
        std::vector<std::pair<std::size_t, std::size_t>> segments;
        std::size_t segment = 0;
        for (std::size_t index = 0; index != indices.size(); ++index) {
            std::uint32_t value = 0;
            std::memcpy(&value, input + index * element_size, element_size);
            indices[index] = value;
            if (state.primitive_restart && value == restart) {
                if (index > segment) segments.emplace_back(segment, index - segment);
                segment = index + 1;
            } else if (value == UINT32_MAX) {
                return fail(ReplayErrorCode::unsupported, "index value collides with the native restart sentinel");
            }
        }
        if (segment < indices.size()) segments.emplace_back(segment, indices.size() - segment);
        if (segments.empty()) return true;
        const auto converted = upload(indices.size() * sizeof(std::uint32_t), indices.data());
        if (!converted) return false;
        note_base_instance(static_cast<std::uint32_t>(command.base_instance));
        if (!open_render(command.primitive)) return false;
        if (!bind_draw_support(static_cast<std::uint32_t>(command.base_instance))) return false;

        for (const auto& [first, count] : segments)
            encode_draw_indexed(*primitive, count, MTLIndexTypeUInt32,
                                converted.gpu + first * sizeof(std::uint32_t),
                                count * sizeof(std::uint32_t), command.instances,
                                command.base_vertex, command.base_instance);
    }
    note_draw_writes();
    ++backend.mutable_statistics().draws;
    ++pending.counters.draws;
    return true;
}

bool Encoder::draw_indirect(const DrawIndirect& command) {
    if (!resolve_predicate()) return false;
    if (!state.render_enabled) return true;
    if (!synchronize_gpu()) return false;
    const auto source = buffer_at(command.indirect_buffer,
        sizeof(MTLDrawPrimitivesIndirectArguments));
    if (!source.buffer.contents)
        return fail(ReplayErrorCode::missing_resource, "indirect draw arguments are unavailable");
    MTLDrawPrimitivesIndirectArguments arguments{};
    std::memcpy(&arguments, static_cast<const std::byte*>(source.buffer.contents) + source.offset,
                sizeof arguments);
    if (arguments.vertexStart > INT_MAX || arguments.vertexCount > INT_MAX ||
        arguments.instanceCount > INT_MAX || arguments.baseInstance > INT_MAX)
        return fail(ReplayErrorCode::invalid_command, "indirect draw range exceeds the API limits");
    return draw({command.primitive, static_cast<int>(arguments.vertexStart),
        static_cast<int>(arguments.vertexCount), static_cast<int>(arguments.instanceCount),
        static_cast<int>(arguments.baseInstance)});
}

bool Encoder::draw_indexed_indirect(const DrawIndexedIndirect& command) {
    if (!resolve_predicate()) return false;
    if (!state.render_enabled) return true;
    if (!synchronize_gpu()) return false;
    const auto source = buffer_at(command.indirect_buffer,
        sizeof(MTLDrawIndexedPrimitivesIndirectArguments));
    if (!source.buffer.contents)
        return fail(ReplayErrorCode::missing_resource, "indexed indirect arguments are unavailable");
    MTLDrawIndexedPrimitivesIndirectArguments arguments{};
    std::memcpy(&arguments, static_cast<const std::byte*>(source.buffer.contents) + source.offset,
                sizeof arguments);
    const std::uint64_t element_size = command.index_type == NVN_INDEX_TYPE_UNSIGNED_BYTE ? 1 :
        command.index_type == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 :
        command.index_type == NVN_INDEX_TYPE_UNSIGNED_INT ? 4 : 0;
    if (!element_size || arguments.indexCount > INT_MAX || arguments.instanceCount > INT_MAX ||
        arguments.baseInstance > INT_MAX ||
        command.index_buffer > UINT64_MAX - arguments.indexStart * element_size)
        return fail(ReplayErrorCode::invalid_command, "indexed indirect draw range exceeds the API limits");
    return draw_indexed({command.primitive, command.index_type, static_cast<int>(arguments.indexCount),
        arguments.baseVertex, static_cast<int>(arguments.baseInstance), static_cast<int>(arguments.instanceCount),
        command.index_buffer + arguments.indexStart * element_size});
}

bool Encoder::dispatch(const Dispatch& command) {
    const SiteScope site_scope(*this, TransferSite::dispatch);
    compute_dump_images.clear();
    if (command.x < 0 || command.y < 0 || command.z < 0)
        return fail(ReplayErrorCode::invalid_command,
                    "compute dispatch contains a negative grid dimension");
    // A grid with no threadgroups does no work, and Metal rejects encoding one.
    if (!command.x || !command.y || !command.z) {
        if (backend.mutable_statistics().empty_dispatches++ < 8)
            armigo::diag::trace( "nvn_empty_dispatch frame=%llu grid=%dx%dx%d caller=0x%llx\n",
                         static_cast<unsigned long long>(pending.fence), command.x, command.y,
                         command.z, static_cast<unsigned long long>(command.caller));
        return true;
    }
    if (state.failed_stages & compute_stage_bit)
        return fail(ReplayErrorCode::missing_program, "program stage failed to translate");
    const auto& program = state.stage_programs[NVN_SHADER_STAGE_COMPUTE];
    if (!program || !program->stages[NVN_SHADER_STAGE_COMPUTE])
        return fail(ReplayErrorCode::missing_program,
                    "compute dispatch has no resolved compute stage");
    if (!check_compute_layout()) return false;
    if (compute_layout()) {
        const std::array<std::uint32_t, 3> groups{std::uint32_t(command.x), std::uint32_t(command.y),
                                                  std::uint32_t(command.z)};
        if (groups != dispatch_groups) {
            dispatch_groups = groups;
            invalidate_compute_driver();
        }
    }
    // Ending the pass orders the dispatch after any earlier draw's writes.
    buffer_write_order.clear();
    if (!flush_uniforms()) return false;
    close_render(PassBreak::dispatch);
    const auto shader_owner = program->stages[NVN_SHADER_STAGE_COMPUTE];
    const auto& shader = *shader_owner;
    std::string pipeline_error;
    const auto pipeline = backend.compute_pipeline(shader_owner, pipeline_error);
    if (!pipeline) return fail(ReplayErrorCode::unsupported, std::move(pipeline_error));
    const bool new_compute_table = !compute_table;
    if (new_compute_table) {
        auto* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 31;
        descriptor.maxTextureBindCount = 128;
        descriptor.maxSamplerStateBindCount = 16;
        descriptor.initializeBindings = YES;
        NSError* error = nil;
        compute_table = [backend.device() newArgumentTableWithDescriptor:descriptor error:&error];
    }
    if (!compute_table)
        return fail(ReplayErrorCode::allocation,
                    "Metal replay compute argument table creation failed");
    if (new_compute_table)
        [pending.retained addObject:compute_table];
    const bool profiled = dispatch_profile() && !predicate_open();
    if (profiled && !profile_cut({})) return false;
    if (!prepare_images({{NVN_SHADER_STAGE_COMPUTE, &shader}})) return false;
    if (!bind_stage(NVN_SHADER_STAGE_COMPUTE, shader, compute_table, false)) return false;
    // A dispatch whose guest accesses are complete runs without a barrier
    // after independent transfer work.
    if (!open_transfer_with(compute_accesses, !compute_untracked)) return false;
    if (!set_compute_pipeline(*pipeline)) return false;
    set_compute_table(compute_table);
    const auto block = shader.compute_local_size;
    encode_dispatch(MTLSizeMake(command.x, command.y, command.z),
                    MTLSizeMake(block[0], block[1], block[2]));
    [pending.retained addObject:pipeline->library];
    note_shader_writes(NVN_SHADER_STAGE_COMPUTE);
    note_writes();
    ++backend.mutable_statistics().dispatches;
    ++pending.counters.dispatches;
    buffer_write_order.write();
    if (backend.pass_dump_enabled() && !compute_dump_images.empty()) {
        if (!close_transfer()) {
            compute_dump_images.clear();
            return false;
        }
        dump_compute_dispatch();
    }
    compute_dump_images.clear();
    return !profiled || profile_cut(profile_label(shader));
}

bool Encoder::dispatch_indirect(const DispatchIndirect& command) {
    const SiteScope site_scope(*this, TransferSite::dispatch);
    compute_dump_images.clear();
    if (state.failed_stages & compute_stage_bit)
        return fail(ReplayErrorCode::missing_program, "program stage failed to translate");
    const auto& program = state.stage_programs[NVN_SHADER_STAGE_COMPUTE];
    if (!program || !program->stages[NVN_SHADER_STAGE_COMPUTE])
        return fail(ReplayErrorCode::missing_program,
                    "indirect dispatch has no resolved compute stage");
    if (!check_compute_layout()) return false;
    // The workgroup counts reach the driver data on the GPU: this dispatch
    // takes driver data of its own, and later dispatches new data again.
    const bool counts_in_driver_data = compute_layout() != nullptr;
    if (counts_in_driver_data) {
        dispatch_groups = {};
        invalidate_compute_driver();
        compute_driver_slice = {};
    }
    const auto arguments = buffer_at(
        command.address, sizeof(MTLDispatchThreadgroupsIndirectArguments));
    if (!arguments.buffer ||
        ((arguments.buffer.gpuAddress + arguments.offset) & 3))
        return fail(ReplayErrorCode::missing_resource,
                    "indirect dispatch arguments are unavailable or misaligned");
    // Ending the pass orders the dispatch after any earlier draw's writes.
    buffer_write_order.clear();
    if (!flush_uniforms()) return false;
    close_render(PassBreak::dispatch);
    const auto shader_owner = program->stages[NVN_SHADER_STAGE_COMPUTE];
    const auto& shader = *shader_owner;
    std::string pipeline_error;
    const auto pipeline = backend.compute_pipeline(shader_owner, pipeline_error);
    if (!pipeline) return fail(ReplayErrorCode::unsupported, std::move(pipeline_error));
    const bool new_compute_table = !compute_table;
    if (new_compute_table) {
        auto* descriptor = [MTL4ArgumentTableDescriptor new];
        descriptor.maxBufferBindCount = 31;
        descriptor.maxTextureBindCount = 128;
        descriptor.maxSamplerStateBindCount = 16;
        descriptor.initializeBindings = YES;
        compute_table = [backend.device() newArgumentTableWithDescriptor:descriptor error:nil];
    }
    if (!compute_table)
        return fail(ReplayErrorCode::allocation, "indirect compute argument table allocation failed");
    if (dispatch_profile() && !predicate_open() && !profile_cut({})) return false;
    if (!prepare_images({{NVN_SHADER_STAGE_COMPUTE, &shader}})) return false;
    if (!bind_stage(NVN_SHADER_STAGE_COMPUTE, shader, compute_table, false)) return false;
    if (new_compute_table)
        [pending.retained addObject:compute_table];
    if (counts_in_driver_data && compute_driver_slice) {
        if (!open_transfer()) return false;
        copy_buffer_range(arguments.buffer, arguments.offset, compute_driver_slice.buffer,
                          compute_driver_slice.offset + 4, sizeof dispatch_groups);
        invalidate_compute_driver();
    }
    // Undeclared, so the dispatch waits for the copy above.
    if (!open_transfer()) return false;
    if (!set_compute_pipeline(*pipeline)) return false;
    set_compute_table(compute_table);
    const auto block = shader.compute_local_size;
    make_resident(pending, arguments.buffer);
    [pending.retained addObjectsFromArray:@[arguments.buffer, pipeline->library]];
    encode_dispatch_indirect(arguments.buffer.gpuAddress + arguments.offset,
                             MTLSizeMake(block[0], block[1], block[2]));
    note_shader_writes(NVN_SHADER_STAGE_COMPUTE);
    note_writes();
    ++backend.mutable_statistics().dispatches;
    ++pending.counters.dispatches;
    buffer_write_order.write();
    if (backend.pass_dump_enabled() && !compute_dump_images.empty()) {
        if (!close_transfer()) {
            compute_dump_images.clear();
            return false;
        }
        dump_compute_dispatch();
    }
    compute_dump_images.clear();
    return !dispatch_profile() || predicate_open() || profile_cut(profile_label(shader));
}

bool Encoder::copy_buffer(const CopyBuffer& command) {
    const SiteScope site_scope(*this, TransferSite::copy);
    if (!prepare_uniform_access(command.source, command.size, command.destination)) return false;
    if (!command.size) return true;
    if (command.flags != NVN_COPY_FLAGS_NONE)
        return fail(ReplayErrorCode::unsupported,
                    "buffer copy flags are unsupported");
    const auto source = buffer_at(command.source, command.size);
    const auto destination = buffer_at(command.destination, command.size);
    if (!source.buffer || !destination.buffer)
        return fail(ReplayErrorCode::missing_resource,
                    "buffer copy range is unavailable");
    if (!open_transfer()) return false;
    make_resident(pending, source.buffer);
    make_resident(pending, destination.buffer);
    [pending.retained addObjectsFromArray:@[source.buffer, destination.buffer]];
    const bool overlap = source.buffer == destination.buffer &&
        source.offset < destination.offset + command.size &&
        destination.offset < source.offset + command.size;
    if (overlap) {
        const auto staging = upload(command.size);
        if (!staging) return false;
        copy_buffer_range(source.buffer, source.offset, staging.buffer, staging.offset, command.size);
        encoder_barrier(transfer, MTLStageBlit, MTLStageBlit);
        copy_buffer_range(staging.buffer, staging.offset, destination.buffer, destination.offset,
                          command.size);
    } else {
        copy_buffer_range(source.buffer, source.offset, destination.buffer, destination.offset,
                          command.size);
    }
    pending.buffer_writes.emplace_back(command.destination, command.size);
    other_buffer_writes.emplace_back(command.destination, command.size);
    discard_gpu_uniform_captures(command.destination, command.size);
    buffer_write_order.write();
    note_writes();
    if (reports) reports->copy(command.source, command.destination, command.size);
    ++backend.mutable_statistics().copies;
    return true;
}

bool Encoder::copy_buffer_to_texture(const CopyBufferToTexture& command) {
    return copy_buffer_texture(command, true);
}

bool Encoder::copy_texture_to_buffer(const CopyBufferToTexture& command) {
    return copy_buffer_texture(command, false);
}

bool Encoder::copy_buffer_texture(const CopyBufferToTexture& command, bool upload) {
    const SiteScope site_scope(*this, upload ? TransferSite::upload : TransferSite::copy);
    if (!flush_uniforms()) return false;
    if (!command.texture)
        return fail(ReplayErrorCode::invalid_command, "buffer/texture copy has no texture");
    if (command.flags != NVN_COPY_FLAGS_NONE)
        return fail(ReplayErrorCode::unsupported, "buffer/texture copy flags are unsupported");
    const auto* view = command.has_view ? &command.view : nullptr;
    id<MTLTexture> texture = texture_for_use(*command.texture, view);
    if (!texture) return fail(ReplayErrorCode::missing_resource, "copy texture is unavailable");
    const auto& region = command.region;
    if (!valid_region(texture, region))
        return fail(ReplayErrorCode::invalid_command, "copy region exceeds the texture extent");
    if (!region.width || !region.height || !region.depth) return true;
    if (texture.sampleCount != 1)
        return fail(ReplayErrorCode::unsupported, "multisample buffer/texture copies require a resolve");
    const auto format = texture_format(view && view->sets_format ? view->format : command.texture->description.format);
    if (!format) return fail(ReplayErrorCode::unsupported, "copy texture format is unsupported");
    const auto native = metal_storage_format(*format);
    const auto* native_format = std::get_if<MTLPixelFormat>(&native);
    const auto info = graphics::texture_format_info(*format);
    if (!native_format || *native_format != texture.pixelFormat || info.stencil)
        return fail(ReplayErrorCode::unsupported, "copy texture storage requires plane conversion");
    if (region.xoffset % info.block_width || region.yoffset % info.block_height ||
        (region.width % info.block_width && region.xoffset + std::uint64_t(region.width) != texture.width) ||
        (region.height % info.block_height && region.yoffset + std::uint64_t(region.height) != texture.height))
        return fail(ReplayErrorCode::invalid_command, "compressed copy region is not block aligned");
    const auto blocks = (std::uint64_t(region.width) + info.block_width - 1) / info.block_width;
    const auto rows = (std::uint64_t(region.height) + info.block_height - 1) / info.block_height;
    const auto tight_row = blocks * info.bytes_per_block;
    const auto row = state.copy_row_stride ? std::uint64_t(state.copy_row_stride) : tight_row;
    if (!row || row < tight_row || row % info.bytes_per_block || rows > UINT64_MAX / row)
        return fail(ReplayErrorCode::invalid_command, "copy row stride is invalid");
    const auto image = state.copy_image_stride ? std::uint64_t(state.copy_image_stride) : row * rows;
    if (image < row * rows || std::uint64_t(region.depth) > UINT64_MAX / image)
        return fail(ReplayErrorCode::invalid_command, "copy image stride is invalid");
    const auto size = (region.depth - 1) * image + (rows - 1) * row + tight_row;
    const auto buffer = buffer_at(command.source, size);
    if (!buffer.buffer) return fail(ReplayErrorCode::missing_resource, "copy buffer range is unavailable");
    if (!open_transfer()) return false;
    make_resident(pending, buffer.buffer);
    make_resident(pending, texture);
    [pending.retained addObjectsFromArray:@[buffer.buffer, texture]];
    const bool array = array_texture(texture);
    const auto layers = array ? region.depth : 1;
    for (int layer = 0; layer != layers; ++layer) {
        const auto origin = MTLOriginMake(region.xoffset, region.yoffset, array ? 0 : region.zoffset);
        const auto extent = MTLSizeMake(region.width, region.height, array ? 1 : region.depth);
        const auto slice = array ? region.zoffset + layer : 0;
        const auto offset = buffer.offset + layer * image;
        if (upload)
            copy_into_texture(buffer.buffer, offset, row, image, extent, texture, slice, 0, origin);
        else
            copy_from_texture(texture, slice, 0, origin, extent, buffer.buffer, offset, row, image);
    }
    if (upload) note_texture_write(*command.texture);
    else {
        pending.buffer_writes.emplace_back(command.source, size);
        other_buffer_writes.emplace_back(command.source, size);
        discard_gpu_uniform_captures(command.source, size);
        buffer_write_order.write();
        if (reports && size) reports->gpu_write(command.source, size);
    }
    note_writes();
    ++backend.mutable_statistics().copies;
    return true;
}

bool Encoder::scale_texture_copy(const CopyTextureToTexture& command, id<MTLTexture> source,
                                 id<MTLTexture> destination) {
    const auto& target = command.destination_region;
    const auto& input = command.source_region;
    const auto format = texture_format(command.has_source_view && command.source_view.sets_format
        ? command.source_view.format : command.source->description.format);
    if (!format || graphics::texture_numeric_class(*format) != graphics::TextureNumericClass::Float ||
        graphics::is_block_compressed(graphics::texture_format_info(*format)) ||
        graphics::texture_format_info(*format).depth || graphics::texture_format_info(*format).stencil ||
        source.textureType != MTLTextureType2D || destination.textureType != MTLTextureType2D ||
        source.sampleCount != 1 || destination.sampleCount != 1 || input.depth != 1 ||
        input.zoffset || target.zoffset || target.width <= 0 || target.height <= 0 ||
        !(destination.usage & MTLTextureUsageRenderTarget))
        return fail(ReplayErrorCode::unsupported, "scaled texture copies require single-sample 2D color textures");
    id<MTLTexture> source_root = source, destination_root = destination;
    while (source_root.parentTexture) source_root = source_root.parentTexture;
    while (destination_root.parentTexture) destination_root = destination_root.parentTexture;
    if (source_root == destination_root)
        return fail(ReplayErrorCode::unsupported, "scaled in-place texture copies require staging");
    id<MTLBuffer> source_buffer = source_root.buffer, destination_buffer = destination_root.buffer;
    if (source_buffer && destination_buffer) {
        // Linear textures can be distinct objects over the same buffer or
        // mapped bytes. This unstaged path requires disjoint texture storage,
        // including row padding; a cropped rectangle does not relax that rule.
        const auto source_size = std::min<std::uint64_t>(
            std::uint64_t(source_root.bufferBytesPerRow) * source_root.height,
            source_buffer.length - source_root.bufferOffset);
        const auto destination_size = std::min<std::uint64_t>(
            std::uint64_t(destination_root.bufferBytesPerRow) * destination_root.height,
            destination_buffer.length - destination_root.bufferOffset);
        if (!source_size || !destination_size)
            return fail(ReplayErrorCode::unsupported, "scaled linear texture storage span is unavailable");
        const auto overlaps = [](std::uint64_t a, std::uint64_t a_size, std::uint64_t b, std::uint64_t b_size) {
            return a <= b ? b - a < a_size : a - b < b_size;
        };
        bool alias = source_buffer == destination_buffer && overlaps(
            source_root.bufferOffset, source_size, destination_root.bufferOffset, destination_size);
        if (source_buffer != destination_buffer) {
            if (source_buffer.contents && destination_buffer.contents)
                alias = overlaps(reinterpret_cast<std::uintptr_t>(source_buffer.contents) + source_root.bufferOffset,
                    source_size, reinterpret_cast<std::uintptr_t>(destination_buffer.contents) + destination_root.bufferOffset,
                    destination_size);
            if (!alias && source_buffer.gpuAddress && destination_buffer.gpuAddress)
                alias = overlaps(source_buffer.gpuAddress + source_root.bufferOffset, source_size,
                    destination_buffer.gpuAddress + destination_root.bufferOffset, destination_size);
        }
        if (alias)
            return fail(ReplayErrorCode::unsupported, "scaled overlapping buffer texture copies require staging");
    }
    const bool linear = command.flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT;
    if (linear && !backend.device().supports32BitFloatFiltering &&
        (source.pixelFormat == MTLPixelFormatR32Float || source.pixelFormat == MTLPixelFormatRG32Float ||
         source.pixelFormat == MTLPixelFormatRGBA32Float))
        return fail(ReplayErrorCode::unsupported, "linear filtering of 32-bit float copy sources is unavailable");
    std::string detail;
    id<MTLRenderPipelineState> pipeline = backend.texture_copy_pipeline(destination.pixelFormat, detail);
    if (!pipeline) return fail(ReplayErrorCode::unsupported, std::move(detail));
    id<MTLSamplerState> sampler = backend.texture_copy_sampler(linear);
    id<MTL4ArgumentTable> arguments = backend.texture_copy_arguments();
    if (!sampler || !arguments)
        return fail(ReplayErrorCode::allocation, "scaled texture copy binding allocation failed");
    if (!close_render(PassBreak::transfer) || !close_transfer()) return false;
    const auto parameters = backend.allocate_upload(pending, 8 * sizeof(float));
    if (!parameters) return fail(ReplayErrorCode::allocation, "scaled texture copy parameter allocation failed");
    const std::array<float, 8> values{float(input.xoffset), float(input.yoffset),
        float(input.width) / float(target.width), float(input.height) / float(target.height),
        float(target.xoffset), float(target.yoffset), float(source.width), float(source.height)};
    std::memcpy(parameters.cpu, values.data(), sizeof values);
    make_resident(pending, source); make_resident(pending, destination); make_resident(pending, parameters.buffer);
    [pending.retained addObjectsFromArray:@[source, destination, parameters.buffer, pipeline, sampler, arguments]];
    const auto region = target;
    const bool whole = !target.xoffset && !target.yoffset && NSUInteger(target.width) == destination.width &&
        NSUInteger(target.height) == destination.height;
    if (!encode_call(^(EncodeCallContext& context) {
        auto* pass = [MTL4RenderPassDescriptor new];
        pass.colorAttachments[0].texture = destination;
        pass.colorAttachments[0].loadAction = whole ? MTLLoadActionDontCare : MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTL4RenderCommandEncoder> render = [context.commands renderCommandEncoderWithDescriptor:pass];
        if (!render) { context.error = "scaled texture copy encoder creation failed"; return; }
        [render barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageVertex | MTLStageFragment
                     visibilityOptions:MTL4VisibilityOptionDevice];
        [arguments setTexture:source.gpuResourceID atIndex:0];
        [arguments setSamplerState:sampler.gpuResourceID atIndex:0];
        [arguments setAddress:parameters.gpu atIndex:0];
        [render setArgumentTable:arguments atStages:MTLRenderStageFragment];
        [render setRenderPipelineState:pipeline];
        [render setCullMode:MTLCullModeNone];
        [render setViewport:MTLViewport{double(region.xoffset), double(region.yoffset),
            double(region.width), double(region.height), 0, 1}];
        [render setScissorRect:MTLScissorRect{NSUInteger(region.xoffset), NSUInteger(region.yoffset),
            NSUInteger(region.width), NSUInteger(region.height)}];
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [render endEncoding];
    }, ReplayErrorCode::encoding)) return false;
    note_texture_write(*command.destination);
    note_writes();
    ++backend.mutable_statistics().copies;
    return true;
}

bool Encoder::copy_texture_to_texture(const CopyTextureToTexture& command) {
    const SiteScope site_scope(*this, TransferSite::copy);
    if (!command.source || !command.destination)
        return fail(ReplayErrorCode::invalid_command,
                    "texture copy has a missing source or destination");
    if (command.flags & ~NVN_COPY_FLAGS_LINEAR_FILTER_BIT)
        return fail(ReplayErrorCode::unsupported, "mirrored or unknown texture copy flags are unsupported");
    if (command.source_region.depth != command.destination_region.depth)
        return fail(ReplayErrorCode::unsupported, "texture copies cannot scale layer or depth counts");
    const auto* source_view = command.has_source_view ? &command.source_view : nullptr;
    const auto* destination_view = command.has_destination_view ? &command.destination_view : nullptr;
    id<MTLTexture> source = texture_for_use(*command.source, source_view);
    id<MTLTexture> destination = texture_for_use(*command.destination,
                                                             destination_view);
    if (!source || !destination)
        return fail(ReplayErrorCode::missing_resource,
                    "texture copy source or destination is unavailable");
    if (source.pixelFormat != destination.pixelFormat)
        return fail(ReplayErrorCode::unsupported,
                    "texture copy requires identical Metal pixel formats");
    if (!valid_region(source, command.source_region) || !valid_region(destination, command.destination_region))
        return fail(ReplayErrorCode::invalid_command, "texture copy region is invalid");
    if (source == destination)
        return fail(ReplayErrorCode::unsupported, "in-place texture copies require staging");
    if (!command.source_region.width || !command.source_region.height ||
        !command.source_region.depth) return true;
    if (std::uint64_t(command.source_region.xoffset) + command.source_region.width > source.width ||
        std::uint64_t(command.source_region.yoffset) + command.source_region.height > source.height ||
        std::uint64_t(command.destination_region.xoffset) + command.destination_region.width >
            destination.width ||
        std::uint64_t(command.destination_region.yoffset) + command.destination_region.height >
            destination.height)
        return fail(ReplayErrorCode::invalid_command,
                    "texture copy exceeds a source or destination extent");
    if (command.destination->memory && backend.resources().texture_is_in_place(*command.destination) &&
        uniform_cpu_conflict(reinterpret_cast<std::uintptr_t>(command.destination->memory),
                             command.destination->size) && !flush_uniforms()) return false;
    if (command.source_region.width != command.destination_region.width ||
        command.source_region.height != command.destination_region.height)
        return scale_texture_copy(command, source, destination);
    if (!open_transfer()) return false;
    make_resident(pending, source);
    make_resident(pending, destination);
    [pending.retained addObjectsFromArray:@[source, destination]];
    const bool source_array = source.textureType == MTLTextureType1DArray ||
                              source.textureType == MTLTextureType2DArray ||
                              source.textureType == MTLTextureTypeCube ||
                              source.textureType == MTLTextureTypeCubeArray;
    const bool destination_array = destination.textureType == MTLTextureType1DArray ||
                                   destination.textureType == MTLTextureType2DArray ||
                                   destination.textureType == MTLTextureTypeCube ||
                                   destination.textureType == MTLTextureTypeCubeArray;
    if (source_array != destination_array)
        return fail(ReplayErrorCode::unsupported,
                    "texture copy cannot reinterpret array layers as depth planes");
    if (source_array) {
        if (std::uint64_t(command.source_region.zoffset) + command.source_region.depth >
                texture_layers(source) ||
            std::uint64_t(command.destination_region.zoffset) + command.destination_region.depth >
                texture_layers(destination))
            return fail(ReplayErrorCode::invalid_command,
                        "texture copy exceeds an array layer range");
        for (std::int32_t layer = 0; layer != command.source_region.depth; ++layer)
            copy_texture_range(source, command.source_region.zoffset + layer, 0,
                               MTLOriginMake(command.source_region.xoffset,
                                             command.source_region.yoffset, 0),
                               MTLSizeMake(command.source_region.width,
                                           command.source_region.height, 1),
                               destination, command.destination_region.zoffset + layer, 0,
                               MTLOriginMake(command.destination_region.xoffset,
                                             command.destination_region.yoffset, 0));
    } else {
        copy_texture_range(source, 0, 0,
                           MTLOriginMake(command.source_region.xoffset,
                                         command.source_region.yoffset,
                                         command.source_region.zoffset),
                           MTLSizeMake(command.source_region.width,
                                       command.source_region.height,
                                       command.source_region.depth),
                           destination, 0, 0,
                           MTLOriginMake(command.destination_region.xoffset,
                                         command.destination_region.yoffset,
                                         command.destination_region.zoffset));
    }
    note_texture_write(*command.destination);
    note_writes();
    ++backend.mutable_statistics().copies;
    return true;
}

bool Encoder::encode_clear_plan(const TextureState& texture_state,
                                const TextureViewState* view,
                                const NVNcopyRegion& copy_region,
                                std::uint8_t color_mask,
                                const std::array<std::uint32_t, 4>& color,
                                bool depth, float depth_value, bool stencil,
                                std::uint32_t stencil_value,
                                std::uint32_t stencil_mask) {
    const SiteScope site_scope(*this, TransferSite::clear_texture);
    if (!copy_region.width || !copy_region.height || !copy_region.depth || (!color_mask && !depth && !stencil))
        return true;
    if (copy_region.xoffset < 0 || copy_region.yoffset < 0 || copy_region.zoffset < 0 ||
        copy_region.width < 0 || copy_region.height < 0 || copy_region.depth < 0)
        return fail(ReplayErrorCode::invalid_command,
                    "clear texture region contains a negative component");
    id<MTLTexture> texture = texture_for_use(texture_state, view);
    if (!texture)
        return fail(ReplayErrorCode::missing_resource,
                    "clear texture resource is unavailable");
    if (!valid_region(texture, copy_region))
        return fail(ReplayErrorCode::invalid_command, "clear region exceeds the texture extent");
    const auto format = texture_format(view && view->sets_format
                                           ? view->format
                                           : texture_state.description.format);
    if (!format)
        return fail(ReplayErrorCode::unsupported,
                    "clear texture format has no replay mapping");
    // A depth or stencil clear of a format without that component clears
    // nothing there, as on the GPU; the other component is still cleared.
    const auto components = graphics::texture_format_info(*format);
    if (components.depth || components.stencil) {
        depth = depth && components.depth;
        stencil = stencil && components.stencil;
        if (!color_mask && !depth && !stencil) return true;
    }
    if (std::uint64_t(copy_region.xoffset) + std::uint64_t(copy_region.width) > texture.width ||
        std::uint64_t(copy_region.yoffset) + std::uint64_t(copy_region.height) > texture.height)
        return fail(ReplayErrorCode::invalid_command,
                    "clear texture region exceeds the resident texture");

    // Cube faces share array storage. The clear encoder addresses that storage
    // through array slices, including when all six faces carry a deferred clear.
    if (texture.textureType == MTLTextureTypeCube || texture.textureType == MTLTextureTypeCubeArray) {
        texture = [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                            textureType:MTLTextureType2DArray
                                                 levels:NSMakeRange(0, texture.mipmapLevelCount)
                                                 slices:NSMakeRange(0, texture.arrayLength * 6)];
        if (!texture)
            return fail(ReplayErrorCode::allocation, "cube clear array view creation failed");
    }
    graphics::TextureDescriptor descriptor{};
    descriptor.width = static_cast<std::uint32_t>(texture.width);
    descriptor.height = static_cast<std::uint32_t>(texture.height);
    descriptor.depth = static_cast<std::uint32_t>(std::max<NSUInteger>(texture.depth, 1));
    descriptor.levels = static_cast<std::uint32_t>(texture.mipmapLevelCount);
    descriptor.layers = static_cast<std::uint32_t>(
        texture.textureType == MTLTextureTypeCube ? 6 :
        texture.textureType == MTLTextureTypeCubeArray ? texture.arrayLength * 6 :
        std::max<NSUInteger>(texture.arrayLength, 1));
    descriptor.samples = static_cast<std::uint32_t>(std::max<NSUInteger>(texture.sampleCount, 1));
    descriptor.format = *format;
    descriptor.type = texture.textureType == MTLTextureType2DArray
        ? graphics::TextureType::Texture2DArray
        : texture.textureType == MTLTextureType3D
            ? graphics::TextureType::Texture3D
            : texture.textureType == MTLTextureTypeCube
                ? graphics::TextureType::Cubemap
                : graphics::TextureType::Texture2D;

    graphics::ClearPlan plan{};
    plan.target.texture = descriptor;
    plan.target.aspect = depth || stencil
        ? (depth && stencil ? graphics::TextureAspect::DepthStencil
                            : depth ? graphics::TextureAspect::Depth
                                    : graphics::TextureAspect::Stencil)
        : graphics::TextureAspect::Color;
    plan.target.level = 0;
    plan.target.layer = static_cast<std::uint32_t>(copy_region.zoffset);
    plan.target.layer_count = static_cast<std::uint32_t>(
        std::max(copy_region.depth, 1));
    plan.region = {copy_region.xoffset, copy_region.yoffset,
                   static_cast<std::uint64_t>(copy_region.width),
                   static_cast<std::uint64_t>(copy_region.height)};
    plan.color_mask = color_mask;
    plan.color_bits = color;
    plan.clear_depth = depth;
    plan.clear_stencil = stencil;
    plan.depth_value = depth_value;
    plan.stencil_value = stencil_value;
    plan.stencil_write_mask = stencil_mask;
    const bool full_region = copy_region.xoffset == 0 && copy_region.yoffset == 0 &&
        std::uint64_t(copy_region.width) == texture.width &&
        std::uint64_t(copy_region.height) == texture.height;
    const bool full_layers = plan.target.layer == 0 &&
        plan.target.layer_count == descriptor.layers;
    plan.fold_into_load = full_region && full_layers &&
        (!color_mask || color_mask == 0xf) && (!stencil || stencil_mask == 0xff);
    plan.fold_layers_into_load = plan.fold_into_load && descriptor.layers > 1;

    if (texture_state.memory && backend.resources().texture_is_in_place(texture_state) &&
        uniform_cpu_conflict(reinterpret_cast<std::uintptr_t>(texture_state.memory),
                             texture_state.size) && !flush_uniforms()) return false;
    if (render || !plan.fold_into_load) close_render(PassBreak::clear);
    close_transfer();
    if (plan.fold_into_load) {
        deferred_clears.push_back({plan, texture, reinterpret_cast<std::uint64_t>(texture_state.memory),
                                   texture_state.size});
        make_resident(pending, texture);
        note_texture_write(texture_state);
        note_writes();
        ++backend.mutable_statistics().clears;
        return true;
    }
    ClearBindings bindings{};
    bindings.level = 0;
    bindings.slice = plan.target.layer;
    if (color_mask) bindings.color = texture;
    if (depth) bindings.depth = texture;
    if (stencil) bindings.stencil = texture;
    if (!encode_planned_clear(plan, bindings, "clear encoding failed: ")) return false;
    make_resident(pending, texture);
    note_texture_write(texture_state);
    note_writes();
    ++backend.mutable_statistics().clears;
    return true;
}

bool Encoder::clear_bound_color(const ClearColor& clear) {
    if (!resolve_predicate()) return false;
    if (!state.render_enabled || !(clear.mask & 0xf)) return true;
    if (clear.index < 0 || clear.index >= state.targets.count || clear.index >= 8 ||
        !state.targets.color[clear.index])
        return fail(ReplayErrorCode::invalid_command,
                    "color clear names an unbound render target");
    const auto* texture = state.targets.color[clear.index];
    const auto* view = state.targets.has_color_view[clear.index]
        ? &state.targets.color_views[clear.index] : nullptr;
    id<MTLTexture> resident = texture_for_use(*texture, view);
    if (!resident)
        return fail(ReplayErrorCode::missing_resource,
                    "color clear render target is unavailable");
    NVNcopyRegion region{0, 0, 0, static_cast<int>(resident.width),
                         static_cast<int>(resident.height), 1};
    if (state.scissor_count) {
        const auto scissor = clip_scissor(state.scissors[0], resident.width, resident.height);
        const auto x = std::min<NSUInteger>(scissor.x, resident.width);
        const auto y = std::min<NSUInteger>(scissor.y, resident.height);
        region.xoffset = static_cast<int>(x);
        region.yoffset = static_cast<int>(y);
        region.width = static_cast<int>(std::min<NSUInteger>(scissor.width,
                                                             resident.width - x));
        region.height = static_cast<int>(std::min<NSUInteger>(scissor.height,
                                                              resident.height - y));
    }
    if (!region.width || !region.height) return true;
    std::array<std::uint32_t, 4> bits{};
    std::memcpy(bits.data(), clear.rgba, sizeof clear.rgba);
    return encode_clear_plan(*texture, view, region,
                             static_cast<std::uint8_t>(clear.mask & 0xf), bits,
                             false, 0.0f, false, 0, 0);
}

bool Encoder::clear_bound_depth(const ClearDepthStencil& clear) {
    if (!resolve_predicate()) return false;
    if (!state.render_enabled || (!clear.depth_mask && !(clear.stencil_mask & 0xff))) return true;
    if (!state.targets.depth) {
        if (!clear.depth_mask && !clear.stencil_mask) return true;
        return fail(ReplayErrorCode::invalid_command,
                    "depth/stencil clear has no bound target");
    }
    const auto* view = state.targets.has_depth_view ? &state.targets.depth_view : nullptr;
    id<MTLTexture> resident = texture_for_use(*state.targets.depth, view);
    if (!resident)
        return fail(ReplayErrorCode::missing_resource,
                    "depth/stencil clear target is unavailable");
    NVNcopyRegion region{0, 0, 0, static_cast<int>(resident.width),
                         static_cast<int>(resident.height), 1};
    if (state.scissor_count) {
        const auto scissor = clip_scissor(state.scissors[0], resident.width, resident.height);
        const auto x = std::min<NSUInteger>(scissor.x, resident.width);
        const auto y = std::min<NSUInteger>(scissor.y, resident.height);
        region.xoffset = static_cast<int>(x);
        region.yoffset = static_cast<int>(y);
        region.width = static_cast<int>(std::min<NSUInteger>(scissor.width,
                                                             resident.width - x));
        region.height = static_cast<int>(std::min<NSUInteger>(scissor.height,
                                                              resident.height - y));
    }
    if (!region.width || !region.height) return true;
    return encode_clear_plan(*state.targets.depth, view, region, 0, {},
                             clear.depth_mask != 0, clear.depth,
                             clear.stencil_mask != 0,
                             static_cast<std::uint32_t>(clear.stencil),
                             static_cast<std::uint32_t>(clear.stencil_mask));
}

bool Encoder::clear_texture(const ClearTexture& clear) {
    if (!clear.texture)
        return fail(ReplayErrorCode::invalid_command,
                    "texture clear has no texture snapshot");
    std::array<std::uint32_t, 4> bits{};
    if (clear.is_integer)
        std::copy(std::begin(clear.color.u), std::end(clear.color.u), bits.begin());
    else
        std::memcpy(bits.data(), clear.color.f, sizeof clear.color.f);
    return encode_clear_plan(*clear.texture, clear.has_view ? &clear.view : nullptr,
                             clear.region, static_cast<std::uint8_t>(clear.mask & 0xf),
                             bits, false, 0.0f, false, 0, 0);
}

// A draw, dispatch, clear or texture copy that replay cannot encode (an
// unsupported feature, a missing resource or program) is left out with a
// counted reason; the rest of the submission still runs. Other failures
// abandon the submission.
std::string Encoder::mesh_stage_identity() const {
    static constexpr const char* names[]{"vs", "fs", "gs", "task", "mesh", "cs"};
    std::string result;
    for (std::size_t slot = 0; slot != stage_count; ++slot) {
        const auto& program = state.stage_programs[slot];
        if (slot) result += ",";
        result += names[slot];
        result += "@" + std::to_string(program ? program->state.serial : 0) + ":";
        result += stage_label(program ? program->stages[slot].get() : nullptr);
    }
    return result + ",failed=" + std::to_string(state.failed_stages);
}

void Encoder::remember_mesh_bind(const BindProgram& command) {
    if (!backend.mesh_diagnostics_enabled()) return;
    if (!mesh_bind_history) mesh_bind_history.emplace();
    auto& history = *mesh_bind_history;
    auto& bind = history.binds[history.total++ % history.binds.size()];
    bind.ordinal = records_seen;
    bind.identity = "mask=" + std::to_string(command.stages) +
        ",program=" + std::to_string(command.program ? command.program->serial : 0) +
        ",result=" + mesh_stage_identity();
}

void Encoder::diagnose_mesh_failure(const DrawMeshTasks& command) {
    if (!backend.mesh_diagnostics_enabled()) return;
    char address[32];
    std::snprintf(address, sizeof address, "%#llx", static_cast<unsigned long long>(command.api_return));
    std::string key = "first=" + std::to_string(command.first) + " count=" + std::to_string(command.count) +
        " api_return=" + (command.api_return ? std::string(address) : "unavailable") +
        " api_return_domain=" + (command.api_return ? "api231-entry-return" : "internal-record") +
        " stages=" + mesh_stage_identity() + " reason=" + error.detail;
    std::string detail = "command_id=" + std::to_string(pending.fence) + ":" + std::to_string(records_seen) +
        " command_id_domain=submission-fence:flattened-record-ordinal " + key +
        " history_scope=current-submission previous_submission=unavailable";
    const auto total = mesh_bind_history ? mesh_bind_history->total : 0;
    const auto count = std::min<std::size_t>(total, 8);
    detail += " history_count=" + std::to_string(count) + " history_dropped=" + std::to_string(total - count);
    for (std::size_t index = total - count; index != total; ++index) {
        const auto& bind = mesh_bind_history->binds[index % 8];
        key += " bind{" + bind.identity + "}";
        detail += " bind{ordinal=" + std::to_string(bind.ordinal) + "," + bind.identity + "}";
    }
    backend.note_mesh_diagnostic(std::move(key), detail);
}

std::unique_ptr<program_capture::Event> Encoder::begin_program_capture(const CommandRecordView& record) const noexcept {
    if (!program_capture::enabled() || (record.op != Op::bind_program && record.op != Op::geometry_processing_mode &&
                                       record.op != Op::draw_mesh_tasks)) return {};
    try {
        auto event = std::make_unique<program_capture::Event>();
        event->backend = reinterpret_cast<std::uint64_t>(&backend); event->submission = pending.fence;
        event->ordinal = records_seen; event->mode_before = state.geometry_mode; event->before = state.capture_caches();
        if (record.op == Op::bind_program) {
            event->kind = program_capture::EventKind::bind;
            event->requested = static_cast<std::uint32_t>(payload<BindProgram>(record).stages);
        } else if (record.op == Op::geometry_processing_mode) {
            event->kind = program_capture::EventKind::mode; event->requested_mode = payload<ScalarInt>(record).value;
        } else {
            event->kind = program_capture::EventKind::direct_draw;
            const auto draw = payload<DrawMeshTasks>(record);
            event->first = draw.first; event->count = draw.count; event->api_return = draw.api_return;
        }
        return event;
    } catch (...) { program_capture::record_loss(); return {}; }
}
void Encoder::finish_program_capture(std::unique_ptr<program_capture::Event> event, const CommandRecordView& record,
                                    bool accepted, std::uint64_t previous_draws) const noexcept {
    if (!event) return;
    try {
        event->accepted = accepted; event->mode_after = state.geometry_mode; event->after = state.capture_caches();
        event->failed = state.failed_stages;
        std::shared_ptr<const ResolvedProgram> program;
        if (record.op == Op::bind_program) {
            const auto command = payload<BindProgram>(record);
            if (command.program) {
                // A zero-mask bind must not wait for a pending translation.
                program = retain_program_snapshot(*command.program);
                bool translating = false;
                const auto ready = retain_ready_program(*command.program, translating);
                if (ready) { event->present = ready->logical_stages; event->present_observed = true; }
            } else event->present_observed = true;
        } else if (record.op == Op::draw_mesh_tasks) {
            const auto& selected_mesh = state.geometry_mode == 1 ? state.task_mesh : state.standalone_mesh;
            const ReplayPrograms::Cached* candidates[]{&state.ordinary[NVN_SHADER_STAGE_FRAGMENT],
                state.geometry_mode ? &selected_mesh : nullptr,
                state.geometry_mode == 1 ? &state.task : nullptr,
                state.geometry_mode == 0 ? &state.ordinary[NVN_SHADER_STAGE_VERTEX] : nullptr};
            for (const auto* candidate : candidates)
                if (candidate && candidate->program && candidate->program->capture_matches) {
                    program = candidate->program; break;
                }
            if (program) { event->present = program->logical_stages; event->present_observed = true; }
            event->encoded = pending.counters.draws > previous_draws;
            if (accepted && !event->encoded) event->detail = "no mesh command encoded";
        }
        if (program) {
            event->generation = program->state.serial; event->capture_call = program->capture_call;
            event->matches = program->capture_matches;
            if (program->capture_mesh_metadata) {
                event->mesh_route = program->captured_mesh_route; event->auxiliary_word = program->captured_auxiliary_word;
            }
        }
        if (!accepted && failed) { event->detail = error.detail; event->error_code = static_cast<int>(error.code); }
        program_capture::observe(std::move(*event));
    } catch (...) { program_capture::record_loss(); }
}

bool Encoder::visit(const CommandRecordView& record) {
    if (failed) return false;
    ++records_seen;
    auto captured = program_capture_enabled ? begin_program_capture(record) : nullptr;
    const auto previous_draws = captured ? pending.counters.draws : 0;
    const bool accepted = visit_command(record);
    if (captured) finish_program_capture(std::move(captured), record, accepted, previous_draws);
    if (accepted) return true;
    if (!failed) return false;
    // A failure that is not a skippable draw, clear or copy abandons the rest of
    // the submission; the host log names each such failure once.
    const auto report_abandon = [&] {
        static std::mutex mutex;
        static std::set<std::string> reported;
        const std::lock_guard lock(mutex);
        if (reported.insert(error.detail).second)
            armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: replay abandoned a submission at op %u: %s\n",
                         static_cast<unsigned>(record.op), error.detail.c_str());
    };
    switch (record.op) {
        case Op::draw:
        case Op::draw_indexed:
        case Op::draw_indirect:
        case Op::draw_indexed_indirect:
        case Op::multi_draw_indirect:
        case Op::multi_draw_indexed_indirect:
        case Op::draw_mesh_tasks:
        case Op::draw_mesh_tasks_indirect:
        case Op::dispatch:
        case Op::dispatch_indirect:
        case Op::clear_color:
        case Op::clear_color_integer:
        case Op::clear_depth_stencil:
        case Op::clear_texture:
        case Op::copy_buffer_to_texture:
        case Op::copy_texture_to_buffer:
        case Op::copy_texture_to_texture: break;
        default: report_abandon(); return false;
    }
    switch (error.code) {
        case ReplayErrorCode::unsupported:
        case ReplayErrorCode::missing_resource:
        case ReplayErrorCode::missing_program: break;
        default: report_abandon(); return false;
    }
    std::string detail;
    if (ReplayImpl::skip_detail_limit()) {
        const auto stage_of = [&](std::size_t slot) {
            const auto& program = state.stage_programs[slot];
            return stage_label(program ? program->stages[slot].get() : nullptr);
        };
        detail = "vs=" + stage_of(NVN_SHADER_STAGE_VERTEX) +
            " task=" + stage_of(NVN_SHADER_STAGE_TESS_CONTROL) +
            " mesh=" + stage_of(NVN_SHADER_STAGE_TESS_EVALUATION) +
            " fs=" + stage_of(NVN_SHADER_STAGE_FRAGMENT) +
            " cs=" + stage_of(NVN_SHADER_STAGE_COMPUTE) +
            " failed_stages=" + std::to_string(state.failed_stages) +
            " rasterizer_discard=" + std::to_string(state.rasterizer_discard);
        const auto describe_texture = [&](const TextureState* texture) {
            if (!texture) return std::string("-");
            const auto& desc = texture->description;
            return std::to_string(texture->serial) + ":target=" + std::to_string(desc.target) +
                ":format=" + std::to_string(desc.format) + ":size=" + std::to_string(desc.width) +
                "x" + std::to_string(desc.height) + "x" + std::to_string(desc.depth);
        };
        for (std::size_t index = 0; index < std::size(state.targets.color); ++index)
            if (state.targets.color[index])
                detail += " c" + std::to_string(index) + "=" + describe_texture(state.targets.color[index]);
        detail += " depth=" + describe_texture(state.targets.depth);
        if (record.op == Op::copy_texture_to_texture) {
            const auto command = payload<CopyTextureToTexture>(record);
            const auto describe_region = [](const NVNcopyRegion& region) {
                return std::to_string(region.xoffset) + "," + std::to_string(region.yoffset) + "," +
                    std::to_string(region.zoffset) + ":" + std::to_string(region.width) + "x" +
                    std::to_string(region.height) + "x" + std::to_string(region.depth);
            };
            detail += " source=" + describe_texture(command.source) +
                " source_region=" + describe_region(command.source_region) +
                " destination=" + describe_texture(command.destination) +
                " destination_region=" + describe_region(command.destination_region) +
                " flags=" + std::to_string(command.flags);
        }
        if (error.code == ReplayErrorCode::missing_resource) {
            detail += " resource=" + backend.resources().take_texture_failure();
        }
    }
    if (record.op == Op::draw_mesh_tasks) diagnose_mesh_failure(payload<DrawMeshTasks>(record));
    backend.note_skip(error.detail, static_cast<unsigned>(record.op), pending.fence, detail);
    feedback_uses.clear();
    feedback_reopen = false;
    feedback_rebind = true;
    failed = false;
    error = {};
    // The render encoder may hold a partial setup; the next draw starts over.
    close_render(PassBreak::skip);
    return true;
}

bool Encoder::visit_command(const CommandRecordView& record) {
    if (failed) return false;
    if (capture_retirement_needed && !retire_captures_if_full()) return false;
    current_op_ = record.op;
    // Transfer and non-draw writes publish new texture contents. Draw writes
    // keep the snapshot visible until the next guest barrier instead.
    switch (record.op) {
        case Op::copy_buffer_to_texture:
        case Op::copy_texture_to_texture:
        case Op::clear_color:
        case Op::clear_color_integer:
        case Op::clear_depth_stencil:
        case Op::clear_texture:
        case Op::downsample:
        case Op::dispatch:
        case Op::dispatch_indirect:
            ++feedback_epoch;
            dirty.arguments[NVN_SHADER_STAGE_FRAGMENT] = true;
            break;
        default: break;
    }
    if (trace_submission_id) {
        switch (record.op) {
            case Op::draw:
            case Op::draw_indexed:
            case Op::draw_indirect:
            case Op::draw_indexed_indirect:
            case Op::multi_draw_indirect:
            case Op::multi_draw_indexed_indirect:
                trace_pass_event("draw", "command", state.render_enabled, state.samples_active,
                                 state.depth_stencil.depth_test, state.depth_stencil.stencil_test);
                break;
            case Op::barrier:
                trace_pass_event("barrier", "command", payload<ScalarInt>(record).value,
                                 pass_depth_only, static_cast<std::uint64_t>(buffer_write_order.pending));
                break;
            case Op::set_render_targets:
                trace_pass_event("targets", "command", same_targets(state.targets, payload<SetRenderTargets>(record)));
                break;
            default: break;
        }
    }
    switch (record.op) {
        case Op::signal_event: {
            const auto command = payload<SignalEvent>(record);
            if (!command.value)
                return fail(ReplayErrorCode::invalid_command,
                            "event signal has no destination");
            if (command.mode != NVN_EVENT_SIGNAL_MODE_WRITE &&
                command.mode != NVN_EVENT_SIGNAL_MODE_ADD)
                return fail(ReplayErrorCode::unsupported,
                            "event signal reduction mode is unsupported");
            if (command.flags)
                return fail(ReplayErrorCode::unsupported,
                            "event signal flags are unsupported");
            if (command.location != NVN_EVENT_SIGNAL_LOCATION_TOP &&
                command.location != NVN_EVENT_SIGNAL_LOCATION_VERTEX_PIPE &&
                command.location != NVN_EVENT_SIGNAL_LOCATION_BOTTOM)
                return fail(ReplayErrorCode::invalid_command,
                            "event signal location is invalid");
            const auto cpu = reinterpret_cast<std::uintptr_t>(command.value);
            if (uniform_cpu_conflict(cpu, sizeof(std::uint32_t)) && !flush_uniforms()) return false;
            discard_uniform_captures(cpu, 0, sizeof(std::uint32_t));
            ++update_generation;
            invalidate_args_all();
            CompletionEffect effect{};
            effect.kind = CompletionEffect::Kind::signal;
            effect.event = command.value;
            effect.value = command.signal_value;
            effect.add = command.mode == NVN_EVENT_SIGNAL_MODE_ADD;
            pending.effects.push_back(effect);
            return true;
        }
        case Op::wait_event: {
            const auto command = payload<WaitEvent>(record);
            if (!command.value)
                return fail(ReplayErrorCode::invalid_command,
                            "event wait has no source");
            const auto& drains = event_wait_drains();
            // The scheduler released a held leading wait when the signal
            // that meets it landed, or when it was encoded as a write in an
            // earlier submission. A GPU signal lands as its submission
            // retires, which publishes that submission's writes under the
            // fence lock this submission's encoding took; an encoded write
            // runs earlier on the one queue, and this submission's first
            // encoder takes a queue barrier. Neither needs a GPU drain.
            if (passed_leading_waits) {
                --passed_leading_waits;
                return !drains.leading || synchronize_gpu();
            }
            if (drains.pending && !synchronize_gpu()) return false;
            // A signal of this submission that leaves the event satisfying the
            // wait: the queue runs it before the wait's later commands, so a
            // barrier orders them on the GPU and the CPU does not wait. Signals
            // that add before any write depend on the earlier value and wait;
            // a later write makes the value known again. A CPU or other-queue
            // write racing the signal is unordered against the wait, so the
            // wait passing right after the signal is one legal order.
            {
                bool signalled = false, known = true;
                std::uint32_t left = 0;
                for (const auto& effect : pending.effects) {
                    if (effect.kind != CompletionEffect::Kind::signal || effect.event != command.value) continue;
                    if (effect.add) {
                        if (!signalled) known = false;
                        left += effect.value;
                    } else {
                        left = effect.value;
                        known = true;
                    }
                    signalled = true;
                }
                const bool met = command.mode == NVN_EVENT_WAIT_MODE_EQUAL
                    ? left == command.wait_value
                    : command.mode == NVN_EVENT_WAIT_MODE_GEQUAL_WRAP && std::int32_t(left - command.wait_value) >= 0;
                if (signalled && known && met && !drains.own) return barrier(PassBreak::wait_event);
                // An earlier submission's encoded write meets it: that
                // submission runs first on the queue, and a barrier orders
                // this one's later commands after its own earlier work.
                if (!signalled && !drains.release &&
                    backend.encoded_write_meets(command.value, command.mode, command.wait_value))
                    return barrier(PassBreak::wait_event);
            }
            // Wait only for a writer of the event word that has not landed:
            // a signal of this submission drains its prefix, one of an
            // earlier submission waits for that submission. An event the CPU
            // or completed work already set needs no GPU wait.
            if (!order_completion_read(nil, 0,
                    reinterpret_cast<const std::byte*>(const_cast<const std::uint32_t*>(command.value)),
                    sizeof(std::uint32_t)))
                return false;
            const auto value = __atomic_load_n(command.value, __ATOMIC_SEQ_CST);
            const bool ready = command.mode == NVN_EVENT_WAIT_MODE_EQUAL
                ? value == command.wait_value
                : command.mode == NVN_EVENT_WAIT_MODE_GEQUAL_WRAP
                    ? std::int32_t(value - command.wait_value) >= 0
                    : false;
            if (!ready)
                return fail(ReplayErrorCode::unsupported,
                            "unsatisfied GPU event waits require a shared-event bridge");
            return true;
        }
        case Op::wait_sync: {
            if (!barrier(PassBreak::wait_sync)) return false;
            const auto command = payload<WaitSync>(record);
            if (!command.sync || !command.sync->fenced)
                return fail(ReplayErrorCode::invalid_command,
                            "command wait names an unfenced sync object");
            if (command.sync->fence >= pending.fence)
                return fail(ReplayErrorCode::unsupported,
                            "command wait names a future queue fence");
            return true;
        }
        case Op::fence_sync: {
            if (!barrier(PassBreak::fence_sync)) return false;
            const auto command = payload<FenceSync>(record);
            if (!command.sync)
                return fail(ReplayErrorCode::invalid_command,
                            "fence command has no sync object");
            if (command.flags ||
                (command.condition != NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE &&
                 command.condition != NVN_SYNC_CONDITION_GRAPHICS_WORLD_SPACE_COMPLETE))
                return fail(ReplayErrorCode::unsupported,
                            "fence condition or flags are unsupported");
            pending.fenced_syncs.push_back(command.sync);
            return true;
        }
        case Op::report_counter: {
            const auto command = payload<ReportCounter>(record);
            // Only isolated query storage may bypass the old ordered lookup.
            // Cached or external consumers of aliases retain that dependency.
            const bool isolated = report_destination_isolated(command.address, sizeof(NVNcounterData));
            const auto destination = isolated
                ? backend.resources().buffer_at(command.address, sizeof(NVNcounterData))
                : buffer_at(command.address, sizeof(NVNcounterData));
            auto* cpu = static_cast<std::byte*>(destination.buffer.contents);
            if (!destination.buffer || !cpu)
                return fail(ReplayErrorCode::missing_resource,
                            "counter report destination is not CPU-visible");
            if (command.type != NVN_COUNTER_TYPE_TIMESTAMP && command.type != NVN_COUNTER_TYPE_TIMESTAMP_TOP &&
                command.type != NVN_COUNTER_TYPE_SAMPLES_PASSED) {
                note_report_counter(command.type, command.address);
                return fail(ReplayErrorCode::unsupported,
                            "requested hardware counter is unsupported by Metal replay");
            }
            ++(isolated ? pending.counters.report_fast : pending.counters.report_ordered);
            ++(isolated ? backend.mutable_statistics().report_fast : backend.mutable_statistics().report_ordered);
            return enqueue_counter(command, destination, cpu + destination.offset);
        }
        case Op::reset_counter: {
            const auto command = payload<ResetCounter>(record);
            if (command.type != NVN_COUNTER_TYPE_SAMPLES_PASSED)
                return fail(ReplayErrorCode::unsupported,
                            "only samples-passed counter reset is supported");
            // Draws count samples into their own visibility slots, so a pass
            // created with the visibility buffer continues across the reset.
            if (!pass_visibility_buffer) close_render(PassBreak::counter);
            CompletionEffect effect{};
            effect.kind = CompletionEffect::Kind::reset;
            pending.effects.push_back(effect);
            state.samples_active = true;
            counter_begin = visibility_cursor;
            counter_reset = true;
            reset_slot = visibility_cursor;
            trace_pass_event("reset", "samples");
            return true;
        }
        case Op::report_value: {
            const SiteScope site_scope(*this, TransferSite::report);
            const auto command = payload<ReportValue>(record);
            const auto destination = buffer_at(command.address,
                                                                    sizeof(command.value));
            if (!destination.buffer || !destination.buffer.contents)
                return fail(ReplayErrorCode::missing_resource,
                            "reported value destination is not CPU-visible");
            if (!prepare_uniform_access(command.address, sizeof command.value)) return false;
            const auto source = upload(sizeof command.value, &command.value);
            if (!source || !open_transfer()) return false;
            make_resident(pending, destination.buffer);
            copy_buffer_range(source.buffer, source.offset, destination.buffer, destination.offset,
                              sizeof command.value);
            other_buffer_writes.emplace_back(command.address, sizeof command.value);
            pending.buffer_writes.emplace_back(command.address, sizeof command.value);
            discard_gpu_uniform_captures(command.address, sizeof command.value);
            buffer_write_order.write();
            note_writes();
            if (reports) reports->write(command.address, sizeof command.value, &command.value, 0);
            return true;
        }
        case Op::copy_buffer: return copy_buffer(payload<CopyBuffer>(record));
        case Op::update_uniforms: {
            const SiteScope site_scope(*this, TransferSite::update_uniforms);
            const auto command = payload<UpdateUniforms>(record);
            const auto header = offsetof(UpdateUniforms, data);
            if (command.size > record.payload_size - header)
                return fail(ReplayErrorCode::invalid_command,
                            "uniform update payload is truncated");
            if (!command.size) return true;
            const auto destination = buffer_at(command.destination,
                                                                    command.size);
            if (!destination.buffer)
                return fail(ReplayErrorCode::missing_resource,
                            "uniform update destination is unavailable");
            const bool alias = uniform_capture_alias(command.destination, command.size, destination);
            if (alias) {
                if (!prepare_uniform_access(command.destination, command.size)) return false;
                const auto cpu = destination.buffer.contents
                    ? reinterpret_cast<std::uintptr_t>(destination.buffer.contents) + destination.offset : 0;
                discard_uniform_captures(cpu, destination.buffer.gpuAddress + destination.offset, command.size);
            }
            const auto* update_bytes = static_cast<const std::byte*>(record.payload) + header;
            if (alias || !version_uniform(command.destination, command.size, update_bytes, destination)) {
                if (failed || !flush_uniforms()) return false;
                // A canonical NVN address may still resolve through a distinct
                // native GPU allocation. Preserve visibility before its copy.
                if (!alias && !prepare_uniform_access(command.destination, command.size)) return false;
                record_uniform_update(command.destination, command.size, update_bytes, destination);
                const auto source = upload(command.size, update_bytes);
                if (!source || !open_transfer()) return false;
                make_resident(pending, destination.buffer);
                copy_buffer_range(source.buffer, source.offset, destination.buffer, destination.offset,
                                  command.size);
                pending.buffer_writes.emplace_back(command.destination, command.size);
                record_uniform_copy(command.destination, command.size, destination);
                buffer_write_order.write();
                note_writes();
                ++backend.mutable_statistics().copies;
            }
            if (!uniform_words.empty()) refresh_uniform_words(command.destination, command.size, update_bytes);
            invalidate_uniforms(command.destination, command.size, cpu_addressed_buffers ? command.destination :
                destination.buffer.contents ? reinterpret_cast<std::uintptr_t>(destination.buffer.contents) + destination.offset : 0);
            if (reports) reports->write(command.destination, command.size, update_bytes, 0);
            if (failed) return false;
            ++pending.counters.uniform_invalidations;
            return true;
        }
        case Op::clear_buffer: {
            const SiteScope site_scope(*this, TransferSite::clear_buffer);
            const auto command = payload<ClearBuffer>(record);
            if (!prepare_uniform_access(command.address, command.size)) return false;
            if (!command.size) return true;
            if (command.size > 64 * 1024 * 1024)
                return fail(ReplayErrorCode::allocation,
                            "buffer clear exceeds the per-command staging limit");
            std::vector<std::uint32_t> values((command.size + 3) / 4, command.value);
            const auto destination = buffer_at(command.address,
                                                                    command.size);
            if (!destination.buffer)
                return fail(ReplayErrorCode::missing_resource,
                            "buffer clear destination is unavailable");
            const auto source = upload(command.size, values.data());
            if (!source) return false;
            if (!open_transfer()) return false;
            make_resident(pending, destination.buffer);
            copy_buffer_range(source.buffer, source.offset, destination.buffer, destination.offset,
                              command.size);
            pending.buffer_writes.emplace_back(command.address, command.size);
            other_buffer_writes.emplace_back(command.address, command.size);
            discard_gpu_uniform_captures(command.address, command.size);
            buffer_write_order.write();
            note_writes();
            if (reports) reports->write(command.address, command.size, &command.value, 4);
            ++backend.mutable_statistics().clears;
            return true;
        }
        case Op::set_render_targets:
            if (same_targets(state.targets, payload<SetRenderTargets>(record)))
                return render ? true : cut_submission();
            // Clears recorded for the old targets stay pending: the next pass
            // folds those of its attachments into its load actions.
            static const bool carry_clears = [] {
                const char* value = std::getenv("ARMIGO_NVN_CARRY_CLEARS");
                return !(value && *value == '0');
            }();
            if (!close_render(PassBreak::render_targets, carry_clears) || !cut_submission()) return false;
            state.targets = payload<SetRenderTargets>(record);
            if (state.targets.count < 0 || state.targets.count > 8)
                return fail(ReplayErrorCode::invalid_command,
                            "render-target count exceeds the attachment limit");
            state.target_owners = {};
            for (std::int32_t index = 0; index != state.targets.count; ++index) {
                if (!state.targets.color[index]) continue;
                state.target_owners[index] =
                    std::make_shared<const TextureState>(*state.targets.color[index]);
                state.targets.color[index] = state.target_owners[index].get();
            }
            state.depth_owner.reset();
            if (state.targets.depth) {
                state.depth_owner = std::make_shared<const TextureState>(*state.targets.depth);
                state.targets.depth = state.depth_owner.get();
            }
            refresh_attachment_cpu_bounds();
            return true;
        case Op::clear_color: return clear_bound_color(payload<ClearColor>(record));
        case Op::clear_color_integer: {
            const auto command = payload<ClearColorInteger>(record);
            if (!resolve_predicate()) return false;
            if (!state.render_enabled || !(command.mask & 0xf)) return true;
            if (command.index < 0 || command.index >= state.targets.count || command.index >= 8 ||
                !state.targets.color[command.index])
                return fail(ReplayErrorCode::invalid_command,
                            "integer color clear names an unbound target");
            const auto* texture = state.targets.color[command.index];
            const auto* view = state.targets.has_color_view[command.index]
                ? &state.targets.color_views[command.index] : nullptr;
            id<MTLTexture> resident = texture_for_use(*texture, view);
            if (!resident)
                return fail(ReplayErrorCode::missing_resource,
                            "integer color clear target is unavailable");
            NVNcopyRegion region{0, 0, 0, static_cast<int>(resident.width),
                                 static_cast<int>(resident.height), 1};
            if (state.scissor_count) {
                const auto scissor = clip_scissor(state.scissors[0], resident.width, resident.height);
                const auto x = std::min<NSUInteger>(scissor.x, resident.width);
                const auto y = std::min<NSUInteger>(scissor.y, resident.height);
                region = {static_cast<int>(x), static_cast<int>(y), 0,
                    static_cast<int>(std::min<NSUInteger>(scissor.width, resident.width - x)),
                    static_cast<int>(std::min<NSUInteger>(scissor.height, resident.height - y)), 1};
            }
            if (!region.width || !region.height) return true;
            std::array<std::uint32_t, 4> bits{};
            std::copy(std::begin(command.rgba), std::end(command.rgba), bits.begin());
            return encode_clear_plan(*texture, view, region,
                                     static_cast<std::uint8_t>(command.mask & 0xf),
                                     bits, false, 0, false, 0, 0);
        }
        case Op::clear_depth_stencil:
            return clear_bound_depth(payload<ClearDepthStencil>(record));
        case Op::viewports: {
            dirty.fixed |= 1u << 8;
            const auto command = payload<Viewports>(record);
            if (command.first < 0 || command.count < 0 || command.first > 16 ||
                command.count > 16 - command.first)
                return fail(ReplayErrorCode::invalid_command, "viewport range is invalid");
            for (std::int32_t index = 0; index != command.count; ++index) {
                const auto& value = command.values[index];
                const auto destination = std::size_t(command.first + index);
                const auto near_value = state.viewports[destination].znear;
                const auto far_value = state.viewports[destination].zfar;
                state.viewports[command.first + index] = {
                    value[0], value[1], value[2], value[3], near_value, far_value};
            }
            state.viewport_count = std::max<std::size_t>(
                state.viewport_count, command.first + command.count);
            return true;
        }
        case Op::scissor: {
            dirty.fixed |= 1u << 9;
            const auto command = payload<Scissor>(record);
            if (command.width < 0 || command.height < 0)
                return fail(ReplayErrorCode::invalid_command, "scissor rectangle is invalid");
            state.scissors[0] = command;
            state.scissor_count = 1;
            return true;
        }
        case Op::scissors: {
            dirty.fixed |= 1u << 9;
            const auto command = payload<Scissors>(record);
            if (command.first < 0 || command.count < 0 || command.first > 16 ||
                command.count > 16 - command.first)
                return fail(ReplayErrorCode::invalid_command, "scissor array range is invalid");
            for (std::int32_t index = 0; index != command.count; ++index) {
                const auto* value = command.values[index];
                if (value[2] < 0 || value[3] < 0)
                    return fail(ReplayErrorCode::invalid_command,
                                "scissor array contains a negative rectangle");
                state.scissors[command.first + index] = {
                    value[0], value[1], value[2], value[3]};
            }
            state.scissor_count = std::max<std::size_t>(
                state.scissor_count, command.first + command.count);
            return true;
        }
        case Op::depth_range: {
            dirty.fixed |= 1u << 8;
            const auto command = payload<DepthRange>(record);
            state.depth_near = command.near_value;
            state.depth_far = command.far_value;
            state.viewports[0].znear = state.depth_near;
            state.viewports[0].zfar = state.depth_far;
            return true;
        }
        case Op::depth_ranges: {
            dirty.fixed |= 1u << 8;
            const auto command = payload<DepthRanges>(record);
            if (command.first < 0 || command.count < 0 || command.first > 16 ||
                command.count > 16 - command.first)
                return fail(ReplayErrorCode::invalid_command,
                            "depth-range array range is invalid");
            for (std::int32_t index = 0; index != command.count; ++index) {
                state.viewports[command.first + index].znear = command.values[index][0];
                state.viewports[command.first + index].zfar = command.values[index][1];
            }
            return true;
        }
        case Op::bind_program: {
            const auto command = payload<BindProgram>(record);
            constexpr auto logical_stage_bits = (1u << logical_binding_stage_count) - 1;
            if (command.stages < 0 || (static_cast<std::uint32_t>(command.stages) & ~logical_stage_bits))
                return fail(ReplayErrorCode::invalid_command, "program binding contains an invalid logical stage mask");
            if (!command.stages) return true;
            std::shared_ptr<const ResolvedProgram> program;
            if (command.program) program = retain_program(*command.program);
            if (command.program && !program)
                return fail(ReplayErrorCode::missing_program,
                            "bound program serial has no resolved artifacts");
            const auto previous_compute = state.stage_programs[NVN_SHADER_STAGE_COMPUTE];
            const bool changed = state.bind(command, program);
            if (previous_compute != state.stage_programs[NVN_SHADER_STAGE_COMPUTE]) invalidate_compute_driver();
            remember_mesh_bind(command);
            // Rebinding the program a stage already uses changes no binding.
            if (changed) {
                invalidate_args_all();
                dirty.pipeline = true;
            }
            return true;
        }
        case Op::bind_state: {
            const auto command = payload<BindState>(record);
            switch (command.kind) {
                case 0: {
                    BlendStateBits value{};
                    std::memcpy(&value, command.bytes, sizeof value);
                    if (value.target >= state.blend.size())
                        return fail(ReplayErrorCode::invalid_command,
                                    "blend state target exceeds the render-target limit");
                    std::uint64_t bits = 0;
                    std::memcpy(&bits, command.bytes, sizeof bits);
                    if ((bits >> 37) & 0x3fu)
                        return fail(ReplayErrorCode::unsupported, "advanced blend modes require shader support");
                    if (replace_state(&state.blend[value.target], &value, sizeof value))
                        dirty.pipeline = true;
                    break;
                }
                case 1:
                    if (replace_state(&state.color, command.bytes, sizeof state.color))
                        dirty.pipeline = true;
                    break;
                case 2:
                    if (replace_state(&state.channel_mask, command.bytes, sizeof state.channel_mask))
                        dirty.pipeline = true;
                    break;
                case 3: {
                    MultisampleStateBits value{};
                    std::memcpy(&value, command.bytes, sizeof value);
                    if (value.sample_locations_enable != state.multisample.sample_locations_enable ||
                        value.sample_locations_grid != state.multisample.sample_locations_grid ||
                        value.sample_location_mask != state.multisample.sample_location_mask ||
                        std::memcmp(value.sample_locations, state.multisample.sample_locations,
                                    sizeof value.sample_locations) != 0) close_render(PassBreak::sample_locations);
                    if (replace_state(&state.multisample, &value, sizeof value))
                        dirty.pipeline = true;
                    break;
                }
                case 4:
                    if (replace_state(&state.polygon, command.bytes, sizeof state.polygon))
                        dirty.fixed |= (1u << 2) | (1u << 3) | (1u << 4) | (1u << 6);
                    break;
                case 5:
                    if (replace_state(&state.depth_stencil, command.bytes, sizeof state.depth_stencil))
                        dirty.fixed |= 1u;
                    break;
                default:
                    return fail(ReplayErrorCode::invalid_command,
                                "state binding kind is invalid");
            }
            return true;
        }
        case Op::bind_vertex_attributes: {
            const auto command = payload<BindVertexAttributes>(record);
            if (command.first < 0 || command.count < 0 || command.first > 16 ||
                command.count > 16 - command.first)
                return fail(ReplayErrorCode::invalid_command,
                            "vertex attribute range is invalid");
            if (replace_state(state.attributes.data() + command.first, command.states,
                              command.count * sizeof(VertexAttributeStateBits))) {
                dirty.vertices = true;
                dirty.pipeline = true;
            }
            return true;
        }
        case Op::bind_vertex_streams: {
            const auto command = payload<BindVertexStreams>(record);
            if (command.first < 0 || command.count < 0 || command.first > 16 ||
                command.count > 16 - command.first)
                return fail(ReplayErrorCode::invalid_command,
                            "vertex stream range is invalid");
            if (replace_state(state.streams.data() + command.first, command.states,
                              command.count * sizeof(VertexStreamStateBits))) {
                dirty.vertices = true;
                dirty.pipeline = true;
            }
            return true;
        }
        case Op::bind_vertex_buffers: {
            const auto command = payload<BindVertexBuffers>(record);
            if (command.first < 0 || command.count < 0 || command.first > 16 ||
                command.count > 16 - command.first)
                return fail(ReplayErrorCode::invalid_command,
                            "vertex buffer range is invalid");
            for (std::int32_t index = 0; index < command.count; ++index)
                if (!prepare_uniform_access(command.ranges[index].address, command.ranges[index].size)) return false;
            if (replace_state(state.vertex_buffers.data() + command.first, command.ranges,
                              command.count * sizeof(NVNbufferRange))) dirty.vertices = true;
            return true;
        }
        case Op::bind_uniform_buffer:
        case Op::bind_storage_buffer: {
            const SiteScope site_scope(*this, TransferSite::storage_bind);
            const auto command = payload<BindBuffer>(record);
            if (command.stage < 0 || command.stage >= int(logical_binding_stage_count) || command.index < 0 ||
                command.index >= int(buffer_slots))
                return fail(ReplayErrorCode::invalid_command,
                            "shader buffer binding range is invalid");
            if (record.op == Op::bind_storage_buffer) {
                if (!prepare_uniform_access(command.address, command.size)) return false;
            } else {
                for (const auto& version : uniform_versions)
                    if (command.address < version.end && version.begin < command.address + command.size &&
                        (command.address < version.begin || command.address + command.size > version.end)) {
                        if (!flush_uniforms()) return false;
                        break;
                    }
            }
            auto& target = record.op == Op::bind_uniform_buffer
                ? state.uniform[command.stage][command.index]
                : state.storage[command.stage][command.index];
            if (target.address != command.address || target.size != command.size) {
                target = {command.address, command.size};
                if (const auto active = state.execution_stage(std::size_t(command.stage))) {
                    if (record.op == Op::bind_uniform_buffer)
                        invalidate_uniform_binding(*active, std::uint32_t(command.index));
                    else invalidate_args(*active);
                    if (record.op == Op::bind_storage_buffer || *active == NVN_SHADER_STAGE_COMPUTE) {
                        ++bindings[*active].driver_generation[0];
                        bindings[*active].windows[0] = {};
                    }
                }
            }
            return true;
        }
        case Op::bind_texture_prefix: {
            const auto command = payload<BindTexturePrefix>(record);
            if (command.stage < 0 || command.stage >= int(logical_binding_stage_count) ||
                command.stage == NVN_SHADER_STAGE_COMPUTE ||
                (command.bytes != 4 && command.bytes != 8))
                return fail(ReplayErrorCode::invalid_command, "texture handle prefix update is invalid");
            auto& prefix = state.texture_prefix[command.stage];
            const auto value = command.bytes == 4
                ? (prefix & 0xffffffff00000000ull) | static_cast<std::uint32_t>(command.handle) : command.handle;
            if (command.sampler_registrations && command.sampler_registrations != state.sampler_registrations) {
                state.sampler_registrations = command.sampler_registrations;
                invalidate_args_all();
            }
            if (prefix != value) {
                prefix = value;
                if (const auto active = state.execution_stage(std::size_t(command.stage))) {
                    invalidate_args(*active);
                    ++bindings[*active].driver_generation[2];
                    bindings[*active].windows[2] = {};
                }
            }
            return true;
        }
        case Op::bind_textures:
        case Op::bind_samplers:
        case Op::bind_images: {
            const auto command = payload<BindHandles>(record);
            const std::size_t limit = record.op == Op::bind_images
                ? image_slots : record.op == Op::bind_samplers ? sampler_slots : bound_texture_slots;
            if (command.stage < 0 || command.stage >= int(logical_binding_stage_count) || command.first < 0 ||
                command.count < 0 || std::size_t(command.first) > limit ||
                std::size_t(command.count) > limit - std::size_t(command.first))
                return fail(ReplayErrorCode::invalid_command,
                            "shader handle binding range is invalid");
            bool changed = false;
            for (std::int32_t index = 0; index != command.count; ++index) {
                const auto slot = std::size_t(command.first + index);
                auto& handle = record.op == Op::bind_textures
                    ? state.textures[command.stage][slot]
                    : record.op == Op::bind_samplers
                        ? state.samplers[command.stage][slot]
                        : state.images[command.stage][slot];
                if (handle != command.handles[index]) {
                    handle = command.handles[index];
                    changed = true;
                }
            }
            // A binding record stamps the sampler registrations made before it
            // (deferred bindings carry none).
            if (record.op != Op::bind_images && command.sampler_registrations &&
                command.sampler_registrations != state.sampler_registrations) {
                state.sampler_registrations = command.sampler_registrations;
                invalidate_args_all();
            }
            if (changed) {
                if (const auto active = state.execution_stage(std::size_t(command.stage))) {
                    invalidate_args(*active);
                    ++bindings[*active].driver_generation[2];
                    bindings[*active].windows[2] = {};
                }
                if (command.stage == NVN_SHADER_STAGE_COMPUTE && compute_layout()) invalidate_compute_driver();
            }
            return true;
        }
        case Op::stencil: {
            const auto command = payload<Stencil>(record);
            const bool front = command.face == NVN_FACE_FRONT ||
                               command.face == NVN_FACE_FRONT_AND_BACK;
            const bool back = command.face == NVN_FACE_BACK ||
                              command.face == NVN_FACE_FRONT_AND_BACK;
            if (!front && !back)
                return fail(ReplayErrorCode::invalid_command,
                            "stencil update has an invalid face selection");
            dirty.fixed |= command.which == 2 ? (1u << 1) : 1u;
            if (command.which == 0) {
                if (front) state.stencil_front_read = command.value_mask;
                if (back) state.stencil_back_read = command.value_mask;
            } else if (command.which == 1) {
                if (front) state.stencil_front_write = command.mask;
                if (back) state.stencil_back_write = command.mask;
            } else if (command.which == 2) {
                if (front) state.stencil_front_ref = command.reference;
                if (back) state.stencil_back_ref = command.reference;
            } else {
                return fail(ReplayErrorCode::invalid_command,
                            "stencil update kind is invalid");
            }
            return true;
        }
        case Op::polygon_offset: {
            dirty.fixed |= 1u << 6;
            const auto command = payload<PolygonOffset>(record);
            state.polygon_factor = command.factor;
            state.polygon_units = command.units;
            state.polygon_clamp = command.clamp;
            return true;
        }
        case Op::draw: return draw(payload<Draw>(record));
        case Op::draw_indexed: return draw_indexed(payload<DrawIndexed>(record));
        case Op::multi_draw_indirect:
        case Op::multi_draw_indexed_indirect: {
            if (!resolve_predicate()) return false;
            if (!state.render_enabled) return true;
            const bool indexed = record.op == Op::multi_draw_indexed_indirect;
            const auto arrays = indexed ? MultiDrawIndirect{} : payload<MultiDrawIndirect>(record);
            const auto elements = indexed ? payload<MultiDrawIndexedIndirect>(record) : MultiDrawIndexedIndirect{};
            const auto maximum = indexed ? elements.max_draw_count : arrays.max_draw_count;
            const auto parameter = indexed ? elements.parameter_buffer : arrays.parameter_buffer;
            const auto arguments = indexed ? elements.indirect_buffer : arrays.indirect_buffer;
            const auto size = indexed ? sizeof(MTLDrawIndexedPrimitivesIndirectArguments)
                                      : sizeof(MTLDrawPrimitivesIndirectArguments);
            const auto specified_stride = indexed ? elements.stride : arrays.stride;
            const auto stride = specified_stride ? specified_stride : static_cast<std::int64_t>(size);
            if (maximum < 0 || stride < static_cast<std::int64_t>(size) || (stride & 3))
                return fail(ReplayErrorCode::invalid_command, "multi-draw count or stride is invalid");
            if (!synchronize_gpu()) return false;
            const auto source = buffer_at(parameter, sizeof(std::uint32_t));
            if (!source.buffer.contents)
                return fail(ReplayErrorCode::missing_resource, "multi-draw count is unavailable");
            std::uint32_t count = 0;
            std::memcpy(&count, static_cast<const std::byte*>(source.buffer.contents) + source.offset, sizeof count);
            count = std::min(count, static_cast<std::uint32_t>(maximum));
            if (count && std::uint64_t(stride) > (UINT64_MAX - arguments) / count)
                return fail(ReplayErrorCode::invalid_command, "multi-draw argument range overflows");
            for (std::uint32_t index = 0; index != count; ++index) {
                const auto address = arguments + index * stride;
                if (indexed) {
                    if (!draw_indexed_indirect({elements.primitive, elements.index_type, elements.index_buffer, address}))
                        return false;
                } else if (!draw_indirect({arrays.primitive, address})) return false;
            }
            return true;
        }
        case Op::draw_indirect: return draw_indirect(payload<DrawIndirect>(record));
        case Op::draw_indexed_indirect:
            return draw_indexed_indirect(payload<DrawIndexedIndirect>(record));
        case Op::dispatch: return dispatch(payload<Dispatch>(record));
        case Op::dispatch_indirect:
            return dispatch_indirect(payload<DispatchIndirect>(record));
        case Op::barrier:
            if (payload<ScalarInt>(record).value) {
                ++feedback_epoch;
                dirty.arguments[NVN_SHADER_STAGE_FRAGMENT] = true;
            }
            // Keep depth-only writes on tile for subsequent depth tests. Any
            // later shader read of that depth closes the pass before drawing.
            // Other passes retain the conservative full encoder boundary.
            if (render && pass_depth_only && !color_aliases.any &&
                !backend.resources().texture_is_in_place(*state.targets.depth) &&
                (payload<ScalarInt>(record).value & ~0xff) == 0) {
                depth_barrier_allocation = render_descriptor.depthAttachment.texture;
                while (id<MTLTexture> parent = depth_barrier_allocation.parentTexture)
                    depth_barrier_allocation = parent;
                barrier_requested = true;
                invalidate_args_all();
                return true;
            }
            return barrier(PassBreak::barrier);
        case Op::set_texture_pool:
        case Op::set_sampler_pool:
            return true;
        case Op::downsample: {
            const auto command = payload<Downsample>(record);
            if (!command.source || !command.destination)
                return fail(ReplayErrorCode::invalid_command, "downsample textures are missing");
            const auto source = texture_for_use(*command.source,
                command.has_source_view ? &command.source_view : nullptr);
            const auto destination = texture_for_use(*command.destination,
                command.has_destination_view ? &command.destination_view : nullptr);
            if (!source || !destination)
                return fail(ReplayErrorCode::missing_resource, "downsample textures are unavailable");
            if (source.sampleCount <= 1 || destination.sampleCount != 1 || source.width != destination.width ||
                source.height != destination.height || source.pixelFormat != destination.pixelFormat ||
                source.textureType != MTLTextureType2DMultisample || destination.textureType != MTLTextureType2D)
                return fail(ReplayErrorCode::unsupported, "downsample requires matching multisample and single-sample targets");
            const auto format = texture_format(command.has_source_view && command.source_view.sets_format
                ? command.source_view.format : command.source->description.format);
            if (!format || graphics::texture_format_info(*format).depth)
                return fail(ReplayErrorCode::unsupported, "depth downsample requires an explicit resolve filter");
            if (command.destination->memory && backend.resources().texture_is_in_place(*command.destination) &&
                uniform_cpu_conflict(reinterpret_cast<std::uintptr_t>(command.destination->memory),
                                     command.destination->size) && !flush_uniforms()) return false;
            barrier(PassBreak::downsample);
            auto* pass = [MTL4RenderPassDescriptor new];
            pass.colorAttachments[0].texture = source;
            pass.colorAttachments[0].resolveTexture = destination;
            pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
            pass.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
            make_resident(pending, source);
            make_resident(pending, destination);
            [pending.retained addObjectsFromArray:@[source, destination]];
            render = begin_render(pass, nil, nil);
            if (!render) return fail(ReplayErrorCode::allocation, "resolve encoder allocation failed");
            close_render(PassBreak::downsample);
            note_texture_write(*command.destination);
            note_writes();
            ++backend.mutable_statistics().copies;
            return true;
        }
        case Op::copy_buffer_to_texture:
            return copy_buffer_to_texture(payload<CopyBufferToTexture>(record));
        case Op::copy_texture_to_buffer:
            return copy_texture_to_buffer(payload<CopyBufferToTexture>(record));
        case Op::copy_texture_to_texture:
            return copy_texture_to_texture(payload<CopyTextureToTexture>(record));
        case Op::copy_stride:
            state.copy_row_stride = payload<CopyStride>(record).stride;
            if (state.copy_row_stride < 0)
                return fail(ReplayErrorCode::unsupported,
                            "negative texture copy row strides are unsupported");
            return true;
        case Op::copy_image_stride:
            state.copy_image_stride = payload<CopyStride>(record).stride;
            if (state.copy_image_stride < 0)
                return fail(ReplayErrorCode::unsupported,
                            "negative texture copy image strides are unsupported");
            return true;
        case Op::blend_color:
            dirty.fixed |= 1u << 7;
            std::memcpy(state.blend_color.data(), payload<BlendColor>(record).rgba,
                        sizeof(state.blend_color));
            return true;
        case Op::clear_texture: return clear_texture(payload<ClearTexture>(record));
        case Op::depth_clamp:
            dirty.fixed |= 1u << 5;
            state.depth_clamp = payload<DepthClamp>(record).enable != 0;
            return true;
        case Op::line_width:
            state.line_width = payload<LineWidth>(record).width;
            return true;
        case Op::point_size:
            state.point_size = payload<PointSize>(record).size;
            return true;
        case Op::primitive_restart: {
            const auto command = payload<PrimitiveRestart>(record);
            state.primitive_restart = command.enable != 0;
            state.restart_index = command.index;
            return true;
        }
        case Op::rasterizer_discard:
            state.rasterizer_discard = payload<ScalarInt>(record).value != 0;
            dirty.pipeline = true;
            return true;
        case Op::sample_mask:
            state.sample_mask = static_cast<std::uint32_t>(payload<ScalarInt>(record).value);
            if (state.sample_mask != UINT32_MAX)
                return fail(ReplayErrorCode::unsupported, "partial sample masks require fragment shader support");
            return true;
        case Op::alpha_ref:
            // Alpha comparison is validated with color state when a pipeline is built.
            return true;
        case Op::conservative_raster:
            if (payload<ConservativeRaster>(record).enable)
                return fail(ReplayErrorCode::unsupported, "conservative rasterization is unavailable");
            return true;
        case Op::conservative_raster_dilate:
            if (payload<ScalarFloat>(record).value != 0)
                return fail(ReplayErrorCode::unsupported, "conservative raster dilation is unavailable");
            return true;
        case Op::subpixel_precision_bias: {
            const auto command = payload<SubpixelPrecisionBias>(record);
            if (command.x_bits || command.y_bits)
                return fail(ReplayErrorCode::unsupported, "subpixel precision bias is unavailable");
            return true;
        }
        case Op::shader_scratch_memory:
            // Native pipelines allocate their own spill memory.
            return true;
        case Op::program_subroutines:
            if (payload<ProgramSubroutines>(record).count)
                return fail(ReplayErrorCode::unsupported, "shader subroutines require specialized artifacts");
            return true;
        case Op::render_enable_conditional: {
            const auto command = payload<RenderEnableConditional>(record);
            if (trace_submission_id) ++trace_conditional;
            trace_pass_event("conditional", "begin", command.address, command.mode);
            if (command.mode != NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL &&
                command.mode != NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL)
                return fail(ReplayErrorCode::invalid_command, "conditional render mode is invalid");
            // A previous block ends here.
            predicate.arguments = nullptr;
            predicate.job = 0;
            state.predicate_pending = false;
            if (reports) {
                // Each record is 16 bytes; the comparison uses the counter field.
                std::uint64_t left = 0, right = 0;
                const bool left_known = reports->value(command.address, left);
                const bool right_known = reports->value(command.address + 16, right);
                const bool if_equal = command.mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL;
                const bool known = left_known && right_known;
                const bool skip = known && (if_equal ? left != right : left == right);
                state.render_enabled = !skip;
                note_conditional(false, if_equal);
                log_conditional(command.address, if_equal, left, left_known, right, right_known, !skip);
                return true;
            }
            bool handled = false;
            if (!gpu_predicate(command, handled)) return false;
            note_conditional(handled, command.mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL);
            if (handled) {
                state.render_enabled = true;
                return true;
            }
            if (!synchronize_gpu()) return false;
            if (!read_predicate(command.mode, command.address)) return false;
            log_conditional(command.address, command.mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL,
                            0, false, 0, false, state.render_enabled);
            return true;
        }
        case Op::render_enable:
            trace_pass_event("conditional", "end", payload<RenderEnable>(record).enable);
            predicate.arguments = nullptr;
            predicate.job = 0;
            state.predicate_pending = false;
            state.render_enabled = payload<RenderEnable>(record).enable != 0;
            return true;
        case Op::discard_color:
        case Op::discard_depth_stencil:
            // Discard permits contents to become undefined. Ending the pass
            // and preserving them is one valid outcome of that contract.
            close_render(PassBreak::discard);
            return true;
        case Op::viewport_swizzles: {
            const auto command = payload<ViewportSwizzles>(record);
            if (command.first < 0 || command.count < 0 || command.first > 16 ||
                command.count > 16 - command.first)
                return fail(ReplayErrorCode::invalid_command,
                            "viewport-swizzle range is invalid");
            constexpr std::array<std::int32_t, 4> identity{
                NVN_VIEWPORT_SWIZZLE_POSITIVE_X, NVN_VIEWPORT_SWIZZLE_POSITIVE_Y,
                NVN_VIEWPORT_SWIZZLE_POSITIVE_Z, NVN_VIEWPORT_SWIZZLE_POSITIVE_W};
            for (std::int32_t index = 0; index != command.count; ++index)
                if (!std::equal(identity.begin(), identity.end(), command.values[index]))
                    return fail(ReplayErrorCode::unsupported,
                                "non-identity viewport swizzles are unsupported");
            return true;
        }
        case Op::depth_bounds: {
            // Metal has no depth-bounds test. An enabled test is approximated as
            // disabled: fragments it would cull are shaded too, which costs
            // fill and can brighten pixels outside the bounds, instead of
            // abandoning the submission. The host log names the first bounds.
            const auto command = payload<DepthBounds>(record);
            if (command.enable && (command.near_value > 0.0f || command.far_value < 1.0f)) {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true))
                    armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN: depth-bounds test [%g, %g] is approximated as disabled\n",
                                 command.near_value, command.far_value);
            }
            return true;
        }
        case Op::draw_mesh_tasks: {
            const auto command = payload<DrawMeshTasks>(record);
            if (command.first < 0 || command.count < 0)
                return fail(ReplayErrorCode::invalid_command, "mesh draw contains a negative range");
            return draw_mesh(std::uint32_t(command.first), std::uint32_t(command.count), 0);
        }
        case Op::draw_mesh_tasks_indirect:
            return draw_mesh(0, 0, payload<DrawMeshTasksIndirect>(record).address);
        case Op::geometry_processing_mode: {
            const auto mode = payload<ScalarInt>(record).value;
            if (mode < 0 || mode > 2)
                return fail(ReplayErrorCode::unsupported, "geometry processing mode is unsupported");
            state.geometry_mode = mode;
            state.select();
            // Every setter carries followup program/resource state, even if
            // its mode value repeats. Drop physical windows across bank swaps.
            for (auto& bound : bindings) {
                bound.windows.fill({});
                for (auto& generation : bound.driver_generation) ++generation;
            }
            invalidate_args_all();
            dirty.pipeline = true;
            return true;
        }
        case Op::push_constants: {
            const auto command = payload<PushConstants>(record);
            if (command.stage < 0 || command.stage >= int(logical_binding_stage_count) || command.bank < 3 ||
                command.bank >= 32 || command.offset > push_constant_bytes ||
                command.size > push_constant_bytes - command.offset)
                return fail(ReplayErrorCode::invalid_command, "push-constant record is invalid");
            const auto stage = std::size_t(command.stage);
            std::memcpy(state.push_constants[stage].data() + command.offset, command.data, command.size);
            state.push_bank[stage] = command.bank;
            if (const auto active = state.execution_stage(stage)) {
                ++bindings[*active].driver_generation[3];
                if (command.bank < bindings[*active].windows.size()) bindings[*active].windows[command.bank] = {};
                if (*active == NVN_SHADER_STAGE_COMPUTE && compute_layout()) invalidate_compute_driver();
                bindings[*active].uniform_slots |= 1u << command.bank;
                invalidate_args(*active);
            }
            return true;
        }
        case Op::sampler_registrations: {
            const auto command = payload<SamplerRegistrations>(record);
            if (command.value != state.sampler_registrations) {
                state.sampler_registrations = command.value;
                invalidate_args_all();
            }
            return true;
        }
        case Op::debug_marker:
            return true;
        case Op::unsupported: {
            const auto command = payload<Unsupported>(record);
            return fail(ReplayErrorCode::unsupported,
                        "recorded NVN entry has no replay implementation (entry " +
                            std::to_string(command.entry) + ")");
        }
        case Op::begin:
        case Op::end:
        case Op::jump:
        case Op::call:
        case Op::bind_textures_deferred:
            return fail(ReplayErrorCode::invalid_command,
                        "structural command escaped validated traversal");
        default:
            return fail(ReplayErrorCode::unsupported,
                        "recorded command is not supported by Metal replay (op " +
                            std::to_string(static_cast<std::uint32_t>(record.op)) + ")");
    }
}

bool Encoder::finish() {
    const SiteScope site_scope(*this, TransferSite::finish);
    if (capture_retirement_needed && !retire_captures_if_full()) return false;
    if (predicate_open()) {
        predicate.arguments = nullptr;
        predicate.job = 0;
        state.predicate_pending = true;
        state.predicate_mode = predicate.mode;
        state.predicate_address = predicate.address;
    }
    if (!flush_uniforms()) return false;
    note_writes();
    close_render(PassBreak::finish);
    report_pass_breaks(pending.fence);
    close_transfer();
    trace_pass_event("submission", "end", pending.counters.passes, pending.counters.draws);
    if (state.samples_active && counter_begin != visibility_cursor) {
        CompletionEffect accumulate{};
        accumulate.visibility_begin = counter_begin;
        accumulate.visibility_end = visibility_cursor;
        pending.effects.push_back(accumulate);
        counter_begin = visibility_cursor;
    }
    return !failed;
}

} // namespace

std::unique_ptr<MetalReplayBackend>
create_replay_backend(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                      ReplayResources& resources, ReplaySchedulingPolicy scheduling) {
    if (!device || !queue || ![device supportsFamily:MTLGPUFamilyMetal4] ||
        !device.hasUnifiedMemory) return {};
    return std::make_unique<ReplayImpl>(device, queue,
        std::shared_ptr<ReplayResources>(&resources, [](ReplayResources*) {}), scheduling);
}

#if __has_include("nvn_resources.h")
namespace {
class ResourceLookup final : public ReplayResources {
public:
    explicit ResourceLookup(Resources& resources) : resources_(resources) {}
    // Resources registers only shared, zero-copy pools at their CPU addresses.
    bool buffer_addresses_are_cpu() const override { return true; }
    void set_completion_hook(std::function<bool()> hook) override {
        resources_.set_completion_hook(std::move(hook));
    }
    BufferSpan buffer_at(std::uint64_t address, std::uint64_t size) override {
        return resources_.buffer_at(address, size);
    }
    BufferAddress buffer_address(std::uint64_t address, std::uint64_t size) override {
        return resources_.buffer_address(address, size);
    }
    std::uint64_t claim_guest_memory_pools(std::uint64_t known,
                                           std::vector<Resources::GuestMemoryPool>& pools) override {
        return resources_.claim_guest_memory_pools(known, pools);
    }
    Resources::GuestMemoryAdmission guest_memory_admission() const override {
        return resources_.guest_memory_admission();
    }
    bool address_in_unaddressed_pool(std::uint64_t address) const override {
        return resources_.address_in_unaddressed_pool(address);
    }
    std::string describe_guest_address(std::uint64_t address) const override {
        return resources_.describe_guest_address(address);
    }
    id<MTLTexture> texture(const nvn::TextureState& state, const nvn::TextureViewState* view) override {
        return resources_.texture(state, view);
    }
    std::string take_texture_failure() override { return Resources::take_texture_failure(); }
    AttachmentFormat peek_attachment(const nvn::TextureState& state,
                                     const nvn::TextureViewState* view) override {
        const auto peeked = resources_.peek_attachment(state, view);
        return {peeked.format, peeked.samples};
    }
    id<MTLTexture> texture_for_id(std::int32_t id) override { return resources_.texture_for_id(id); }
    id<MTLTexture> upgrade_for_image_write(std::uint64_t serial, __strong id<MTLTexture>& previous) override {
        return resources_.upgrade_for_image_write(serial, previous);
    }
    id<MTLTexture> upgrade_for_image_atomic(std::uint64_t serial, bool is_signed,
                                            __strong id<MTLTexture>& previous) override {
        return resources_.upgrade_for_image_atomic(serial, is_signed, previous);
    }
    bool image_atomic_capable(std::uint64_t serial) override {
        return resources_.image_atomic_capable(serial);
    }
    bool image_atomic_needs_upgrade(std::uint64_t serial, bool is_signed) override {
        return resources_.image_atomic_needs_upgrade(serial, is_signed);
    }
    id<MTLTexture> texture_for_id_stamped(std::int32_t id, DescriptorStamp& stamp) override {
        return resources_.texture_for_id(id, stamp);
    }
    id<MTLTexture> image_base_for_id(std::int32_t id, DescriptorStamp& stamp) override {
        return resources_.image_base_for_id(id, stamp);
    }
    id<MTLTexture> raw_image_view(std::int32_t texture_id, id<MTLTexture> texture) override {
        return resources_.raw_image_view(texture_id, texture);
    }
    bool texture_stamp_current(std::int32_t id, const DescriptorStamp& stamp) const override {
        return resources_.texture_stamp_current(id, stamp);
    }
    std::string take_sampler_failure() override { return resources_.take_sampler_failure(); }
    std::optional<std::uint32_t> sampler_pinned_level(std::int32_t id, std::uint64_t registrations) const override {
        return resources_.sampler_pinned_level(id, registrations);
    }
    CoverageSampler coverage_sampler_for_id(std::int32_t id, std::uint64_t registrations) override {
        return resources_.coverage_sampler_for_id(id, registrations);
    }
    id<MTLSamplerState> sampler_for_id(std::int32_t id) override { return resources_.sampler_for_id(id); }
    id<MTLSamplerState> sampler_for_id(std::int32_t id, std::uint64_t registrations) override {
        return resources_.sampler_for_id(id, registrations);
    }
    id<MTLTexture> fitted_texture(std::int32_t texture_id, id<MTLTexture> texture,
                                  MTLTextureType type, bool typed_handle, MTLPixelFormat empty_format) override {
        return resources_.fitted_texture(texture_id, texture, type, typed_handle, empty_format);
    }
    void begin_walk() override { resources_.begin_walk(); }
    void end_walk() override { resources_.end_walk(); }
    std::shared_ptr<Metal4SubmissionOwner> submission_owner() const override { return resources_.submission_owner(); }
    void set_upload_publication(UploadPublication publication) override { resources_.set_upload_publication(std::move(publication)); }
    bool publish_uploads() override { return resources_.publish_uploads(); }
    bool flush_uploads() override { return resources_.flush_uploads(); }
    void commit_uploads(UploadFence& fence) override { resources_.commit_uploads(fence); }
    bool uploads_failed(const UploadFence& fence) override { return resources_.uploads_failed(fence); }
    void note_gpu_write(const nvn::TextureState& state) override { resources_.note_gpu_write(state); }
    void note_completed_gpu_write(std::uint64_t address, std::uint64_t size) override {
        resources_.note_completed_gpu_write(address, size);
    }
    bool backing_feeds_texture(std::uint64_t address, std::uint64_t size) override {
        return resources_.backing_feeds_texture(address, size);
    }
    void note_gpu_write(std::uint64_t address, std::uint64_t size) override {
        resources_.note_gpu_write(address, size);
    }
    std::uint64_t gpu_write_generation() const override { return resources_.gpu_write_generation(); }
    void complete_gpu_writes(std::uint64_t generation) override { resources_.complete_gpu_writes(generation); }
    std::uint64_t registration_generation() const override {
        return resources_.registration_generation();
    }
    bool overlaps_texture_backing(std::uint64_t address, std::uint64_t size) const override {
        return resources_.overlaps_texture_backing(address, size);
    }
    bool texture_is_in_place(const nvn::TextureState& state) override {
        return resources_.texture_is_in_place(state);
    }
    LookupCounts texture_lookup_counts() const override {
        const auto counts = resources_.texture_lookup_counts();
        return {counts.lookups, counts.misses};
    }
    std::uint64_t registration_log_generation() const override {
        return resources_.registration_log_generation();
    }
    bool read_registration_note(std::uint64_t generation, RegistrationNote& note) const override {
        Resources::RegistrationNote source;
        if (!resources_.read_registration_note(generation, source)) return false;
        note.id = source.id;
        note.kind = source.kind;
        return true;
    }
    std::uint64_t storage_log_generation() const override { return resources_.storage_log_generation(); }
    bool read_storage_note(std::uint64_t generation, std::uint64_t& serial) const override {
        return resources_.read_storage_note(generation, serial);
    }
    std::vector<std::int32_t> descriptor_ids_for_serial(std::uint64_t serial) const override {
        return resources_.descriptor_ids_for_serial(serial);
    }
    bool texture_id_empty(std::int32_t id) const override { return resources_.texture_id_empty(id); }
    std::int32_t registered_texture_count() const override { return resources_.registered_texture_count(); }
    std::int32_t sampler_count() const override { return resources_.sampler_count(); }
    bool texture_registered(std::int32_t id) const override { return resources_.texture_registered(id); }
    bool texture_storage_made(std::int32_t id) const override { return resources_.texture_storage_made(id); }
    bool texture_storage_shared(std::int32_t id) const override { return resources_.texture_storage_shared(id); }
    bool texture_compressible(std::int32_t id) const override { return resources_.texture_compressible(id); }
    std::uint64_t texture_storage_size(std::int32_t id) const override {
        return resources_.texture_storage_size(id);
    }
    bool texture_storage_live(std::int32_t id) const override { return resources_.texture_storage_live(id); }
    std::uint64_t texture_serial(std::int32_t id) const override { return resources_.texture_serial(id); }
    std::uint64_t texture_descriptor_key(std::int32_t id) const override {
        return resources_.texture_descriptor_key(id);
    }
    bool texture_read_identity(std::int32_t id, TextureReadIdentity& identity) const override {
        return resources_.texture_read_identity(id, identity);
    }
    std::uint64_t texture_empty_generation() const override { return resources_.texture_empty_generation(); }
    bool texture_is_depth(std::int32_t id) const override { return resources_.texture_is_depth(id); }
    std::uint64_t texture_storage_key(std::int32_t id) const override { return resources_.texture_storage_key(id); }
    std::uint64_t texture_pool_generation() const override { return resources_.texture_pool_generation(); }
    void note_pool_depth_read(std::uint64_t serial, std::uint64_t storage_key) override {
        resources_.note_pool_depth_read(serial, storage_key);
    }
    bool pool_depth_read(std::uint64_t serial, std::uint64_t storage_key) const override {
        return resources_.pool_depth_read(serial, storage_key);
    }
private:
    Resources& resources_;
};
}
std::unique_ptr<MetalReplayBackend>
create_replay_backend(id<MTLDevice> device, id<MTL4CommandQueue> queue, Resources& resources,
                      ReplaySchedulingPolicy scheduling) {
    if (!device || !queue || ![device supportsFamily:MTLGPUFamilyMetal4] ||
        !device.hasUnifiedMemory) return {};
    if (!resources.submission_owner() || resources.submission_owner()->queue() != queue) return {};
    return std::make_unique<ReplayImpl>(device, queue, std::make_shared<ResourceLookup>(resources), scheduling);
}
#endif

} // namespace armigo::platform::metal::nvn_driver
