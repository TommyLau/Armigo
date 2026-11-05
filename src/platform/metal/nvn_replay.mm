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
#include <TargetConditionals.h>
#include "nvn_replay.h"
#include "nvn_draw_state.h"
#include "nvn_color_aliases.h"
#include "nvn_submission_segments.h"
#include "nvn_encode_ring.h"
#include "nvn_attachment_actions.h"
#include "nvn/flat_map.h"
#include <armigo/memory/write_tracker.h>
#include <armigo/platform/presentation_thread.h>
#include <armigo/platform/periodic_thread_budget.h>
#include <objc/runtime.h>

#include "../../nvn/commands.h"
#include "../../nvn/programs.h"
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
#include <armigo/shader/vertex_fetch.h>

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
#include <cstdio>
#include <cstring>
#include <limits>
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
        NSString* source = [[NSString alloc] initWithBytes:code.data() length:code.size()
                                                  encoding:NSUTF8StringEncoding];
        if (source) library = [device newLibraryWithSource:source options:options error:&error];
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
constexpr std::size_t buffer_slots = 16;
constexpr std::size_t texture_slots = 128;
constexpr std::size_t sampler_slots = 128;
constexpr std::size_t image_slots = 32;
// Several queue submissions can belong to one frame. Keep room for the next
// frame to encode while earlier submissions still own their GPU storage.
constexpr std::size_t submission_slots = 6;
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

struct ReplayState {
    SetRenderTargets targets{};
    std::array<std::shared_ptr<const TextureState>, 8> target_owners;
    std::shared_ptr<const TextureState> depth_owner;
    std::array<std::array<BoundBuffer, buffer_slots>, stage_count> uniform{};
    std::array<std::array<BoundBuffer, buffer_slots>, stage_count> storage{};
    std::array<std::array<std::uint64_t, texture_slots>, stage_count> textures{};
    std::array<std::array<std::uint64_t, sampler_slots>, stage_count> samplers{};
    std::array<std::array<std::uint64_t, image_slots>, stage_count> images{};
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
    std::array<std::shared_ptr<const ResolvedProgram>, stage_count> stage_programs;
    // Bit s: the program bound to stage s has that stage, but it failed to
    // translate (stage_programs[s] is then empty). Draws and dispatches that
    // would use it are refused rather than run without it.
    std::uint32_t failed_stages = 0;
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
    std::uint32_t render_if_equal, capacity;
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
    uint capacity;
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
    for (uint record = 0; record < job.capacity; ++record) job.arguments[record * 8 + 1] = 0;
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
    cpu_drain, finish, predicate, buffer_writes, count
};
constexpr const char* pass_break_names[] = {
    "render_targets", "transfer", "clear", "dispatch", "barrier", "wait_sync", "fence_sync", "counter",
    "depth_needed", "storage_after_depth_barrier", "color_alias", "depth_sampled", "sample_locations",
    "downsample", "discard", "skip", "cpu_drain", "finish", "predicate", "buffer_writes"};
static_assert(std::size(pass_break_names) == std::size_t(PassBreak::count));

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
};

PassBreakCounts& pass_break_counts() {
    static auto& counts = *new PassBreakCounts;
    return counts;
}

void note_pass_break(PassBreak why) {
    if (pass_reasons_enabled())
        pass_break_counts().reasons[std::size_t(why)].fetch_add(1, std::memory_order_relaxed);
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
};

class EncodeThread;

struct PendingSubmission : std::enable_shared_from_this<PendingSubmission> {
    std::uint64_t scheduling_period_ns = 0;
    bool preserve_encode_spin = false;
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
    std::uint64_t encode_begin_ns = 0, first_commit_ns = 0, final_commit_ns = 0;
    std::size_t committed_effects = 0;
    bool commands_open = false;
    std::shared_ptr<const CommandSubmission> captured;
    std::vector<UniformUpdate> uniform_updates;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> other_buffer_writes;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> completion_writes;
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
    id<MTLResidencySet> residency = nil;
    id<MTLBuffer> visibility = nil;
    NSMutableArray<id>* retained = [NSMutableArray array];
    ResidentTable residents;
    ResidentTable retains;
    std::vector<CompletionEffect> effects;
    std::vector<SyncState*> fenced_syncs;
    std::size_t notified_textures = 0, notified_buffers = 0;
    std::size_t timestamp_count = 0;
    std::vector<TextureState> texture_writes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> buffer_writes;
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
    stop = EncodeRing::wrap + 1, begin_commands, end_commands, commit, resident, call,
    render_begin, transfer_begin, end_encoder, encoder_barrier,
    render_pipeline, depth_stencil, stencil_reference, cull, winding, fill, depth_clip,
    depth_bias, blend_color, viewports, scissors, visibility,
    table_address, table_vertex, table_texture, table_sampler,
    draw, draw_indexed, compute_pipeline_compile, compute_table, dispatch, dispatch_indirect,
    copy_buffer, copy_buffer_texture, copy_texture_buffer, copy_texture, timestamp,
    copy_memory, predicate, draw_indirect_args, draw_indexed_indirect_args, render_timestamp, predicate_draw,
};

struct CommitRecord { void* options; PendingSubmission* owner; std::uint64_t segment; void* upload_event; std::uint64_t upload_value; };
struct ResidentRecord { PendingSubmission* owner; void* allocation; };
struct RenderBeginRecord { void* descriptor; void* vertex_table; void* fragment_table; };
struct BarrierRecord { std::uint64_t after, before; };
struct StencilRecord { std::uint32_t front, back; };
struct TableRecord { void* table; std::uint64_t value, stride; };
struct DrawRecord { std::uint32_t primitive, start, count, instances, base_instance; };
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
    // frame paced at a long period leaves the thread idle most of the time, and
    // spinning there only burns CPU; an unpaced or short frame keeps the spin
    // because a wake-up costs the producer a system call and delays the commit.
    static constexpr std::uint64_t spin_unpaced_ns = 500'000;
    static constexpr std::uint64_t spin_paced_min_period_ns = 8'000'000;
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

// With the encode thread the residency set belongs to it: the adds go as
// records ahead of the commit that makes them resident.
static void add_resident(PendingSubmission& pending, id<MTLAllocation> allocation) {
    if (pending.encode_thread)
        pending.encode_thread->put(EncodeOp::resident, 0,
                                   ResidentRecord{&pending, (__bridge void*)allocation});
    else
        [pending.residency addAllocation:allocation];
    [pending.retained addObject:allocation];
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

// The texture type a shader declares for a sampled resource, for the types a
// view can convert between: two-dimensional, two-dimensional array, cube and
// cube array.
static std::optional<MTLTextureType> declared_texture_type(
    const platform::CompiledShaderResource& resource) {
    if (!resource.texture_descriptor || resource.texture_descriptor->multisample) return std::nullopt;
    const auto& declared = *resource.texture_descriptor;
    switch (declared.dimension) {
    case platform::CompiledShaderTextureDimension::d2:
        return declared.array ? MTLTextureType2DArray : MTLTextureType2D;
    case platform::CompiledShaderTextureDimension::cube:
        return declared.array ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
    default:
        return std::nullopt;
    }
}

static void make_resident(PendingSubmission& pending, __unsafe_unretained id<MTLAllocation> allocation) {
    if (!allocation) return;
    if (!pending.residents.insert_unique((__bridge void*)allocation)) return;
    add_resident(pending, allocation);
    if (is_texture_allocation(allocation)) {
        for (id<MTLTexture> parent = ((id<MTLTexture>)allocation).parentTexture; parent;
             parent = parent.parentTexture) {
            if (!pending.residents.insert_unique((__bridge void*)parent)) break;
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
    key.rasterizer_discard = s.rasterizer_discard;
    key.samples = std::max<NSUInteger>(samples, 1);
    key.alpha_to_coverage = s.multisample.alpha_to_coverage;
    key.blend_enable = s.color.blend_enable;
    key.channel_mask = s.channel_mask.mask;
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
    id<MTL4ArgumentTable> compute_table = nil;
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
        std::byte* cpu = nullptr;
        // GPU writes recorded when the snapshot was taken: a later one over
        // the source makes the snapshot stale.
        std::size_t writes = 0;
    };
    // A stage owns its table shadow, reflected read extents and write list.
    // All resolved addresses and uploads live only for this submission.
    struct StageBindings {
        const platform::CompiledGraphicsStage* shader = nullptr;
        std::array<ArgumentSlot, 32> buffers{};
        std::array<ArgumentSlot, texture_slots> textures{};
        std::array<ArgumentSlot, sampler_slots> samplers{};
        std::array<DriverBlock, 3> driver{};
        std::array<std::uint64_t, 3> driver_generation{1, 1, 1};
        std::array<WindowUpload, buffer_slots + 3> windows{};
        std::array<std::uint32_t, buffer_slots + 3> window_extents{};
        std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
        bool writes_buffers = false;
        // The words of hardware constant buffers the shader's texture handles
        // and storage descriptors are read from.
        struct ContentRead { std::uint32_t slot, offset, length; };
        std::vector<ContentRead> content_reads;
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
    void invalidate_args_all() { dirty.arguments.fill(true); }
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
        for (std::size_t stage = 0; stage < stage_count; ++stage) dirty.uniforms[stage] = true;
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
    std::unordered_map<std::uint64_t, std::uint32_t> uniform_words;

    Encoder(ReplayImpl& owner, PendingSubmission& submission, const ReplayState& initial)
        : backend(owner), pending(submission), queue(submission.encode_thread), state(initial) {
        dirty.arguments.fill(true);
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
    void end_encoder(id encoder, MTLStages after);
    void encoder_barrier(id encoder, MTLStages after, MTLStages before);
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
    Op current_op_ = Op::end;
    std::uint64_t trace_submission_id = 0;
    std::uint64_t trace_encoder_sequence = 0, trace_active_encoder = 0;
    std::uint64_t trace_conditional = 0;
    std::size_t finalized_visibility = 0;
    std::size_t pass_buffer_writes_begin = 0;
    bool pass_storage_writes = false;
    void trace_pass_event(const char* event, const char* reason,
                          std::uint64_t a = 0, std::uint64_t b = 0,
                          std::uint64_t c = 0, std::uint64_t d = 0) const;
    bool close_render(PassBreak why, bool keep_clears = false);
    bool close_transfer();
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
    };
    // Prior update bytes remain owned until their submission fence retires.
    std::vector<std::shared_ptr<PendingSubmission>> previous_submissions;
    std::uint64_t retired_fence = 0;
    // Samplers resolve as registered when the current draw was recorded
    // (as registered now before any stamp).
    std::uint64_t draw_sampler_registrations() const {
        return state.sampler_registrations ? state.sampler_registrations : UINT64_MAX;
    }
    void order_cpu_read(std::uint64_t address, std::uint64_t size, bool overlay);
    // Preserve command order even when addresses repeat: a later outer update
    // supersedes older inner writes, while a shorter rewrite preserves the tail
    // of an earlier larger update. Address-only replacement loses both rules.
    std::vector<UniformUpdate> uniform_updates;
    // Moves on every recorded update to invalidate cached constant windows.
    std::uint64_t update_generation = 0;
    void record_uniform_update(std::uint64_t address, std::uint64_t size,
                               const std::byte* bytes) {
        ++pending.counters.uniform_updates;
        ++update_generation;
        uniform_updates.push_back({address, size, bytes});
        note_update_pages(address, size);
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
    std::vector<std::pair<std::uint64_t, std::uint64_t>> other_buffer_writes;
    bool fill_uniform(UniformVersion& version, std::uint64_t end);
    bool version_uniform(std::uint64_t address, std::uint64_t size, const std::byte* bytes);
    bool flush_uniforms();
    bool uniform_conflict(std::uint64_t address, std::uint64_t size) const;
    std::uint64_t uniform_address(const BoundBuffer& binding, std::uint64_t gpu,
                                  std::uint64_t extent);
    void overlay_uniform(std::uint64_t address, std::byte* bytes, std::size_t size) const;
    bool enqueue_counter(const ReportCounter& command, const BufferSpan& destination, std::byte* destination_cpu);
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
    void invalidate_uniforms(std::uint64_t address, std::uint64_t size);
    bool bind_draw_support(std::uint32_t base_instance);
    bool bind_vertex_fetch(const platform::CompiledGraphicsStage& vertex);
    UploadSlice upload(std::size_t size, const void* data = nullptr);
    // Hardware constant buffers as the shaders address them: 0 is driver
    // data, 1 the stage's own constants, 2 the bound-texture handle table and
    // 3 + n the NVN uniform buffer slot n.
    BoundBuffer hardware_buffer(std::size_t stage, std::uint32_t slot) const;
    std::array<std::byte, 0xa00> driver_constants(std::size_t stage, std::uint32_t slot) const;
    bool draw(const Draw& draw);
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
    if (queue) {
        // The worker keeps changing its descriptor after the pass begins.
        queue->put(EncodeOp::render_begin, 0,
                   RenderBeginRecord{(__bridge_retained void*)[pass copy], (__bridge void*)vertex,
                                     (__bridge void*)fragment});
        queue->publish();
        return (id<MTL4RenderCommandEncoder>)queue->token();
    }
    id<MTL4RenderCommandEncoder> encoder = [pending.commands renderCommandEncoderWithDescriptor:pass];
    if (!encoder) return nil;
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
                   visibilityOptions:MTL4VisibilityOptionDevice];
    if (vertex) [encoder setArgumentTable:vertex atStages:MTLRenderStageVertex];
    if (fragment) [encoder setArgumentTable:fragment atStages:MTLRenderStageFragment];
    return encoder;
}

id<MTL4ComputeCommandEncoder> Encoder::begin_transfer() {
    if (queue) {
        queue->put(EncodeOp::transfer_begin, 0);
        queue->publish();
        return (id<MTL4ComputeCommandEncoder>)queue->token();
    }
    id<MTL4ComputeCommandEncoder> encoder = [pending.commands computeCommandEncoder];
    if (!encoder) return nil;
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit | MTLStageDispatch
                   visibilityOptions:MTL4VisibilityOptionDevice];
    return encoder;
}

void Encoder::end_encoder(id encoder, MTLStages after) {
    if (queue) {
        queue->put(EncodeOp::end_encoder, 0, std::uint64_t(after));
        queue->publish();
        return;
    }
    [(id<MTL4CommandEncoder>)encoder barrierAfterStages:after beforeQueueStages:MTLStageAll
                                       visibilityOptions:MTL4VisibilityOptionDevice];
    [(id<MTL4CommandEncoder>)encoder endEncoding];
}

void Encoder::encoder_barrier(id encoder, MTLStages after, MTLStages before) {
    if (queue) queue->put(EncodeOp::encoder_barrier, 0, BarrierRecord{after, before});
    else [(id<MTL4CommandEncoder>)encoder barrierAfterEncoderStages:after beforeEncoderStages:before
                                                   visibilityOptions:MTL4VisibilityOptionDevice];
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
    if (queue) {
        queue->put(EncodeOp::call, 0, (__bridge_retained void*)[call copy]);
        queue->publish();
        return true;
    }
    EncodeCallContext context{pending.commands, pending.residency, pending.retained, {}};
    call(context);
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

// The key a row names, with its stages loaded from the package.
RowResolution resolve_pipeline_row(const PsoRow& row, platform::MetalShaderPackage& package, DrawPipelineKey& key,
                                   std::shared_ptr<const platform::CompiledGraphicsStage>& vertex,
                                   std::shared_ptr<const platform::CompiledGraphicsStage>& fragment) {
    constexpr auto state_offset = offsetof(DrawPipelineKey, colors);
    if ((row.kind != 'R' && row.kind != 'O') || row.fields.size() != 3 || row.fields[2].rfind("v1:", 0) != 0)
        return RowResolution::undecodable_state;
    vertex = package.stage_by_identity(row.fields[0]);
    fragment = row.fields[1] == "-" ? nullptr : package.stage_by_identity(row.fields[1]);
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
            const auto stage = package_->stage_by_identity(row.fields[0]);
            if (!stage) { ++unresolved_; return; }
            if (hooks_.library(stage)) ++built_; else ++failed_;
        } else if (row.kind == 'R') {
            DrawPipelineKey key;
            const auto resolution = resolve_pipeline_row(row, *package_, key, vertex, fragment);
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
        } else if (row.kind == 'O') {
            if (!hooks_.open) return;
            DrawPipelineKey key;
            const auto resolution = resolve_pipeline_row(row, *package_, key, vertex, fragment);
            if (resolution != RowResolution::resolved) {
                ++(resolution == RowResolution::missing_stage ? unresolved_ : undecodable_);
                return;
            }
            if (hooks_.open(key, std::move(vertex), std::move(fragment))) ++built_; else ++failed_;
        } else if (row.kind == 'C') {
            const auto stage = package_->stage_by_identity(row.fields[0]);
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

class ReplayImpl final : public MetalReplayBackend {
public:
    ReplayImpl(id<MTLDevice> device, id<MTL4CommandQueue> queue, std::shared_ptr<ReplayResources> resources,
               ReplaySchedulingPolicy scheduling)
        : device_(device), queue_(queue), resources_(std::move(resources)), scheduling_(scheduling) {
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
        resources_->set_completion_hook([this] {
            if (active_encoder && &active_encoder->backend == this)
                return active_encoder->synchronize_gpu();
            std::unique_lock lock(fence_mutex_);
            const auto fence = next_fence_ - 1;
            completion_.wait(lock, [&] { return completed_fence_ >= fence; });
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
        worker_ = std::thread([this] { run_worker(); });
    }

    ~ReplayImpl() override {
        if (archive_loader_.joinable()) archive_loader_.join();
        if (internal_preparer_.joinable()) internal_preparer_.join();
        warm_.reset();
        {
            // The exit handler must not reach a destroyed replay.
            const std::lock_guard lock(archive_exit_mutex());
            if (archive_exit_target() == this) archive_exit_target() = nullptr;
        }
        std::uint64_t fence = 0;
        {
            std::lock_guard lock(mutex_);
            fence = next_fence_ - 1;
        }
        if (fence) wait_fence(fence, UINT64_MAX);
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        queued_ready_.notify_all();
        if (worker_.joinable()) worker_.join();
        // Every submission retired, so every record was replayed.
        encode_thread_.reset();
        resources_->set_completion_hook({});
    }

    // The manifest lists the stages in the order the title first selects them,
    // which is the order its first draws ask the package for them. Open-form
    // rows count only for an archive build, the one session that builds them.
    static void prefetch_manifest_stages(platform::MetalShaderPackage& package, const std::vector<PsoRow>& rows,
                                         bool archive_build) {
        std::vector<std::string> identities;
        for (const auto& row : rows) {
            if (row.fields.empty() || row.kind == 'K' || (row.kind == 'O' && !archive_build)) continue;
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
        scheduling_period_ns_ = period_ns;
        preserve_encode_spin_ = preserve_encode_spin;
        queued_.push_back({0, {}, period_ns, preserve_encode_spin, true});
        queued_ready_.notify_one();
    }

    // Pin the lists and snapshot deferred bindings on the guest thread. The
    // replay worker reads ordinary records in place through fence retirement.
    std::uint64_t submit(std::span<const NVNcommandHandle> handles) override {
        const auto timeline_begin = timeline_now_ns();
        const auto cpu_start = thread_cpu_ns();
        CommandWalkStatus status;
        auto captured = prepare_commands(handles, status);
        if (!captured) {
            std::lock_guard lock(mutex_);
            last_error_ = {ReplayErrorCode::invalid_command, std::string(command_walk_status_name(status))};
            return 0;
        }
        std::lock_guard lock(mutex_);
        const auto fence = next_fence_++;
        fence_commands(*captured, fence);
        queued_.push_back({fence, std::move(captured), scheduling_period_ns_, preserve_encode_spin_});
        queued_ready_.notify_one();
        submit_cpu_ns_ += thread_cpu_ns() - cpu_start;
        timeline_event("submit", fence, timeline_now_ns() - timeline_begin);
        return fence;
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
            {
                std::unique_lock lock(mutex_);
                queued_ready_.wait(lock, [&] { return stopping_ || !queued_.empty(); });
                if (queued_.empty()) return;
                item = std::move(queued_.front());
                queued_.pop_front();
            }
            if (item.frame_boundary) {
                periodic_graphics_tick(item.scheduling_period_ns, ReplayThread::worker, scheduling_);
                continue;
            }
            // Encoder and descriptor temporaries must retire with each replay
            // submission, even though the worker thread itself is long-lived.
            const auto cpu_start = thread_cpu_ns();
            @autoreleasepool {
                encode(item.fence, std::move(item.captured), item.scheduling_period_ns,
                       item.preserve_encode_spin);
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
                     "segments=%llu gpu_interval_us=%llu timed_segments=%llu first_commit_us=%llu final_commit_us=%llu\n",
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
                         ? (pending.final_commit_ns - pending.encode_begin_ns) / 1000 : 0));
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
        while (finished_.contains(completed_fence_ + 1)) {
            const auto fence = completed_fence_ + 1;
            if (const auto it = inflight_.find(fence); it != inflight_.end()) {
                if (it->second->execution_error) execution_error_ = it->second->execution_error;
                else apply_effects_locked(*it->second);
                if (it->second->encode_warning) encode_warning_ = it->second->encode_warning;
                resources_->complete_gpu_writes(it->second->resource_write_generation);
                const auto lookups = resources_->texture_lookup_counts();
                retired_log_.push_back({it->second, fence, lookups.lookups - reported_lookups_.lookups,
                                        lookups.misses - reported_lookups_.misses});
                reported_lookups_ = lookups;
                inflight_.erase(it);
            }
            finished_.erase(fence);
            completed_fence_ = fence;
            published_completed_fence_.store(fence, std::memory_order_release);
        }
    }

    // Wakes the fence waiters, releases fence_mutex_, then writes the retired
    // submissions' diagnostics: the guest never waits for formatting. A woken
    // waiter may destroy the backend once the lock is free, so nothing after
    // the unlock touches it: the lines come from the submissions alone.
    void publish_retired(std::unique_lock<std::mutex>& lock) {
        auto retired = std::move(retired_log_);
        retired_log_.clear();
        completion_.notify_all();
        lock.unlock();
        for (const auto& entry : retired) log_retired(entry);
    }

    // A submission that failed to encode completes in order with no GPU work.
    void finish_failed(std::uint64_t fence) {
        publish_encoder();
        std::unique_lock lock(fence_mutex_);
        finished_[fence] = true;
        retire_finished();
        publish_retired(lock);
    }

    void encode(std::uint64_t fence, std::shared_ptr<const CommandSubmission> captured,
                std::uint64_t scheduling_period_ns, bool preserve_encode_spin) {
        const auto phase_start = timeline_now_ns();
        // The worker owns encoding. Only a full ring waits, and no encoder or
        // publication lock is held while GPU completion makes a slot available.
        const auto ring_wait_begin = timeline_now_ns();
        {
            std::unique_lock lock(fence_mutex_);
            completion_.wait(lock, [&] {
                return fence <= completed_fence_ + submission_slots;
            });
        }
        timeline_event("ring_wait", fence, timeline_now_ns() - ring_wait_begin);
        timeline_event("encode_begin", fence, 0);
        const auto encode_start = std::chrono::steady_clock::now();
        if (!compiler_) {
            encoder_error_ = {ReplayErrorCode::allocation, "Metal replay compiler is unavailable"};
            finish_failed(fence);
            return;
        }
        auto pending = std::make_shared<PendingSubmission>();
        pending->fence = fence;
        pending->scheduling_period_ns = scheduling_period_ns;
        pending->preserve_encode_spin = preserve_encode_spin;
        pending->encode_begin_ns = phase_start;
        pending->captured = std::move(captured);
        auto& slot = submission_storage_[(fence - 1) % submission_slots];
        if (!slot.allocator) slot.allocator = [device_ newCommandAllocator];
        else [slot.allocator reset];
        pending->allocator = slot.allocator;
        pending->encode_thread = encode_thread_.get();
        pending->residents.reserve(resident_hint_);
        auto* residency_descriptor = [MTLResidencySetDescriptor new];
        residency_descriptor.initialCapacity = 256;
        NSError* error = nil;
        pending->residency = [device_ newResidencySetWithDescriptor:residency_descriptor
                                                               error:&error];
        // Reuse query storage only after the slot's previous fence retired.
        if (!slot.visibility) {
            slot.visibility = [device_ newBufferWithLength:64 * 1024
                                                    options:MTLResourceStorageModeShared];
            ++statistics_.new_buffer_allocations;
        }
        pending->visibility = slot.visibility;
        // Without the encode thread the command buffer is created here; with
        // it, no record names the submission until every allocation exists.
        if (!pending->allocator || !pending->residency || !pending->visibility ||
            (!encode_thread_ && !begin_commands(*pending))) {
            encoder_error_ = {ReplayErrorCode::allocation,
                           "Metal replay submission allocation failed"};
            finish_failed(fence);
            return;
        }
        std::memset(pending->visibility.contents, 0, pending->visibility.length);
        [pending->retained addObjectsFromArray:@[pending->allocator,
                                                 pending->residency, pending->visibility]];
        if (encode_thread_) begin_commands(*pending);
        make_resident(*pending, pending->visibility);
        encoding_submission_ = pending;
        {
            std::lock_guard lock(fence_mutex_);
            inflight_[fence] = pending;
        }

        std::vector<std::shared_ptr<PendingSubmission>> released_submissions;
        resources_->begin_walk();
        Encoder encoder(*this, *pending, state_);
        encoder.retired_fence = completed_fence();
        // Releasing a retired submission frees its command buffers, allocators
        // and resources, which takes longer than the rest of the preamble. The
        // references are dropped when this function returns, after the commits.
        while (!previous_submissions_.empty() &&
               previous_submissions_.front()->fence <= encoder.retired_fence) {
            released_submissions.push_back(std::move(previous_submissions_.front()));
            previous_submissions_.pop_front();
        }
        encoder.previous_submissions.assign(previous_submissions_.begin(), previous_submissions_.end());
        for (const auto& previous : encoder.previous_submissions)
            for (const auto& update : previous->uniform_updates)
                encoder.note_update_pages(update.address, update.size);
        pending->phases.preamble = timeline_now_ns() - phase_start;
        const auto phase_walk = timeline_now_ns();
        EncodingScope scope(encoder);
        const auto walk_status = walk_commands(*pending->captured,
            [](const CommandRecordView& record, void* context) {
                return static_cast<Encoder*>(context)->visit(record);
            }, &encoder);
        if (walk_status != CommandWalkStatus::complete && !encoder.failed) {
            encoder.failed = true;
            encoder.error = {ReplayErrorCode::invalid_command, std::string(command_walk_status_name(walk_status))};
        }
        pending->phases.walk = timeline_now_ns() - phase_walk;
        resident_hint_ = pending->residents.count;
        const auto phase_finish = timeline_now_ns();
        if (!encoder.failed && !encoder.finish()) {
            // finish records its exact error.
        }
        pending->resource_write_generation = resources_->gpu_write_generation();
        if (encoder.failed) {
            // A committed prefix owns this guest fence until all feedback has
            // arrived. Never recycle its allocator through finish_failed().
            const auto failure = encoder.error;
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
        }
        previous_submissions_.push_back(pending);
        for (auto* sync : pending->fenced_syncs) {
            if (!sync) continue;
            sync->fence = pending->fence;
            sync->fenced = 1;
        }
        pending->phases.finish = timeline_now_ns() - phase_finish;
        ++statistics_.submissions;
        commit_segment(*pending, SegmentKind::final);
        statistics_.encode_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - encode_start).count());
        pending->phases.encode_end = timeline_now_ns();
        timeline_event("encode_end", fence, timeline_now_ns() - ring_wait_begin);
    }

    void flush() override {
        // Metal 4 commits begin asynchronously; submit has already handed all
        // accepted work to the queue. Texture uploads made outside a
        // submission (a texel write before a present) finish here, before a
        // reader on another queue can see the texture.
        if (!resources_->flush_uploads())
            armigo::diag::write(armigo::diag::Level::error,  "nvn_replay: texture upload failed before a present\n");
    }

    // Lock-free: guests poll syncs with a zero timeout in a loop, and taking
    // fence_mutex_ for each poll contended with fence retirement. The value is
    // stored after the fence's effects were applied (release), so a reader
    // that sees a fence also sees its effects.
    std::uint64_t completed_fence() const override {
        return published_completed_fence_.load(std::memory_order_acquire);
    }

    bool wait_fence(std::uint64_t fence, std::uint64_t timeout_ns) override {
        std::unique_lock lock(fence_mutex_);
        if (!fence || fence <= completed_fence_) return true;
        if (fence >= next_fence_) return false;
        if (timeout_ns == UINT64_MAX) {
            completion_.wait(lock, [&] { return completed_fence_ >= fence; });
            return true;
        }
        return completion_.wait_for(lock, std::chrono::nanoseconds(std::min<std::uint64_t>(timeout_ns, INT64_MAX)),
                                    [&] { return completed_fence_ >= fence; });
    }

    ReplayError last_error() const override {
        std::scoped_lock lock(mutex_, fence_mutex_);
        return execution_error_ ? execution_error_ : last_error_ ? last_error_ : encode_warning_;
    }

    ReplayStatistics statistics() const noexcept override {
        std::scoped_lock lock(mutex_, fence_mutex_);
        auto result = published_statistics_;
        result.submit_cpu_ns = submit_cpu_ns_;
        result.encode_thread_cpu_ns = encode_thread_ ? encode_thread_->cpu_ns() : 0;
        result.gpu_ns = gpu_ns_;
        result.completion_cpu_ns = completion_cpu_ns_;
        result.counter_timestamp_fallbacks = counter_fallbacks_;
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

    id<MTL4CounterHeap> counter_heap(std::uint64_t fence) {
        auto& heap = submission_storage_[(fence - 1) % submission_slots].timestamp;
        if (!heap) {
            auto* descriptor = [MTL4CounterHeapDescriptor new];
            descriptor.type = MTL4CounterHeapTypeTimestamp;
            descriptor.count = 1024;
            heap = [device_ newCounterHeapWithDescriptor:descriptor error:nil];
        }
        return heap;
    }

    // Worker-owned; readers use the published snapshot. The first skip for
    // each reason is reported, so dropped work is never silent.
    void note_skip(const std::string& reason, unsigned op, const std::string& detail) {
        ++statistics_.skipped;
        const auto count = ++skips_[reason];
        if (count == 1)
            armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: skipped unsupported work: %s\n", reason.c_str());
        if (count <= skip_detail_limit())
            armigo::diag::opt_in( "Armigo NVN: skip %llu of \"%s\" at op %u: %s\n",
                         static_cast<unsigned long long>(count), reason.c_str(), op, detail.c_str());
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
            const auto retired = completed_fence();
            for (auto& candidate : upload_arenas_) {
                if (candidate->fence <= retired && candidate->capacity >= reserved) {
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
            [pending.commands useResidencySet:pending.residency];
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
        // Texture uploads run on their own queue and are committed here; the
        // queue orders the segment after them on the GPU (commit_now), except
        // for the first fence with uploads in flight, which the resources
        // settle on the CPU once. The event is owned by the resources, which
        // outlive the replay's commits.
        UploadFence uploads;
        resources_->commit_uploads(uploads);
        pending->commands_open = false;
        pending->committed_effects = pending->effects.size();
        const auto segment = ++next_segment_;
        {
            std::lock_guard lock(fence_mutex_);
            pending->segments.submit();
        }
        outstanding_segments_.fetch_add(1, std::memory_order_relaxed);
        auto* options = [MTL4CommitOptions new];
        const UploadFence range{nil, uploads.value, uploads.floor};
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            // A failed upload batch fails the segment that waited for it: its
            // draws sampled textures without their content.
            if (resources_->uploads_failed(range))
                report_encode_error(*pending, "texture upload failed");
            complete_segment(pending, segment, kind, feedback);
        }];
        if (encode_thread_) {
            encode_thread_->put(EncodeOp::commit, static_cast<std::uint32_t>(kind),
                                CommitRecord{(__bridge_retained void*)options, pending.get(), segment,
                                         (__bridge void*)uploads.event, uploads.value});
            encode_thread_->publish();
        } else {
            commit_now(*pending, pending->commands, options, segment, kind, uploads.event, uploads.value);
        }
        pending->phases.commit += timeline_now_ns() - begin;
    }

    // Makes the segment's residency current, ends its command buffer and
    // commits it: on the worker, or on the encode thread.
    void commit_now(PendingSubmission& pending, id<MTL4CommandBuffer> commands,
                    MTL4CommitOptions* options, std::uint64_t segment, SegmentKind kind,
                    id<MTLEvent> upload_event, std::uint64_t upload_value) {
        if (!commands) {
            abandon_segment(pending, "Metal replay segment has no command buffer");
            return;
        }
        [pending.residency commit];
        [commands endCommandBuffer];
        id<MTL4CommandBuffer> buffers[] = {commands};
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
        // The texture uploads this segment may read complete on the GPU first.
        if (upload_event) [queue_ waitForEvent:upload_event value:upload_value];
        [queue_ commit:buffers count:1 options:options];
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
        apply_effects_locked(pending);
    }

    // Units of the timestamps a counter report is built from, fixed for the
    // device: the heap's tick frequency, whether sampleTimestamps returns
    // nanoseconds (current systems) or mach ticks, and the offset from that
    // clock to the steady clock the driver's current timestamp reads.
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

    void apply_effects_locked(PendingSubmission& pending) {
        const auto* visibility = static_cast<const std::uint64_t*>(pending.visibility.contents);
        for (const auto& effect : pending.effects) {
            if (effect.kind == CompletionEffect::Kind::reset) {
                samples_counter_ = 0;
                continue;
            }
            if (effect.kind == CompletionEffect::Kind::signal) {
                if (effect.add) __atomic_fetch_add(effect.event, effect.value, __ATOMIC_SEQ_CST);
                else __atomic_store_n(effect.event, effect.value, __ATOMIC_SEQ_CST);
                continue;
            }
            for (auto index = effect.visibility_begin; index < effect.visibility_end; ++index)
                samples_counter_ += visibility[index];
            if (effect.kind != CompletionEffect::Kind::report) continue;
            MTLTimestamp cpu_after = 0, gpu_after = 0;
            [device_ sampleTimestamps:&cpu_after gpuTimestamp:&gpu_after];
            NSData* resolved = [effect.timestamp resolveCounterRange:NSMakeRange(effect.timestamp_index, 1)];
            if (!resolved || resolved.length < sizeof(MTL4TimestampHeapEntry)) {
                execution_error_ = {ReplayErrorCode::execution, "GPU timestamp heap resolution failed"};
                continue;
            }
            MTL4TimestampHeapEntry entry{};
            std::memcpy(&entry, resolved.bytes, sizeof entry);
            // The heap resolves at the device's timestamp frequency, while
            // sampleTimestamps returns nanoseconds on current systems and mach
            // ticks on older ones. Interpolate between the two sampled pairs in
            // nanoseconds so the guest sees the GPU's own spacing between
            // reports, then place the result on the clock the driver's current
            // timestamp reads through the device's fixed offset.
            const auto& clock = counter_clock();
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
            const auto steady = static_cast<std::int64_t>(ns) + clock.steady_offset_ns;
            ns = steady > 0 ? static_cast<std::uint64_t>(steady) : 0;
            NVNcounterData value{};
            value.timestamp = (ns / 625) * 384 + (ns % 625) * 384 / 625;
            value.counter = effect.samples ? samples_counter_ : 0;
            std::memcpy(effect.destination_cpu, &value, sizeof value);
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
                          std::uint64_t segment, SegmentKind kind,
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
    id<MTL4CommandQueue> queue_ = nil;
    id<MTL4Compiler> compiler_ = nil;
    std::shared_ptr<ReplayResources> resources_;
    const ReplaySchedulingPolicy scheduling_;
    ReplayResources::LookupCounts reported_lookups_{};
    // Neither mutex is held while encoding or waiting for a GPU segment.
    mutable std::mutex mutex_;
    mutable std::mutex fence_mutex_;
    std::condition_variable completion_;
    std::vector<RetiredLog> retired_log_;
    ReplayState state_{};
    ReplayError last_error_{};
    ReplayStatistics statistics_{}, published_statistics_{};
    ReplayError encoder_error_{}, execution_error_{};
    // The latest retired submission's encode warning (fence_mutex_).
    ReplayError encode_warning_{};
    std::uint64_t submit_cpu_ns_ = 0, gpu_ns_ = 0, completion_cpu_ns_ = 0;
    std::unordered_map<std::string, std::uint64_t> published_skips_;
    std::unordered_map<std::string, std::uint64_t> skips_;
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
        id<MTL4CounterHeap> timestamp = nil;
    };
    std::array<SubmissionSlot, submission_slots> submission_storage_;
    // Allocations the last submission made resident: the next one's set starts that large.
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
    std::map<std::uint64_t, bool> finished_;
    struct QueuedSubmission {
        std::uint64_t fence = 0;
        std::shared_ptr<const CommandSubmission> captured;
        std::uint64_t scheduling_period_ns = 0;
        bool preserve_encode_spin = false;
        bool frame_boundary = false;
    };
    // The queue mutex keeps both hint fields coherent when a submission captures them.
    std::uint64_t scheduling_period_ns_ = 0;
    bool preserve_encode_spin_ = false;
    std::deque<QueuedSubmission> queued_;
    std::condition_variable queued_ready_;
    bool stopping_ = false;
    std::thread worker_;
    SegmentPolicy segment_policy_;
    std::atomic<std::uint64_t> outstanding_segments_{0};
    std::uint64_t next_segment_ = 0;
    std::unique_ptr<EncodeThread> encode_thread_;
    std::shared_ptr<PendingSubmission> encoding_submission_;
    DrawTable<DrawPipelineKey, NativePipeline> pipelines_;
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
    const auto render = (id<MTL4RenderCommandEncoder>)encoder_;
    const auto compute = (id<MTL4ComputeCommandEncoder>)encoder_;
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
        [commands_ useResidencySet:owner_->residency];
        break;
    case EncodeOp::end_commands:
        [commands_ endCommandBuffer];
        commands_ = nil;
        owner_ = nullptr;
        owned_.reset();
        break;
    case EncodeOp::commit: {
        const auto record = record_value<CommitRecord>(payload);
        const auto scheduling_period_ns = record.owner->scheduling_period_ns;
        const auto kind = static_cast<SegmentKind>(argument);
        MTL4CommitOptions* options = (__bridge_transfer MTL4CommitOptions*)record.options;
        backend_.commit_now(*record.owner, commands_, options, record.segment, kind,
                            (__bridge id<MTLEvent>)record.upload_event, record.upload_value);
        commands_ = nil;
        owner_ = nullptr;
        owned_.reset();
        if (kind == SegmentKind::final) {
            idle_ = true;
            options = nil;
            if (pool_) objc_autoreleasePoolPop(pool_);
            pool_ = nullptr;
            const auto now = thread_cpu_ns();
            cpu_ns_.fetch_add(now - cpu_mark_, std::memory_order_relaxed);
            cpu_mark_ = now;
            backend_.update_encode_budget(scheduling_period_ns);
        }
        break;
    }
    case EncodeOp::resident: {
        const auto record = record_value<ResidentRecord>(payload);
        [record.owner->residency addAllocation:(__bridge id<MTLAllocation>)record.allocation];
        break;
    }
    case EncodeOp::call: {
        EncodeCall call = (__bridge_transfer EncodeCall)record_value<void*>(payload);
        EncodeCallContext context{commands_, owner_ ? owner_->residency : nil,
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
        [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
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
                                           beforeQueueStages:MTLStageAll
                                           visibilityOptions:MTL4VisibilityOptionDevice];
        [(id<MTL4CommandEncoder>)encoder_ endEncoding];
        encoder_ = nil;
        break;
    case EncodeOp::encoder_barrier: {
        const auto record = record_value<BarrierRecord>(payload);
        [(id<MTL4CommandEncoder>)encoder_ barrierAfterEncoderStages:record.after
                                                beforeEncoderStages:record.before
                                                  visibilityOptions:MTL4VisibilityOptionDevice];
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
    vertex.name = [NSString stringWithUTF8String:key.vertex->entry_point.c_str()];
    auto* fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = result.fragment_library;
    if (key.fragment) fragment.name = [NSString stringWithUTF8String:key.fragment->entry_point.c_str()];
    auto* descriptor = [MTL4RenderPipelineDescriptor new];
    descriptor.vertexFunctionDescriptor = vertex;
    if (key.fragment) {
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
        // The replay binds disabled custom-border records (see bind_stage).
        if (key.fragment->custom_border) {
            bool records = false;
            [constants setConstantValue:&records type:MTLDataTypeBool atIndex:12];
        }
        auto* specialized = [MTL4SpecializedFunctionDescriptor new];
        specialized.functionDescriptor = fragment;
        specialized.constantValues = constants;
        descriptor.fragmentFunctionDescriptor = specialized;
    }
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
        if (key.vertex->constant_buffer_support_index) reserve(*key.vertex->constant_buffer_support_index);
        if (key.vertex->draw_support_buffer_index) reserve(*key.vertex->draw_support_buffer_index);
        if (key.vertex->argument_buffer) reserve(key.vertex->argument_buffer->buffer_index);
        if (key.vertex->custom_border) reserve(key.vertex->custom_border->buffer_index);
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

    for (std::size_t index = 0; index != key.colors.size(); ++index) {
        auto* attachment = descriptor.colorAttachments[index];
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
            return nullptr;
        }
        attachment.blendingState = MTL4BlendStateEnabled;
        attachment.sourceRGBBlendFactor = *src_color;
        attachment.destinationRGBBlendFactor = *dst_color;
        attachment.rgbBlendOperation = *color_op;
        attachment.sourceAlphaBlendFactor = *src_alpha;
        attachment.destinationAlphaBlendFactor = *dst_alpha;
        attachment.alphaBlendOperation = *alpha_op;
    }
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
        if (!capture_open_form(key, open, failure))
            armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: open form not archived (%s %s): %s\n", key.vertex->identity.c_str(),
                         key.fragment->identity.c_str(), failure.c_str());
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

bool Encoder::close_render(PassBreak why, bool keep_clears) {
    if (render) {
        trace_pass_event("end", pass_break_names[std::size_t(why)]);
        finalized_visibility = visibility_cursor;
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
        end_encoder(render, MTLStageAll);
        render = nil;
        pass_depth_only = true;
        depth_barrier_allocation = nil;
        depth_barrier_reads = false;
        depth_omitted = false;
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

bool Encoder::close_transfer() {
    if (!transfer) return true;
    end_encoder(transfer, MTLStageBlit | MTLStageDispatch);
    transfer = nil;
    return true;
}

bool Encoder::barrier(PassBreak why) {
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
    cut_draws = pending.counters.draws;
    return true;
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
    ++backend.mutable_statistics().synchronize_gpu_waits;
    ++gpu_synchronizations;
    if (!pending.commands_open)
        return fail(ReplayErrorCode::encoding, "CPU drain has no open command buffer");
    if (!flush_uniforms() || !close_render(PassBreak::cpu_drain) || !close_transfer()) return false;
    note_writes();
    backend.commit_segment(pending, SegmentKind::drain);
    const auto execution_error = backend.wait_segments(pending);
    if (execution_error) return fail(execution_error.code, execution_error.detail);
    // CPU effects from earlier guest submissions precede this prefix's effects.
    backend.wait_fence(pending.fence - 1, UINT64_MAX);
    retired_fence = backend.completed_fence();
    backend.apply_effects(pending);
    pending.committed_effects = 0;
    // Applying cleared pending.effects: the reports are in memory now, and
    // later effects reuse the indices that gpu_counters names.
    gpu_counters.clear();
    effect_span = {};
    if (state.samples_active && counter_begin != visibility_cursor) {
        backend.accumulate_samples(pending.visibility, counter_begin, visibility_cursor);
        counter_begin = visibility_cursor;
    }
    buffer_write_order.clear();
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
        backend.wait_fence(pending.fence - 1, UINT64_MAX);
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
    effect_span = {};
    failed = true;
    error = original;
}

bool Encoder::sync_draw_resources() {
    const auto registration = backend.resources().registration_generation();
    const auto writes = memory::WriteTracker::now();
    if (registration != dirty.registration) ++pending.counters.syncs_registration;
    else if (writes != dirty.cpu_writes) ++pending.counters.syncs_writes;
    return dirty.sync_resources(registration, writes);
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
    for (auto word = address & ~std::uint64_t{3}; word < end;) {
        if (uniform_words.contains(word)) return false;
        if (word > UINT64_MAX - 4) break;
        word += 4;
    }
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

bool Encoder::order_texture_backing(std::uint64_t begin, std::uint64_t size, bool& ordered) {
    if (!begin || !size) return true;
    if (size > UINT64_MAX - begin)
        return fail(ReplayErrorCode::missing_resource, "texture backing range overflows");
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
    // Every binding reads through here: a read outside the spans the
    // completion effects write returns without scanning them.
    if (pending.effects.size() < effect_span.counted) effect_span = {};
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
    }
    const bool outside_pending = cpu && (begin + size <= effect_span.low || effect_span.high <= begin);
    for (const auto& previous : previous_submissions) {
        if (previous->fence <= retired_fence) continue;
        if (cpu && (begin + size <= previous->completion_low || previous->completion_high <= begin)) continue;
        for (const auto& [write_begin, write_end] : previous->completion_writes)
            if (begin < write_end && write_begin < begin + size) {
                if (ordered) *ordered = true;
                const auto start = wait_tag ? timeline_now_ns() : 0;
                backend.wait_fence(previous->fence, UINT64_MAX);
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
    if (!flush_uniforms()) return false;
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
    // A samples-passed report inside a pass created with the visibility
    // buffer keeps the pass: its slots are those of the draws before it, and
    // the render encoder writes its timestamp. Other reports end the pass,
    // so a GPU timestamp marks completed work precisely.
    const bool in_pass = render && pass_visibility_buffer && effect.samples;
    if (!in_pass) {
        close_render(PassBreak::counter);
        close_transfer();
    }
    if (pending.timestamp_count >= 1024)
        return fail(ReplayErrorCode::allocation, "submission timestamp storage is exhausted");
    effect.timestamp = backend.counter_heap(pending.fence);
    effect.timestamp_index = pending.timestamp_count++;
    if (!effect.timestamp)
        return fail(ReplayErrorCode::allocation, "GPU counter heap creation failed");
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
    return true;
}

bool Encoder::resolve_carried_predicate() {
    if (!state.predicate_pending) return true;
    state.predicate_pending = false;
    backend.wait_fence(pending.fence - 1, UINT64_MAX);
    retired_fence = backend.completed_fence();
    return read_predicate(state.predicate_mode, state.predicate_address);
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
    job.capacity = predicate_capacity;
    for (std::size_t index = 0; index != 2; ++index) {
        const auto address = command.address + index * sizeof(NVNcounterData);
        // The raw lookup: Encoder::buffer_at would wait for this submission's
        // own pending writers of the counter.
        const auto span = backend.resources().buffer_at(address, sizeof(NVNcounterData));
        if (!span.buffer || !span.buffer.contents) return true;
        const auto* cpu = static_cast<const std::byte*>(span.buffer.contents) + span.offset;
        // The latest pending effect that writes the counter decides its value.
        std::optional<std::size_t> writer;
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

void Encoder::note_texture_write(const TextureState& texture) {
    pending.texture_writes.push_back(texture);
    if (texture.memory && texture.size && backend.resources().texture_is_in_place(texture))
        other_buffer_writes.emplace_back(reinterpret_cast<std::uint64_t>(texture.memory), texture.size);
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
// operations that are never tracked (copies, clears): they take the barrier
// and leave the encoder untracked. Dispatches write no texture, because
// bind_stage rejects storage image writes.
void Encoder::note_compute_texture(__unsafe_unretained id<MTLTexture> texture) {
    __unsafe_unretained id<MTLTexture> root = texture;
    while (__unsafe_unretained id<MTLTexture> parent = root.parentTexture) root = parent;
    if (root.buffer) compute_untracked = true;
}


bool Encoder::open_render(std::int32_t primitive) {
    close_transfer();
    const auto& fragment_program = state.stage_programs[NVN_SHADER_STAGE_FRAGMENT];
    const auto* fragment_shader = fragment_program
        ? fragment_program->stages[NVN_SHADER_STAGE_FRAGMENT].get() : nullptr;
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
    if ((!render || color_aliases.any) && fragment_shader && !state.rasterizer_discard) {
        const auto channels = state.channel_mask.mask & fragment_shader->interface.color_output_mask;
        for (unsigned i = 0; i < 8; ++i)
            if ((channels >> (i * 4)) & 15) color_writers |= 1u << i;
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
        if (!attachment_width || !attachment_height)
            return fail(ReplayErrorCode::unsupported,
                        "draw requires at least one render attachment");
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
    if (!vertex_program)
        return fail(ReplayErrorCode::missing_program, "draw has no bound vertex program");
    const auto* vertex_shader = vertex_program->stages[NVN_SHADER_STAGE_VERTEX].get();
    if (!last_pipeline || dirty.pipeline || last_pipeline_primitive != primitive) {
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
    if (sync_draw_resources()) {
        for (auto& stage : bindings) stage.windows.fill({});
    }
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
    if (dirty.vertices) {
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
                    armigo::diag::opt_in(
                                 "nvn_pass frame=%llu %lux%lu samples=%u targets=%d depth=%d in_place=%d "
                                 "pass=%llu fragment=%u channels=%u depth_test=%u depth_write=%u "
                                 "stencil_test=%u rasterizer_discard=%u samples_active=%u\n",
                                 static_cast<unsigned long long>(pending.fence),
                                 static_cast<unsigned long>(attachment_width),
                                 static_cast<unsigned long>(attachment_height), unsigned(attachment_samples),
                                 targets, state.targets.depth != nullptr, in_place,
                                 static_cast<unsigned long long>(pending.counters.passes),
                                 unsigned(fragment_shader != nullptr),
                                 fragment_shader ? state.channel_mask.mask & fragment_shader->interface.color_output_mask : 0,
                                 unsigned(state.depth_stencil.depth_test), unsigned(state.depth_stencil.depth_write),
                                 unsigned(state.depth_stencil.stencil_test), unsigned(state.rasterizer_discard),
                                 unsigned(state.samples_active));
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
    return true;
}

BoundBuffer Encoder::hardware_buffer(std::size_t stage, std::uint32_t slot) const {
    if (slot >= 3) return slot - 3 < buffer_slots ? state.uniform[stage][slot - 3] : BoundBuffer{};
    if (slot == 1 && state.stage_programs[stage]) {
        const auto& constants = state.stage_programs[stage]->constants[stage];
        return {constants.address, constants.size};
    }
    return {};
}

// Driver data (slot 0): words 0..7 hold 0..7 and every other word all ones,
// with each stage's storage-buffer table (address, size per 16-byte row) at
// its base and the compute stage's uniform-buffer table at 0x210. The
// bound-texture handle table (slot 2) holds the 32-bit handle of texture
// slot n at 0x20 + 8 n.
std::array<std::byte, 0xa00> Encoder::driver_constants(std::size_t stage, std::uint32_t slot) const {
    std::array<std::byte, 0xa00> bytes{};
    bytes.fill(std::byte{0});
    const auto put = [&](std::size_t offset, const void* value, std::size_t size) {
        if (offset + size <= bytes.size()) std::memcpy(bytes.data() + offset, value, size);
    };
    if (slot == 0) {
        std::memset(bytes.data(), 0xff, bytes.size());
        for (std::uint32_t word = 0; word < 8; ++word) put(4 * word, &word, 4);
        const int base = stage == NVN_SHADER_STAGE_VERTEX ? 0x110
            : stage == NVN_SHADER_STAGE_COMPUTE ? 0x310
            : stage == NVN_SHADER_STAGE_FRAGMENT ? 0x510 : -1;
        if (base >= 0)
            for (std::size_t row = 0; row < buffer_slots; ++row) {
                const auto& buffer = state.storage[stage][row];
                const auto size = static_cast<std::uint32_t>(std::min<std::uint64_t>(buffer.size, UINT32_MAX));
                put(base + 16 * row, &buffer.address, 8);
                put(base + 16 * row + 8, &size, 4);
            }
        if (stage == NVN_SHADER_STAGE_COMPUTE)
            for (std::size_t row = 0; row < buffer_slots; ++row) {
                const auto& buffer = state.uniform[stage][row];
                const auto size = static_cast<std::uint32_t>(std::min<std::uint64_t>(buffer.size, UINT32_MAX));
                put(0x210 + 16 * row, &buffer.address, 8);
                put(0x210 + 16 * row + 8, &size, 4);
            }
    } else if (slot == 2) {
        for (std::size_t texture = 0; texture < 32; ++texture) {
            const auto handle = static_cast<std::uint32_t>(state.textures[stage][texture]);
            put(0x20 + 8 * texture, &handle, 4);
        }
    }
    return bytes;
}

void Encoder::order_cpu_read(std::uint64_t address, std::uint64_t size, bool overlay) {
    for (const auto& previous : previous_submissions) {
        if (previous->fence <= retired_fence) continue;
        // Uniform updates have captured CPU bytes, so an overlay can consume
        // them without waiting. Other GPU writes require actual completion.
        const auto& writes = overlay ? previous->other_buffer_writes : previous->buffer_writes;
        for (const auto& [begin, length] : writes)
            if (address < begin + length && begin < address + size) {
                backend.wait_fence(previous->fence, UINT64_MAX);
                retired_fence = backend.completed_fence();
                break;
            }
    }
}

void Encoder::overlay_uniform(std::uint64_t address, std::byte* bytes, std::size_t size) const {
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
        }
    };
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
    order_cpu_read(address, end - address, true);
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

bool Encoder::uniform_conflict(std::uint64_t address, std::uint64_t size) const {
    return std::any_of(uniform_versions.begin(), uniform_versions.end(), [&](const auto& version) {
        return address && size && address < version.end && version.begin < address + size;
    });
}

bool Encoder::version_uniform(std::uint64_t address, std::uint64_t size, const std::byte* bytes) {
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
    if (!end || end - begin > 64 * 1024 || uniform_updates.size() >= 4096) return false;
    const auto overlaps = [&](std::uint64_t address, std::uint64_t size) {
        return address && size && address < end && begin < address + size;
    };
    for (std::size_t stage = 0; stage < stage_count; ++stage) {
        for (const auto& binding : state.uniform[stage])
            if (overlaps(binding.address, binding.size) &&
                (binding.address < begin || binding.address + binding.size > end)) return false;
        for (const auto& binding : state.storage[stage])
            if (overlaps(binding.address, binding.size)) return false;
    }
    for (const auto& binding : state.vertex_buffers)
        if (overlaps(binding.address, binding.size)) return false;
    for (const auto& [address, size] : other_buffer_writes)
        if (overlaps(address, size)) return false;
    UniformVersion* version = nullptr;
    for (auto& candidate : uniform_versions)
        if (overlaps(candidate.begin, candidate.end - candidate.begin)) {
            if (candidate.begin != begin || candidate.end != end) return false;
            version = &candidate;
        }
    if (!version) {
        if (uniform_versions.size() >= 64) return false;
        ++pending.counters.version_creations;
        uniform_versions.push_back({begin, end, begin, address, address + size, {}});
        version = &uniform_versions.back();
    }
    version->dirty_begin = std::min(version->dirty_begin, address);
    version->dirty_end = std::max(version->dirty_end, address + size);
    record_uniform_update(address, size, bytes);
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
    auto versions = std::move(uniform_versions);
    uniform_versions.clear();
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
        copy_buffer_range(version.slice.buffer, version.slice.offset + version.dirty_begin - version.begin,
                          destination.buffer, destination.offset, size);
        pending.buffer_writes.emplace_back(version.dirty_begin, size);
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
void Encoder::invalidate_uniforms(std::uint64_t address, std::uint64_t size) {
    if (!uniform_rebind_enabled()) return invalidate_args_all();
    for (std::size_t stage = 0; stage < stage_count; ++stage) {
        if (stage != NVN_SHADER_STAGE_VERTEX && stage != NVN_SHADER_STAGE_FRAGMENT) {
            dirty.arguments[stage] = true;
            continue;
        }
        bool contents = false;
        for (const auto& read : bindings[stage].content_reads) {
            const auto bound = hardware_buffer(stage, read.slot);
            const auto begin = bound.address + read.offset;
            contents = bound.address && address < begin + read.length && begin < address + size;
            if (contents) break;
        }
        if (contents) dirty.arguments[stage] = true;
        else dirty.uniforms[stage] = true;
    }
}

bool Encoder::bind_stage(std::size_t stage_index,
                         const platform::CompiledGraphicsStage& shader,
                         __unsafe_unretained id<MTL4ArgumentTable> table, bool vertex,
                         bool uniforms_only) {
    ++backend.mutable_statistics().stage_binds;
    ++pending.counters.stage_binds;
    if (uniforms_only) {
        ++backend.mutable_statistics().stage_uniform_binds;
        ++pending.counters.stage_uniform_binds;
    }
    if (stage_index >= stage_count)
        return fail(ReplayErrorCode::invalid_command, "shader stage index exceeds the replay limit");
    if (stage_index == NVN_SHADER_STAGE_COMPUTE &&
        sync_draw_resources())
        for (auto& bound : bindings) bound.windows.fill({});
    auto& stage = bindings[stage_index];
    if (stage.shader != &shader) {
        stage.window_extents.fill(0);
        for (const auto& window : shader.constant_buffer_windows)
            if (window.guest_binding < stage.window_extents.size())
                stage.window_extents[window.guest_binding] = window.window_bytes;
        stage.writes_buffers = std::ranges::any_of(shader.storage_buffers, [](const auto& binding) {
            return binding.access != platform::CompiledShaderResourceAccess::Read;
        });
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
    if (shader.custom_border) {
        // Disabled border records: the shader samples with the Metal sampler's
        // own border colour (transparent black, opaque black or opaque white,
        // the nearest to the NVN colour); exact custom colours are not applied.
        id<MTLBuffer> records = backend.zero_buffer();
        if (shader.custom_border->buffer_index >= 31 || !records ||
            shader.custom_border->pairs.size() * 64 > records.length)
            return fail(ReplayErrorCode::unsupported, "custom border records exceed the replay limit");
        write_buffer_slot(table, stage_index, shader.custom_border->buffer_index,
                          records.gpuAddress);
        make_resident(pending, records);
    }
    const auto bind_buffer = [&](const BoundBuffer& binding, std::uint32_t index,
                                 std::uint64_t minimum, bool uniform = false,
                                 bool writable = false) -> bool {
        if (index >= 31)
            return fail(ReplayErrorCode::unsupported,
                        "shader buffer index exceeds the Metal argument-table limit");
        if (!binding.address) {
            write_buffer_slot(table, stage_index, index, 0);
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
    const auto bind_driver = [&](std::uint32_t slot, std::uint32_t index) -> bool {
        if (index >= 31)
            return fail(ReplayErrorCode::unsupported,
                        "shader buffer index exceeds the Metal argument-table limit");
        auto& cached = stage.driver[slot];
        const auto generation = stage.driver_generation[slot];
        if (cached.generation == generation && cached.gpu) {
            write_buffer_slot(table, stage_index, index, cached.gpu);
            return true;
        }
        const auto bytes = driver_constants(stage_index, slot);
        const auto buffer = upload(bytes.size(), bytes.data());
        if (!buffer) return false;
        cached = {generation, buffer.gpu};
        write_buffer_slot(table, stage_index, index, buffer.gpu);
        return true;
    };
    // A window binding's proven read extent: a uniform version has to carry
    // only the bytes the shader can read, not the range the game bound.
    const auto& window_extents = stage.window_extents;
    for (const auto& binding : shader.constant_buffers) {
        if (binding.guest_binding >= buffer_slots + 3)
            return fail(ReplayErrorCode::unsupported, "constant buffer slot exceeds the replay limit");
        if (binding.guest_binding == 0 || binding.guest_binding == 2) {
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
        const auto source_binding = hardware_buffer(stage_index, binding.guest_binding);
        // A guest buffer that covers the whole window binds in place, and the
        // GPU reads the bytes at execution as the hardware does. Shaders read
        // every word of the window without checking the bound size, so a
        // smaller buffer gets a window copy that is zero past its bound size.
        // Earlier in-stream updates reach memory by GPU copies ordered before
        // this draw; an active version carries the later ones. Driver data
        // and ranges the GPU wrote in this submission still use a window.
        if (binding.guest_binding != 0 && binding.guest_binding != 2 &&
            source_binding.address && source_binding.size >= binding.window_bytes) {
            const std::uint64_t extent = binding.window_bytes;
            const bool gpu_written = std::any_of(other_buffer_writes.begin(), other_buffer_writes.end(),
                [&](const auto& range) {
                    return source_binding.address < range.first + range.second &&
                        range.first < source_binding.address + extent;
                });
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
            return std::any_of(other_buffer_writes.begin() + std::ptrdiff_t(snapshot.writes),
                               other_buffer_writes.end(), [&](const auto& range) {
                return snapshot.source < range.first + range.second && range.first < end;
            });
        };
        if (cached.gpu && cached.bytes == binding.window_bytes &&
            cached.source == source_binding.address && cached.source_size == source_binding.size &&
            cached.generation == update_generation &&
            cached.cpu && cached.writes <= other_buffer_writes.size() && !written_since(cached)) {
            ++pending.counters.window_reuses;
            write_buffer_slot(table, stage_index, binding.metal_buffer_index, cached.gpu);
            continue;
        }
        const auto window = upload(binding.window_bytes);
        if (!window) return false;
        ++pending.counters.window_uploads;
        write_buffer_slot(table, stage_index, binding.metal_buffer_index, window.gpu);
        // The upload is zero-filled: words past the copied bytes read zero, as
        // past the hardware's bound size.
        std::uint64_t filled = 0;
        if (binding.guest_binding == 0 || binding.guest_binding == 2) {
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
            const bool gpu_written = std::any_of(other_buffer_writes.begin(), other_buffer_writes.end(),
                [&](const auto& range) {
                    return source_binding.address < range.first + range.second &&
                        range.first < source_binding.address + copied;
                });
            if (gpu_written) {
                if (!flush_uniforms() || !open_transfer()) return false;
                make_resident(pending, source.buffer);
                if (filled)
                    copy_buffer_range(source.buffer, source.offset, window.buffer, window.offset, filled);
                close_transfer();
                // Only the GPU copy reads the source after encoding.
                [pending.retained addObject:source.buffer];
            } else {
                order_cpu_read(source_binding.address, copied, true);
                std::memcpy(window.cpu,
                            static_cast<const std::byte*>(source.buffer.contents) + source.offset, filled);
                overlay_uniform(source_binding.address, window.cpu, filled);
            }
        }
        cached = {window.gpu, source_binding.address, source_binding.size, update_generation,
                  binding.window_bytes, window.cpu, other_buffer_writes.size()};
        write_buffer_slot(table, stage_index, binding.metal_buffer_index, window.gpu);
    }

    // A uniform-only rebind ends here: the rest resolves from bindings an
    // in-stream update does not touch (invalidate_uniforms checks that).
    if (uniforms_only) return true;
    bindings[stage_index].writes.clear();
    const bool compute = stage_index == NVN_SHADER_STAGE_COMPUTE;
    if (compute) {
        // Guest constant buffers the GPU reads: every bound range the shader
        // names (driver data is CPU-written upload memory).
        compute_accesses.clear();
        compute_untracked = !shader.constant_buffers_reflected;
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
        // Only the driver-data tables give the dispatch's exact range; a
        // descriptor read from guest memory or the binding-index fallback
        // leaves the dispatch's accesses incomplete.
        bool from_table = false;
        if (reflected.cbuf_slot == 0) {
            // Descriptors in the driver data: the storage and compute uniform tables.
            const auto offset = reflected.descriptor_byte_offset;
            const int base = stage_index == NVN_SHADER_STAGE_VERTEX ? 0x110
                : stage_index == NVN_SHADER_STAGE_COMPUTE ? 0x310
                : stage_index == NVN_SHADER_STAGE_FRAGMENT ? 0x510 : -1;
            if (base >= 0 && offset >= std::uint32_t(base) && offset < base + 16 * buffer_slots && !(offset % 16)) {
                resolved = state.storage[stage_index][(offset - base) / 16];
                from_table = true;
            } else if (stage_index == NVN_SHADER_STAGE_COMPUTE && offset >= 0x210 &&
                       offset < 0x210 + 16 * buffer_slots && !(offset % 16)) {
                resolved = state.uniform[stage_index][(offset - 0x210) / 16];
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
                    order_cpu_read(address, descriptor_size, true);
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
                }
            }
        }
        if (!resolved.address && index < buffer_slots)
            resolved = state.storage[stage_index][index];
        if (!bind_buffer(resolved, reflected.metal_buffer_index, reflected.alignment, false,
                         reflected.access != platform::CompiledShaderResourceAccess::Read)) return false;
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
        if (slot >= (image ? image_slots : texture_slots)) return 0;
        const auto handle = image ? state.images[stage_index][slot]
                                  : state.textures[stage_index][slot];
        return static_cast<std::int32_t>(handle_texture_id(handle));
    };
    const auto bound_texture = [&](std::size_t slot, bool image) -> id<MTLTexture> {
        const auto id = bound_texture_id(slot, image);
        return id ? sampled_texture(id) : nil;
    };
    // The sampler bound at `slot`, else the one in the texture handle bound at
    // `texture_slot`.
    const auto bound_sampler = [&](std::size_t slot, std::size_t texture_slot)
        -> id<MTLSamplerState> {
        const std::uint64_t texture_handle =
            texture_slot < texture_slots ? state.textures[stage_index][texture_slot] : 0;
        std::int32_t id = 0;
        if (slot < sampler_slots && state.samplers[stage_index][slot])
            id = static_cast<std::int32_t>(handle_sampler_id(
                state.samplers[stage_index][slot]));
        if (!id) id = static_cast<std::int32_t>(handle_sampler_id(texture_handle));
        return id ? backend.resources().sampler_for_id(id, registrations) : nil;
    };
    const auto read_handle = [&](const platform::CompiledShaderResource& resource,
                                 std::uint32_t element)
        -> std::uint64_t {
        const auto offset = std::uint64_t(resource.handle_byte_offset) + element * 4ull;
        if (resource.cbuf_slot == 2) {
            // Bound-texture handle table: slot n at 0x20 + 8 n.
            if (offset < 0x20 || (offset - 0x20) % 8 || (offset - 0x20) / 8 >= texture_slots) return 0;
            return static_cast<std::uint32_t>(state.textures[stage_index][(offset - 0x20) / 8]);
        }
        if (resource.cbuf_slot == 0 || resource.cbuf_slot >= buffer_slots + 3) return 0;
        const auto binding = hardware_buffer(stage_index, resource.cbuf_slot);
        if (!binding.address || offset > binding.size || sizeof(std::uint32_t) > binding.size - offset)
            return 0;
        const auto address = binding.address + offset;
        if ((address & 3) == 0)
            if (const auto found = uniform_words.find(address); found != uniform_words.end())
                return found->second;
        const auto span = buffer_at(address, sizeof(std::uint32_t));
        std::uint32_t handle = 0;
        order_cpu_read(address, sizeof handle, true);
        if (span.buffer.contents && span.offset <= span.buffer.length &&
            sizeof handle <= span.buffer.length - span.offset)
            std::memcpy(&handle, static_cast<const std::byte*>(span.buffer.contents) +
                                   span.offset, sizeof handle);
        // Every recorded update also lands in the word map above, so a word that
        // is not there cannot be changed by one; no overlay scan is needed.
        return handle;
    };

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
            for (std::uint32_t element = 0; element != resource.array_length; ++element) {
                const auto handle = read_handle(resource, element);
                if (resource.kind == platform::CompiledShaderResourceKind::Sampler) {
                    id<MTLSamplerState> sampler = handle
                        ? backend.resources().sampler_for_id(
                              static_cast<std::int32_t>(handle_sampler_id(handle)),
                              registrations) : nil;
                    if (!sampler) {
                        const auto slot = resource.resource_id + element;
                        sampler = bound_sampler(slot, slot);
                    }
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
                if (handle) {
                    texture_id = static_cast<std::int32_t>(handle_texture_id(handle));
                    texture = sampled_texture(texture_id);
                }
                if (!texture) {
                    texture_id = bound_texture_id(resource.resource_id + element, image);
                    texture = texture_id ? sampled_texture(texture_id) : nil;
                }
                if (!texture)
                    return fail(ReplayErrorCode::missing_resource,
                                image ? "resolved storage image is unavailable"
                                      : "resolved sampled texture is unavailable");
                note_depth_barrier_read(texture);
                note_attachment_sample(texture);
                if (compute) note_compute_texture(texture);
                // A texture of another type than the shader declares reads
                // undefined values: bind the view in the declared type.
                if (const auto type = declared_texture_type(resource);
                    !image && type && texture.textureType != *type)
                    if (id<MTLTexture> fitted =
                            backend.resources().fitted_texture(texture_id, texture, *type))
                        texture = fitted;
                if (image && resource.access != platform::CompiledShaderResourceAccess::Read)
                    return fail(ReplayErrorCode::unsupported,
                                "storage image writes require descriptor-to-texture write tracking");
                if (!set_resource(resource, element, texture.gpuResourceID)) return false;
                if (resource_changed) make_resident(pending, texture);
            }
        }
    } else {
        for (const auto& binding : shader.textures) {
            if (binding.texture_index >= texture_slots)
                return fail(ReplayErrorCode::unsupported,
                            "legacy texture binding exceeds the replay limit");
            id<MTLTexture> texture = bound_texture(binding.texture_index, false);
            if (!texture)
                return fail(ReplayErrorCode::missing_resource,
                            "legacy sampled texture is unavailable");
            note_depth_barrier_read(texture);
            note_attachment_sample(texture);
            if (compute) note_compute_texture(texture);
            table_texture(table, binding.texture_index, texture.gpuResourceID._impl);
            make_resident(pending, texture);
            if (binding.sampler_index != shader::graphics_msl::TextureBinding::no_sampler) {
                if (binding.sampler_index >= 16)
                    return fail(ReplayErrorCode::unsupported, "legacy sampler index exceeds the Metal limit");
                id<MTLSamplerState> sampler = bound_sampler(binding.sampler_index, binding.texture_index);
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

bool Encoder::bind_draw_support(std::uint32_t base_instance) {
    const auto& program = state.stage_programs[NVN_SHADER_STAGE_VERTEX];
    if (!program || !program->stages[NVN_SHADER_STAGE_VERTEX] ||
        !program->stages[NVN_SHADER_STAGE_VERTEX]->draw_support_buffer_index) return true;
    const auto index = *program->stages[NVN_SHADER_STAGE_VERTEX]->draw_support_buffer_index;
    const auto support = upload(sizeof base_instance, &base_instance);
    if (!support) return false;
    write_buffer_slot(vertex_table, NVN_SHADER_STAGE_VERTEX, index, support.gpu);
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
                    constexpr std::uint32_t all_stage_bits = (1u << stage_count) - 1;
                    if (command.stages < 0 || (std::uint32_t(command.stages) & ~all_stage_bits)) break;
                    // The look-ahead never waits for a program whose stages
                    // are still being translated: its draws are not predicted
                    // (as for a failed stage) until another program is bound.
                    std::shared_ptr<const ResolvedProgram> program;
                    bool translating = false;
                    if (command.program) program = retain_ready_program(*command.program, translating);
                    if (command.program && !program && !translating) break;
                    for (std::size_t stage_index = 0; stage_index != stage_count; ++stage_index) {
                        const auto bit = 1u << stage_index;
                        if (!(std::uint32_t(command.stages) & bit)) continue;
                        shadow.stage_programs[stage_index] =
                            program && program->stages[stage_index] ? program : nullptr;
                        const bool unusable = translating || (program && (program->failed_stages & bit));
                        shadow.failed_stages = unusable ? shadow.failed_stages | bit : shadow.failed_stages & ~bit;
                    }
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
    const auto status = walk_commands(*pending.captured,
        [](const CommandRecordView& record, void* context) {
            return static_cast<Lookahead*>(context)->visit(record);
        }, &lookahead);
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
    if (!order_buffer_writes()) return false;
    if (!open_render(command.primitive)) return false;
    if (!bind_draw_support(static_cast<std::uint32_t>(command.base_instance))) return false;
    encode_draw(*primitive, command.first, command.count, command.instances, command.base_instance);
    note_draw_writes();
    ++backend.mutable_statistics().draws;
    ++pending.counters.draws;
    return true;
}

bool Encoder::draw_indexed(const DrawIndexed& command) {
    if (!resolve_carried_predicate()) return false;
    if (predicate_open() && predicate.used == predicate_capacity && !resolve_predicate()) return false;
    if (!state.render_enabled) return true;
    const auto primitive = primitive_type(command.primitive);
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
    if (!order_buffer_writes()) return false;
    const auto bytes = std::uint64_t(command.count) * element_size;
    state.index_buffer = {command.index_buffer, bytes};
    const auto source = buffer_address(command.index_buffer, bytes);
    if (!source.cpu || source.offset > source.length || bytes > source.length - source.offset)
        return fail(ReplayErrorCode::missing_resource, "index buffer range is unavailable");
    if (element_size >= 2 && (!state.primitive_restart ||
        (state.restart_index & (element_size == 2 ? 0xffffu : UINT32_MAX)) ==
            (element_size == 2 ? 0xffffu : UINT32_MAX))) {
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
        const bool written = std::any_of(other_buffer_writes.begin(), other_buffer_writes.end(),
            [&](const auto& range) {
                return command.index_buffer < range.first + range.second &&
                    range.first < command.index_buffer + bytes;
            });
        if (written && !synchronize_gpu()) return false;
        order_cpu_read(command.index_buffer, bytes, false);
        const auto* input = source.cpu;
        std::vector<std::uint32_t> indices(command.count);
        std::vector<std::pair<std::size_t, std::size_t>> segments;
        const auto restart = state.restart_index & (UINT32_MAX >> (32 - element_size * 8));
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
    buffer_write_order.write();
    return true;
}

bool Encoder::dispatch_indirect(const DispatchIndirect& command) {
    if (state.failed_stages & compute_stage_bit)
        return fail(ReplayErrorCode::missing_program, "program stage failed to translate");
    const auto& program = state.stage_programs[NVN_SHADER_STAGE_COMPUTE];
    if (!program || !program->stages[NVN_SHADER_STAGE_COMPUTE])
        return fail(ReplayErrorCode::missing_program,
                    "indirect dispatch has no resolved compute stage");
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
    if (!bind_stage(NVN_SHADER_STAGE_COMPUTE, shader, compute_table, false)) return false;
    if (new_compute_table)
        [pending.retained addObject:compute_table];
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
    buffer_write_order.write();
    return true;
}

bool Encoder::copy_buffer(const CopyBuffer& command) {
    if ((uniform_conflict(command.source, command.size) ||
         uniform_conflict(command.destination, command.size)) && !flush_uniforms()) return false;
    other_buffer_writes.emplace_back(command.destination, command.size);
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
    buffer_write_order.write();
    note_writes();
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
        buffer_write_order.write();
    }
    note_writes();
    ++backend.mutable_statistics().copies;
    return true;
}

bool Encoder::copy_texture_to_texture(const CopyTextureToTexture& command) {
    if (!command.source || !command.destination)
        return fail(ReplayErrorCode::invalid_command,
                    "texture copy has a missing source or destination");
    if (command.flags != NVN_COPY_FLAGS_NONE)
        return fail(ReplayErrorCode::unsupported,
                    "filtered or mirrored texture copies are unsupported");
    if (command.source_region.width != command.destination_region.width ||
        command.source_region.height != command.destination_region.height ||
        command.source_region.depth != command.destination_region.depth)
        return fail(ReplayErrorCode::unsupported,
                    "scaled texture copies require the two-dimensional blit path");
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
bool Encoder::visit(const CommandRecordView& record) {
    if (failed) return false;
    ++records_seen;
    if (visit_command(record)) return true;
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
        if (error.code == ReplayErrorCode::missing_resource) {
            detail = backend.resources().take_texture_failure();
        } else if (error.code == ReplayErrorCode::missing_program) {
            const auto& vertex = state.stage_programs[NVN_SHADER_STAGE_VERTEX];
            const auto& fragment = state.stage_programs[NVN_SHADER_STAGE_FRAGMENT];
            char line[160];
            std::snprintf(line, sizeof line,
                          "vertex %s fragment %s rasterizer discard %d colors %d depth %s",
                          vertex ? "bound" : "none", fragment ? "bound" : "none",
                          int(state.rasterizer_discard), int(state.targets.count),
                          state.targets.depth ? "bound" : "none");
            detail = line;
        }
    }
    backend.note_skip(error.detail, static_cast<unsigned>(record.op), detail);
    failed = false;
    error = {};
    // The render encoder may hold a partial setup; the next draw starts over.
    close_render(PassBreak::skip);
    return true;
}

bool Encoder::visit_command(const CommandRecordView& record) {
    if (failed) return false;
    current_op_ = record.op;
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
            if (!synchronize_gpu()) return false;
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
                command.type != NVN_COUNTER_TYPE_SAMPLES_PASSED)
                return fail(ReplayErrorCode::unsupported,
                            "requested hardware counter is unsupported by Metal replay");
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
            const auto command = payload<ReportValue>(record);
            const auto destination = buffer_at(command.address,
                                                                    sizeof(command.value));
            if (!destination.buffer || !destination.buffer.contents)
                return fail(ReplayErrorCode::missing_resource,
                            "reported value destination is not CPU-visible");
            if (uniform_conflict(command.address, sizeof command.value) && !flush_uniforms()) return false;
            const auto source = upload(sizeof command.value, &command.value);
            if (!source || !open_transfer()) return false;
            make_resident(pending, destination.buffer);
            copy_buffer_range(source.buffer, source.offset, destination.buffer, destination.offset,
                              sizeof command.value);
            other_buffer_writes.emplace_back(command.address, sizeof command.value);
            pending.buffer_writes.emplace_back(command.address, sizeof command.value);
            buffer_write_order.write();
            note_writes();
            return true;
        }
        case Op::copy_buffer: return copy_buffer(payload<CopyBuffer>(record));
        case Op::update_uniforms: {
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
            const auto* update_bytes = static_cast<const std::byte*>(record.payload) + header;
            if (!version_uniform(command.destination, command.size, update_bytes)) {
                if (failed || !flush_uniforms()) return false;
                record_uniform_update(command.destination, command.size, update_bytes);
                const auto source = upload(command.size, update_bytes);
                if (!source || !open_transfer()) return false;
                make_resident(pending, destination.buffer);
                copy_buffer_range(source.buffer, source.offset, destination.buffer, destination.offset,
                                  command.size);
                pending.buffer_writes.emplace_back(command.destination, command.size);
                buffer_write_order.write();
                note_writes();
                ++backend.mutable_statistics().copies;
            }
            {
                const auto* bytes = static_cast<const std::byte*>(record.payload) + header;
                const auto* contents = static_cast<const std::byte*>(destination.buffer.contents);
                const std::int64_t base = std::int64_t(destination.offset) - std::int64_t(command.destination);
                const std::uint64_t end = command.destination + command.size;
                for (std::uint64_t word = command.destination & ~std::uint64_t{3}; word < end; word += 4) {
                    std::uint32_t value = 0;
                    auto* value_bytes = reinterpret_cast<std::byte*>(&value);
                    if (word >= command.destination && word + 4 <= end) {
                        std::memcpy(&value, bytes + word - command.destination, 4);
                        uniform_words[word] = value;
                        continue;
                    }
                    if (const auto found = uniform_words.find(word); found != uniform_words.end()) {
                        value = found->second;
                    } else if (contents) {
                        order_cpu_read(word, 4, true);
                        for (unsigned byte = 0; byte < 4; ++byte) {
                            const std::int64_t at = base + std::int64_t(word + byte);
                            if (at >= 0 && std::uint64_t(at) < destination.buffer.length)
                                value_bytes[byte] = contents[at];
                        }
                    }

                    for (unsigned byte = 0; byte < 4; ++byte)
                        if (word + byte >= command.destination && word + byte < end)
                            value_bytes[byte] = bytes[word + byte - command.destination];
                    uniform_words[word] = value;
                }
            }
            invalidate_uniforms(command.destination, command.size);
            ++pending.counters.uniform_invalidations;
            return true;
        }
        case Op::clear_buffer: {
            const auto command = payload<ClearBuffer>(record);
            if (uniform_conflict(command.address, command.size) && !flush_uniforms()) return false;
            other_buffer_writes.emplace_back(command.address, command.size);
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
            buffer_write_order.write();
            note_writes();
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
            std::shared_ptr<const ResolvedProgram> program;
            if (command.program) program = retain_program(*command.program);
            if (command.program && !program)
                return fail(ReplayErrorCode::missing_program,
                            "bound program serial has no resolved artifacts");
            constexpr std::uint32_t all_stage_bits = (1u << stage_count) - 1;
            if (command.stages < 0 || (std::uint32_t(command.stages) & ~all_stage_bits))
                return fail(ReplayErrorCode::invalid_command,
                            "program binding contains an invalid stage mask");
            bool changed = false;
            for (std::size_t stage_index = 0; stage_index != stage_count; ++stage_index) {
                const auto bit = 1u << stage_index;
                if (!(std::uint32_t(command.stages) & bit)) continue;
                const auto* next = program && program->stages[stage_index] ? program.get() : nullptr;
                if (state.stage_programs[stage_index].get() != next) {
                    state.stage_programs[stage_index] = program && program->stages[stage_index]
                        ? program : nullptr;
                    changed = true;
                }
                const auto failed = program ? program->failed_stages & bit : 0u;
                if ((state.failed_stages & bit) != failed) {
                    state.failed_stages = (state.failed_stages & ~bit) | failed;
                    changed = true;
                }
            }
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
                if (uniform_conflict(command.ranges[index].address, command.ranges[index].size) &&
                    !flush_uniforms()) return false;
            if (replace_state(state.vertex_buffers.data() + command.first, command.ranges,
                              command.count * sizeof(NVNbufferRange))) dirty.vertices = true;
            return true;
        }
        case Op::bind_uniform_buffer:
        case Op::bind_storage_buffer: {
            const auto command = payload<BindBuffer>(record);
            if (command.stage < 0 || command.stage >= int(stage_count) || command.index < 0 ||
                command.index >= int(buffer_slots))
                return fail(ReplayErrorCode::invalid_command,
                            "shader buffer binding range is invalid");
            if (record.op == Op::bind_storage_buffer) {
                if (uniform_conflict(command.address, command.size) && !flush_uniforms()) return false;
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
                invalidate_args(std::size_t(command.stage));
                if (record.op == Op::bind_storage_buffer || command.stage == NVN_SHADER_STAGE_COMPUTE) {
                    ++bindings[command.stage].driver_generation[0];
                    bindings[command.stage].windows[0] = {};
                }
            }
            return true;
        }
        case Op::bind_textures:
        case Op::bind_samplers:
        case Op::bind_images: {
            const auto command = payload<BindHandles>(record);
            const std::size_t limit = record.op == Op::bind_images
                ? image_slots : record.op == Op::bind_samplers ? sampler_slots : texture_slots;
            if (command.stage < 0 || command.stage >= int(stage_count) || command.first < 0 ||
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
                invalidate_args(std::size_t(command.stage));
                if (record.op == Op::bind_textures) {
                    ++bindings[command.stage].driver_generation[2];
                    bindings[command.stage].windows[2] = {};
                }
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
            // Keep depth-only writes on tile for subsequent depth tests. Any
            // later shader read of that depth closes the pass before drawing.
            // Other passes retain the conservative full encoder boundary.
            if (render && pass_depth_only && !color_aliases.any &&
                !backend.resources().texture_is_in_place(*state.targets.depth) &&
                (payload<ScalarInt>(record).value & ~0xff) == 0) {
                depth_barrier_allocation = render_descriptor.depthAttachment.texture;
                while (id<MTLTexture> parent = depth_barrier_allocation.parentTexture)
                    depth_barrier_allocation = parent;
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
            bool handled = false;
            if (!gpu_predicate(command, handled)) return false;
            note_conditional(handled, command.mode == NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL);
            if (handled) {
                state.render_enabled = true;
                return true;
            }
            if (!synchronize_gpu()) return false;
            return read_predicate(command.mode, command.address);
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
    void set_completion_hook(std::function<bool()> hook) override {
        resources_.set_completion_hook(std::move(hook));
    }
    BufferSpan buffer_at(std::uint64_t address, std::uint64_t size) override {
        return resources_.buffer_at(address, size);
    }
    BufferAddress buffer_address(std::uint64_t address, std::uint64_t size) override {
        return resources_.buffer_address(address, size);
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
    id<MTLTexture> texture_for_id_stamped(std::int32_t id, DescriptorStamp& stamp) override {
        return resources_.texture_for_id(id, stamp);
    }
    bool texture_stamp_current(std::int32_t id, const DescriptorStamp& stamp) const override {
        return resources_.texture_stamp_current(id, stamp);
    }
    id<MTLSamplerState> sampler_for_id(std::int32_t id) override { return resources_.sampler_for_id(id); }
    id<MTLSamplerState> sampler_for_id(std::int32_t id, std::uint64_t registrations) override {
        return resources_.sampler_for_id(id, registrations);
    }
    id<MTLTexture> fitted_texture(std::int32_t texture_id, id<MTLTexture> texture,
                                  MTLTextureType type) override {
        return resources_.fitted_texture(texture_id, texture, type);
    }
    void begin_walk() override { resources_.begin_walk(); }
    bool flush_uploads() override { return resources_.flush_uploads(); }
    void commit_uploads(UploadFence& fence) override { resources_.commit_uploads(fence); }
    bool uploads_failed(const UploadFence& fence) override { return resources_.uploads_failed(fence); }
    void note_gpu_write(const nvn::TextureState& state) override { resources_.note_gpu_write(state); }
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
private:
    Resources& resources_;
};
}
std::unique_ptr<MetalReplayBackend>
create_replay_backend(id<MTLDevice> device, id<MTL4CommandQueue> queue, Resources& resources,
                      ReplaySchedulingPolicy scheduling) {
    if (!device || !queue || ![device supportsFamily:MTLGPUFamilyMetal4] ||
        !device.hasUnifiedMemory) return {};
    return std::make_unique<ReplayImpl>(device, queue, std::make_shared<ResourceLookup>(resources), scheduling);
}
#endif

} // namespace armigo::platform::metal::nvn_driver
