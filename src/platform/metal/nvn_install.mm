// ============================================================================
// Armigo Platform - NVN Metal Driver Installation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "nvn_install.h"

#include "nvn_present.h"
#include "nvn_replay.h"
#include "nvn_resources.h"
#include <armigo/platform/metal_shader_package.h>
#include "nvn/backend.h"
#include "nvn/entry_table.h"
#include "nvn/host_stack.h"
#include "nvn/present.h"
#include "nvn/programs.h"

#include <armigo/services/video_interface.h>
#include <armigo/services/vi_producer.h>

#import <Metal/Metal.h>

#include <filesystem>

#ifdef ARMIGO_HAS_NATIVE_SHADERS
#include "nvn_native_stages.h"
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>

extern "C" std::uintptr_t armigo_execution_gate_callback_stack() noexcept;

namespace armigo::nvn {
void register_resource_entries();
} // namespace armigo::nvn

namespace armigo::platform::metal::nvn_driver {
namespace {

// The NVN entry table is process-wide, so the driver it points at is too.
struct Installed {
    std::unique_ptr<Resources> resources;
    std::unique_ptr<MetalReplayBackend> replay;
};
Installed& installed() {
    static auto* value = new Installed;
    return *value;
}

std::mutex display_mutex;
std::weak_ptr<services::vi::DisplaySession> display;

using SetNativeWindow = nvn::EntrySignature<nvn::Entry::nvnWindowBuilderSetNativeWindow>::type;
SetNativeWindow set_native_window = nullptr;

void bind_native_window(nvn::api::NVNwindowBuilder* builder, void* native_window) {
    std::shared_ptr<services::vi::DisplaySession> session;
    {
        const std::lock_guard lock(display_mutex);
        session = display.lock();
    }
    if (session && native_window && !services::vi::find_native_window(native_window))
        static_cast<void>(session->bind_native_window(native_window));
    set_native_window(builder, native_window);
}

} // namespace

namespace {

// Where the pipeline warm-up reads and writes its state. The environment
// overrides the title's configuration:
//   ARMIGO_PSO_MANIFEST  seed manifest path, or "off"
//   ARMIGO_PSO_USER      user manifest path (default: in the cache directory), or "off"
//   ARMIGO_PSO_RECORD    file that receives every stage and pipeline in first-use order
//   ARMIGO_PSO_CACHE     cache directory (default Armigo/<TitleID>/pso in the user's caches), or "off"
//   ARMIGO_PSO_ARCHIVE   pipelines come from the binary archives in the cache directory (an offline
//                        build puts them there) without compiling; "capture" (or 1) also captures this
//                        session's pipelines into a new archive; "off" (or 0) ignores the archives
//   ARMIGO_PSO_WARM=0 no background build
//   ARMIGO_PSO_WARM_THREADS  background build threads (default 3)
PipelineWarmupConfig warmup_config(const std::shared_ptr<MetalShaderPackage>& package,
                                   const std::string& title_manifest, const std::string& title_cache) {
    const auto text = [](const char* name) -> std::optional<std::string> {
        const char* value = std::getenv(name);
        if (!value || !*value) return std::nullopt;
        return std::string(value);
    };
    PipelineWarmupConfig config;
    config.package = package;
    config.seed_manifest = text("ARMIGO_PSO_MANIFEST").value_or(title_manifest);
    if (config.seed_manifest == "off") config.seed_manifest.clear();
    config.record_manifest = text("ARMIGO_PSO_RECORD").value_or("");
    std::string cache = text("ARMIGO_PSO_CACHE").value_or("");
    if (cache.empty() && !title_cache.empty()) cache = title_cache + "/pso";
    if (cache.empty()) {
        const char* home = std::getenv("HOME");
        if (home && *home) cache = std::string(home) + "/Library/Caches/Armigo/pso";
    }
    if (cache != "off" && !cache.empty()) {
        const auto chain = package ? package->chain_id().substr(0, 16) : std::string();
        config.user_manifest = text("ARMIGO_PSO_USER").value_or(cache + "/user-" + chain + ".tsv");
        const auto archive = text("ARMIGO_PSO_ARCHIVE");
        // Archives are looked up by default. Capturing is opt-in: while Metal
        // captures into an archive its compiles bypass the system shader cache.
        if (!archive || (*archive != "off" && *archive != "0")) {
            config.archive_root = cache;
            // A packed app's archive, built when it was packed.
            @autoreleasepool {
                if (NSString* resources = NSBundle.mainBundle.resourcePath) {
                    const auto packed = std::string(resources.fileSystemRepresentation) + "/Armigo/pipelines";
                    std::error_code error;
                    if (std::filesystem::is_directory(packed, error)) config.archive_bundle_root = packed;
                }
            }
        }
        config.archive_capture = archive && (*archive == "1" || *archive == "capture");
    } else {
        config.user_manifest = text("ARMIGO_PSO_USER").value_or("");
    }
    if (config.user_manifest == "off") config.user_manifest.clear();
    if (const auto warm = text("ARMIGO_PSO_WARM"); warm && *warm == "0") config.warm = false;
    if (const auto threads = text("ARMIGO_PSO_WARM_THREADS"))
        config.warm_threads = static_cast<unsigned>(std::max(1, std::atoi(threads->c_str())));
    return config;
}

} // namespace

std::string install_driver(std::shared_ptr<MetalShaderPackage> package, const std::string& pso_manifest,
                           const std::string& title_cache) {
    auto& driver = installed();
    if (driver.replay) return "The NVN Metal driver is already installed.";
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return "No Metal device is available for the NVN driver.";
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    if (!queue) return "The NVN driver could not create a Metal 4 command queue.";
    auto submission = std::make_shared<Metal4SubmissionOwner>(device, queue);
    auto resources = std::make_unique<Resources>(device, submission);
    auto replay = create_replay_backend(device, queue, *resources);
    if (!replay) return "The NVN driver could not create its replay backend.";
    replay->configure_pipeline_warmup(warmup_config(package, pso_manifest, title_cache));
    nvn::set_backends({resources.get(), replay.get()});
    // Driver callbacks run on the host stack of the gate, not the guest's.
    if (!std::getenv("ARMIGO_NVN_GUEST_STACK"))
        nvn::set_host_stack_provider(&armigo_execution_gate_callback_stack);
#ifdef ARMIGO_HAS_NATIVE_SHADERS
    if (package) nvn::set_native_shader_package(package->directory());
#endif
    nvn::set_program_package(std::move(package), title_cache);
    if (const auto version = nvn::selected_api_version(); version && version->major == 231) {
#ifdef ARMIGO_HAS_NATIVE_SHADERS
        nvn::set_native_program_resolver();
#else
        armigo::diag::write(armigo::diag::Level::error,
                            "Armigo NVN 231.40: this build has no native shader front end; programs fail\n");
        nvn::set_program_resolver(nullptr, nullptr);
#endif
    }
    nvn::set_texture_exporter(make_texture_exporter(device, *resources, *replay));
    nvn::register_resource_entries();
    nvn::register_program_entries();
    nvn::register_state_entries();
    nvn::register_command_entries();
    nvn::register_present_entries();
    set_native_window = reinterpret_cast<SetNativeWindow>(
        nvn::installed_entry(nvn::Entry::nvnWindowBuilderSetNativeWindow));
    nvn::install<nvn::Entry::nvnWindowBuilderSetNativeWindow>(&bind_native_window);
    driver.resources = std::move(resources);
    driver.replay = std::move(replay);
    armigo::diag::write(armigo::diag::Level::info,  "Armigo NVN: Metal driver on %s.\n", device.name.UTF8String);
    return {};
}

bool stop_submission_workers(std::optional<std::chrono::steady_clock::time_point> deadline) {
    auto& driver = installed();
    return !driver.replay || driver.replay->stop_submission_workers(deadline);
}

std::string failure_detail() {
    const auto* replay = nvn::backends().replay;
    return replay ? replay->fatal_error_detail() : std::string{};
}

void bind_native_windows(std::weak_ptr<services::vi::DisplaySession> session) {
    const std::lock_guard lock(display_mutex);
    display = std::move(session);
}

} // namespace armigo::platform::metal::nvn_driver
