// ============================================================================
// Armigo NVN - Driver Backends
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/nvn/driver.h>
#include "../../nvn/programs.h"
#include <armigo/platform/metal_shader_package.h>
#include <armigo/shader_input/shader_input.h>
#include <armigo/shader_translation/stage_cache.h>
#include <armigo/shader_translation/stage_translation.h>
#include <mach/mach.h>
#include "../mach_vm.h"
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace armigo::nvn {
namespace {
std::mutex package_mutex;
std::shared_ptr<platform::MetalShaderPackage> shader_package;

bool read_memory(std::uint64_t address, std::span<std::byte> bytes) {
    if (!address || address > std::numeric_limits<std::uint64_t>::max() - bytes.size()) return false;
    mach_vm_size_t actual = 0;
    return mach_vm_read_overwrite(mach_task_self(), address, bytes.size(),
        reinterpret_cast<mach_vm_address_t>(bytes.data()), &actual) == KERN_SUCCESS && actual == bytes.size();
}
std::uint32_t word(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

// ARMIGO_SHADER_MISS_DIR=DIR writes each stage the package does not hold to
// DIR/<stage>-<code size>-<code checksum>/{control.bin,code.bin}, the layout an
// offline package build reads, so a title's missing stages can be prepared.
void record_missing_stage(std::span<const std::byte> control, std::span<const std::byte> code, unsigned stage) {
    static const char* directory = std::getenv("ARMIGO_SHADER_MISS_DIR");
    if (!directory || !*directory) return;
    std::uint64_t checksum = 1469598103934665603ull;  // FNV-1a over the code
    for (const auto byte : code) checksum = (checksum ^ std::to_integer<unsigned>(byte)) * 1099511628211ull;
    char name[64];
    std::snprintf(name, sizeof name, "%u-%zu-%016llx", stage, code.size(), static_cast<unsigned long long>(checksum));
    std::error_code ignored;
    const auto path = std::filesystem::path(directory) / name;
    std::filesystem::create_directories(path, ignored);
    std::ofstream(path / "control.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(control.data()), static_cast<std::streamsize>(control.size()));
    std::ofstream(path / "code.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(code.data()), static_cast<std::streamsize>(code.size()));
    armigo::diag::write(armigo::diag::Level::warning,  "Armigo NVN program: recorded missing stage %s\n", name);
}

// Run-time preparation. A stage the package does not hold, or every stage when
// no package is given, is translated in-process with the options the offline
// package builds use: the texture-handle constant buffer 2 and derived texture
// descriptors; vertex stages take the device's depth convention; a fragment
// stage is first translated without a position convention and again with the
// reciprocal-clip-w one when the first translation says it needs one; compute
// stages take their local size (control+0x748) and shared memory
// (control+0x754) from the control. The stage carries its Metal source, which
// the replay compiles off the guest thread when a pipeline first uses it.
// ARMIGO_SHADER_RUNTIME=0 turns this off. Translations are kept in
// Armigo/<TitleID>/shaders in the user's caches, or per title in
// ARMIGO_SHADER_CACHE when it names a directory ("off" keeps them for the run
// only). Like a package's stages, a title's cached stages are
// prepared by background threads when the driver starts, and their Metal
// source stays on disk until a pipeline compiles it.
//
// A stage that has to be translated is translated in the background: SetShaders
// queues it for a pool of worker threads and publishes the program with the
// stage pending, so a guest that sets many programs at once goes on while the
// pool translates them in parallel; the program waits for the stage when the
// replay first binds it. A waiter whose stage no worker has started translates
// it itself, so a stage the replay needs never waits behind the queue. The
// queue holds at most a few stages per worker; a guest that sets programs
// faster than the pool translates waits for room, so pending stages never hold
// much guest code in memory. ARMIGO_SHADER_THREADS sets the pool size (default:
// the performance cores less two); 0 translates on the guest thread during
// SetShaders, as when each stage was prepared before the call returned.
namespace translation = shader::translation;
constexpr std::size_t compute_local_size_offset = 0x748;
constexpr std::size_t compute_shared_memory_offset = 0x754;
constexpr unsigned preload_threads = 8;
constexpr std::size_t queued_stages_per_thread = 4;
constexpr std::size_t translation_stack_bytes = 64u << 20;

using StagePointer = std::shared_ptr<const platform::CompiledGraphicsStage>;

// One stage to translate. Whoever claims it first (a worker, or a thread that
// waits for it before any worker has) translates it; everyone else waits.
struct TranslationJob final : PendingStage, std::enable_shared_from_this<TranslationJob> {
    std::string key;
    std::vector<std::byte> control, code;
    shader_input::Stage stage = shader_input::Stage::Vertex;
    translation::StageOptions options;
    std::atomic<bool> claimed{false};
    std::mutex mutex;
    std::condition_variable done;
    bool finished = false;
    StagePointer result;
    std::string error;
    StagePointer wait() override;
};

struct RuntimeStages {
    bool enabled = true;
    unsigned threads = 0;
    // ARMIGO_SHADER_CACHE's directory, which keeps one subdirectory per title.
    std::filesystem::path root;
    // Without it, the title cache's shaders/ ("off" turns both off).
    bool title_cache = true;
    std::optional<translation::StageCache> cache;
    // Guards the maps and counters below: the resolver and the translation
    // workers both fill them.
    std::mutex mutex;
    std::map<std::string, StagePointer> prepared;
    std::map<std::string, std::string> failed;
    std::map<std::string, std::shared_ptr<TranslationJob>> translating;
    std::size_t translated = 0, loaded = 0, failures = 0, reported = 0;
    // With a package: the stages it lacks that run-time preparation took on.
    bool package = false;
    std::size_t not_in_package = 0;
    double translate_ms = 0;
    std::optional<std::chrono::steady_clock::time_point> first_translation;
    // Filled by the preload threads; the resolver takes stages from it.
    std::mutex preload_mutex;
    std::map<std::string, StagePointer> preloaded;
    std::vector<std::thread> preloaders;
    ~RuntimeStages();
};

unsigned default_translation_threads() {
    int cores = 0;
    std::size_t size = sizeof cores;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, nullptr, 0) != 0 || cores <= 0)
        cores = static_cast<int>(std::thread::hardware_concurrency() / 2);
    return static_cast<unsigned>(std::max(cores - 2, 1));
}

RuntimeStages& runtime_stages() {
    static RuntimeStages state;
    static std::once_flag configured;
    std::call_once(configured, [] {
        const char* enabled = std::getenv("ARMIGO_SHADER_RUNTIME");
        state.enabled = !(enabled && std::string_view(enabled) == "0");
        state.threads = default_translation_threads();
        if (const char* threads = std::getenv("ARMIGO_SHADER_THREADS"); threads && *threads)
            state.threads = static_cast<unsigned>(std::min(std::strtoul(threads, nullptr, 10), 64ul));
        const char* root = std::getenv("ARMIGO_SHADER_CACHE");
        if (root && *root) {
            if (std::string_view(root) != "off") state.root = root;
            else state.title_cache = false;
        }
    });
    return state;
}

platform::SourceStageResult make_cached(const translation::CachedStage& cached) {
    return platform::make_source_stage(cached.stage, cached.program,
        {reinterpret_cast<const std::byte*>(cached.reflection.data()), cached.reflection.size()}, {},
        cached.source_file.string());
}

// Prepares every cached stage of the title on background threads, so the
// resolver finds them ready instead of reading them on the guest thread.
void start_preload(RuntimeStages& state) {
    if (!state.cache || !state.enabled) return;
    auto keys = std::make_shared<std::vector<std::string>>(state.cache->keys());
    if (keys->empty()) return;
    auto next = std::make_shared<std::atomic<std::size_t>>(0);
    auto finished = std::make_shared<std::atomic<unsigned>>(0);
    auto unusable = std::make_shared<std::atomic<std::size_t>>(0);
    const auto begin = std::chrono::steady_clock::now();
    for (unsigned worker = 0; worker != preload_threads; ++worker) {
        state.preloaders.emplace_back([&state, keys, next, finished, unusable, begin] {
            for (auto index = next->fetch_add(1); index < keys->size(); index = next->fetch_add(1)) {
                @autoreleasepool {
                    const auto& key = (*keys)[index];
                    auto cached = state.cache->load(key);
                    if (!cached) { unusable->fetch_add(1); continue; }
                    auto made = make_cached(*cached);
                    auto* ready = std::get_if<StagePointer>(&made);
                    if (!ready) { unusable->fetch_add(1); continue; }
                    const std::lock_guard lock(state.preload_mutex);
                    state.preloaded.emplace(key, std::move(*ready));
                }
            }
            if (finished->fetch_add(1) + 1 == preload_threads)
                armigo::diag::write(armigo::diag::Level::info,  "Armigo shader runtime: prepared %zu cached stages in %.1f s (%zu unusable)\n",
                             keys->size() - unusable->load(),
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(),
                             unusable->load());
        });
    }
}

// Translation recurses deeply, so it runs on a thread with a large stack
// rather than on the guest thread, whose host stack a driver callback shares.
translation::TranslatedStage translate_on_worker(std::span<const std::byte> control,
                                                 std::span<const std::byte> code,
                                                 const translation::StageOptions& options) {
    struct Job {
        std::span<const std::byte> control, code;
        const translation::StageOptions* options;
        std::optional<translation::TranslatedStage> result;
        std::exception_ptr error;
    } job{control, code, &options, std::nullopt, nullptr};
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, translation_stack_bytes);
    pthread_t thread;
    const auto body = [](void* raw) -> void* {
        auto& work = *static_cast<Job*>(raw);
        try {
            work.result = translation::translate_stage(work.control, work.code, *work.options);
        } catch (...) {
            work.error = std::current_exception();
        }
        return nullptr;
    };
    const int created = pthread_create(&thread, &attributes, body, &job);
    pthread_attr_destroy(&attributes);
    if (created != 0) throw std::runtime_error("The shader translation thread could not start");
    pthread_join(thread, nullptr);
    if (job.error) std::rethrow_exception(job.error);
    return std::move(*job.result);
}

const char* stage_name(shader_input::Stage stage) {
    return stage == shader_input::Stage::Vertex ? "vertex" :
           stage == shader_input::Stage::Fragment ? "fragment" : "compute";
}

// Called with the state's mutex held.
void report_runtime(RuntimeStages& state) {
    const auto total = state.translated + state.loaded;
    if (total == state.reported || (total != 1 && total != 10 && total != 100 && total % 500)) return;
    state.reported = total;
    const double since = state.first_translation
        ? std::chrono::duration<double>(std::chrono::steady_clock::now() - *state.first_translation).count() : 0.0;
    char package[48] = "";
    if (state.package) std::snprintf(package, sizeof package, ", %zu not in the package", state.not_in_package);
    armigo::diag::write(armigo::diag::Level::info,  "Armigo shader runtime: %zu stages (%zu translated, %zu from cache, %zu failed%s), "
                         "%.1f ms per translation, %.1f s since the first translation\n",
                 total, state.translated, state.loaded, state.failures, package,
                 state.translated ? state.translate_ms / static_cast<double>(state.translated) : 0.0, since);
}

void stop_translation() noexcept;

// Translates a claimed job, keeps the result (cache, prepared or failed) and
// wakes its waiters. On a translation worker, whose stack is large, the
// translator runs directly; elsewhere on its own large-stack thread.
void run_job(TranslationJob& job, bool large_stack) noexcept {
    auto& state = runtime_stages();
    const auto translate = [&](const translation::StageOptions& options) {
        return large_stack ? translation::translate_stage(job.control, job.code, options)
                           : translate_on_worker(job.control, job.code, options);
    };
    const auto begin = std::chrono::steady_clock::now();
    StagePointer result;
    std::string error;
    try {
        std::optional<translation::TranslatedStage> translated;
        try {
            try {
                translated = translate(job.options);
            } catch (const std::exception& first) {
                std::string message = first.what();
                std::transform(message.begin(), message.end(), message.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                // The position convention is part of the stage's identity; only a
                // stage that needs one is translated with it.
                if (job.stage != shader_input::Stage::Fragment ||
                    message.find("position convention") == std::string::npos)
                    throw;
                auto options = job.options;
                options.emission.fragment_position =
                    shader::graphics_msl::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
                translated = translate(options);
            }
        } catch (const std::exception& failure) {
            error = std::string("Run-time shader translation failed: ") + failure.what();
        }
        // The workers stop at exit before the translator's statics are
        // destroyed: exit runs handlers and destructors in reverse order of
        // registration, so the handler is registered after the first attempt
        // (whatever its outcome) and again after the first translation that
        // succeeded, which has constructed every static on the emission path.
        static std::once_flag stop_after_attempt, stop_after_success;
        std::call_once(stop_after_attempt, [] { std::atexit(stop_translation); });
        if (translated) std::call_once(stop_after_success, [] { std::atexit(stop_translation); });
        if (translated) {
            // A stored translation keeps its source on disk; otherwise the stage
            // holds it in memory for the run.
            const bool stored = state.cache && state.cache->store(job.key, *translated);
            if (state.cache && !stored) {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true))
                    armigo::diag::write(armigo::diag::Level::warning,  "Armigo shader runtime: the stage cache %s cannot be written; "
                                         "translated stages keep their source in memory and are translated again "
                                         "next run\n",
                                 state.cache->directory().string().c_str());
            }
            auto made = platform::make_source_stage(job.stage, translated->program,
                {reinterpret_cast<const std::byte*>(translated->reflection.data()), translated->reflection.size()},
                stored ? std::string() : std::move(translated->source),
                stored ? (state.cache->directory() / job.key / "source.metal").string() : std::string());
            if (auto* ready = std::get_if<StagePointer>(&made)) result = std::move(*ready);
            else error = "Run-time shader stage was rejected: " + std::get<platform::MetalShaderPackageError>(made).detail;
        }
        if (!result) record_missing_stage(job.control, job.code, static_cast<unsigned>(job.stage));
    } catch (const std::exception& failure) {
        result = nullptr;
        error = std::string("Run-time shader preparation failed: ") + failure.what();
    } catch (...) {
        result = nullptr;
        error = "Run-time shader preparation failed";
    }
    try {
        const std::lock_guard lock(state.mutex);
        state.translate_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        ++state.translated;
        if (result) {
            state.prepared.emplace(job.key, result);
        } else {
            if (++state.failures <= 8)
                armigo::diag::write(armigo::diag::Level::error,  "Armigo shader runtime: %s stage failed: %s\n", stage_name(job.stage),
                             error.c_str());
            state.failed.emplace(job.key, error);
        }
        state.translating.erase(job.key);
        report_runtime(state);
    } catch (...) {
    }
    // The guest code is no longer needed; a program may hold the job until the
    // replay first binds it.
    std::vector<std::byte>().swap(job.control);
    std::vector<std::byte>().swap(job.code);
    {
        const std::lock_guard lock(job.mutex);
        job.result = std::move(result);
        job.error = std::move(error);
        job.finished = true;
    }
    job.done.notify_all();
}

// The translation workers. The queue is never destroyed (the workers are
// detached); the state's destructor stops them before the state goes away.
struct TranslationQueue {
    std::mutex mutex;
    std::condition_variable work, room, idle;
    std::deque<std::shared_ptr<TranslationJob>> jobs;
    unsigned started = 0, running = 0;
    bool stopping = false;
};

TranslationQueue& translation_queue() {
    static auto& queue = *new TranslationQueue;
    return queue;
}

StagePointer TranslationJob::wait() {
    const auto self = shared_from_this();
    if (!claimed.exchange(true)) {
        // Translated by the waiting thread (the replay, or the guest when
        // there are no workers): counted as running, so stopping at exit
        // waits for it like for a worker; once stopping, it fails instead.
        auto& queue = translation_queue();
        bool run = false;
        {
            const std::lock_guard lock(queue.mutex);
            if (!queue.stopping) {
                ++queue.running;
                run = true;
            }
        }
        if (run) {
            @autoreleasepool {
                run_job(*this, false);
            }
            {
                const std::lock_guard lock(queue.mutex);
                --queue.running;
            }
            queue.idle.notify_all();
        } else {
            {
                const std::lock_guard lock(mutex);
                error = "Run-time shader translation stopped at exit";
                finished = true;
            }
            done.notify_all();
        }
    }
    std::unique_lock lock(mutex);
    done.wait(lock, [&] { return finished; });
    return result;
}

void* translation_worker(void*) {
    pthread_setname_np("Armigo shader translation");
    auto& queue = translation_queue();
    for (;;) {
        std::shared_ptr<TranslationJob> job;
        {
            std::unique_lock lock(queue.mutex);
            queue.work.wait(lock, [&] { return queue.stopping || !queue.jobs.empty(); });
            if (queue.stopping) break;
            job = std::move(queue.jobs.front());
            queue.jobs.pop_front();
            ++queue.running;
        }
        queue.room.notify_one();
        if (!job->claimed.exchange(true)) {
            @autoreleasepool {
                run_job(*job, true);
            }
        }
        job.reset();
        {
            const std::lock_guard lock(queue.mutex);
            --queue.running;
        }
        queue.idle.notify_all();
    }
    {
        const std::lock_guard lock(queue.mutex);
        --queue.started;
    }
    queue.idle.notify_all();
    return nullptr;
}

enum class Queued { yes, no_workers, stopping };

// Queues a job, starting the workers the first time; waits while the queue is
// full. With no_workers the caller translates the job itself; with stopping
// (the process exits) nothing translates it.
Queued queue_translation(std::shared_ptr<TranslationJob> job, unsigned threads) {
    auto& queue = translation_queue();
    std::unique_lock lock(queue.mutex);
    if (queue.stopping) return Queued::stopping;
    while (queue.started < threads) {
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setstacksize(&attributes, translation_stack_bytes);
        pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
        pthread_attr_set_qos_class_np(&attributes, QOS_CLASS_USER_INITIATED, 0);
        pthread_t thread;
        const int created = pthread_create(&thread, &attributes, translation_worker, nullptr);
        pthread_attr_destroy(&attributes);
        if (created != 0) break;
        ++queue.started;
    }
    if (!queue.started) return Queued::no_workers;
    const auto limit = queued_stages_per_thread * queue.started;
    queue.room.wait(lock, [&] { return queue.stopping || queue.jobs.size() < limit; });
    if (queue.stopping) return Queued::stopping;
    queue.jobs.push_back(std::move(job));
    lock.unlock();
    queue.work.notify_one();
    return Queued::yes;
}

// Stops the workers at exit: queued jobs are finished as failures (nobody
// waits for them any more) and the jobs that are running are waited for.
// Runs from an exit handler and again, doing nothing, when the state is
// destroyed.
void stop_translation() noexcept {
    auto& queue = translation_queue();
    std::deque<std::shared_ptr<TranslationJob>> abandoned;
    {
        std::unique_lock lock(queue.mutex);
        queue.stopping = true;
        abandoned.swap(queue.jobs);
    }
    queue.work.notify_all();
    queue.room.notify_all();
    for (auto& job : abandoned) {
        if (job->claimed.exchange(true)) continue;
        {
            const std::lock_guard lock(job->mutex);
            job->error = "Run-time shader translation stopped at exit";
            job->finished = true;
        }
        job->done.notify_all();
    }
    std::unique_lock lock(queue.mutex);
    queue.idle.wait(lock, [&] { return queue.running == 0; });
}

RuntimeStages::~RuntimeStages() {
    stop_translation();
    for (auto& thread : preloaders) if (thread.joinable()) thread.join();
}

// Returns the stage when it is ready now. Otherwise either sets `pending` to
// the background translation that will prepare it, or returns nullptr with
// `error` set when it cannot be prepared. `package_miss` is why the configured
// package does not hold the stage (null without a package); the first time a
// run prepares such a stage it is counted, and the first sixteen are logged.
StagePointer prepare_runtime_stage(std::vector<std::byte>&& control, std::vector<std::byte>&& code,
                                   shader_input::Stage stage, shader::graphics_msl::VertexDepthConvention depth,
                                   std::string& error, std::shared_ptr<PendingStage>& pending,
                                   const std::string* package_miss) {
    auto& state = runtime_stages();
    const auto refuse = [&](std::string message) -> StagePointer {
        record_missing_stage(control, code, static_cast<unsigned>(stage));
        error = std::move(message);
        return nullptr;
    };
    translation::StageOptions options;
    options.texture_handle_cbuf = 2;
    options.texture_descriptors_derived = true;
    std::string text = "ported tc=2 td=derived stage=" + std::to_string(static_cast<unsigned>(stage));
    // The title's opt-in changes the translation, so it is part of the key.
    if (nvn::title_policy().hoist_implicit_samples) {
        options.hoist_implicit_samples = true;
        text += " hi=1";
    }
    if (stage == shader_input::Stage::Vertex) {
        options.emission.vertex_depth = depth;
        text += " vd=" + std::to_string(static_cast<unsigned>(depth));
        // ARMIGO_VERTEX_FETCH=1: vertex stages read their inputs through a
        // per-draw fetch table instead of a vertex descriptor.
        static const bool vertex_fetch = [] {
            const char* value = std::getenv("ARMIGO_VERTEX_FETCH");
            return value && std::string_view(value) == "1";
        }();
        if (vertex_fetch) {
            options.emission.vertex_fetch = true;
            text += " vf=1";
        }
    } else if (stage == shader_input::Stage::Compute) {
        if (control.size() < compute_shared_memory_offset + 4)
            return refuse("Compute shader control is too short for its launch facts");
        std::array<std::uint32_t, 3> local{};
        for (unsigned axis = 0; axis != 3; ++axis) local[axis] = word(control, compute_local_size_offset + 4 * axis);
        options.compute_local_size = local;
        options.compute_shared_memory_bytes = word(control, compute_shared_memory_offset);
        text += " cl=" + std::to_string(local[0]) + ',' + std::to_string(local[1]) + ',' +
                std::to_string(local[2]) + " cs=" + std::to_string(*options.compute_shared_memory_bytes);
    }
    const auto key = translation::StageCache::key(control, code, text);
    if (key.empty()) return refuse("Shader translation key could not be computed");
    {
        std::unique_lock lock(state.mutex);
        if (const auto found = state.prepared.find(key); found != state.prepared.end()) return found->second;
        if (const auto found = state.failed.find(key); found != state.failed.end()) {
            auto message = found->second;
            lock.unlock();
            return refuse(std::move(message));
        }
        if (const auto found = state.translating.find(key); found != state.translating.end()) {
            pending = found->second; return nullptr;
        }
        if (package_miss && ++state.not_in_package <= 16)
            armigo::diag::write(armigo::diag::Level::info,  "Armigo shader runtime: stage not in the package, prepared at run time: "
                                 "%s program %.16s (%s)\n", stage_name(stage), key.c_str(), package_miss->c_str());
    }
    const auto keep = [&](StagePointer ready) {
        const std::lock_guard lock(state.mutex);
        ++state.loaded;
        report_runtime(state);
        return state.prepared.emplace(key, std::move(ready)).first->second;
    };
    {
        std::unique_lock lock(state.preload_mutex);
        if (const auto found = state.preloaded.find(key); found != state.preloaded.end()) {
            auto ready = std::move(found->second);
            state.preloaded.erase(found);
            lock.unlock();
            return keep(std::move(ready));
        }
    }
    if (state.cache) {
        if (const auto cached = state.cache->load(key)) {
            auto made = make_cached(*cached);
            if (auto* ready = std::get_if<StagePointer>(&made)) return keep(std::move(*ready));
            // A damaged entry is replaced by a fresh translation.
            std::error_code ignored;
            std::filesystem::remove_all(state.cache->directory() / key, ignored);
        }
    }
    auto job = std::make_shared<TranslationJob>();
    job->key = key;
    job->control = std::move(control);
    job->code = std::move(code);
    job->stage = stage;
    job->options = std::move(options);
    {
        const std::lock_guard lock(state.mutex);
        if (!state.first_translation) state.first_translation = std::chrono::steady_clock::now();
        state.translating.emplace(key, job);
    }
    if (state.threads) {
        const auto queued = queue_translation(job, state.threads);
        if (queued == Queued::yes) {
            pending = std::move(job);
            return nullptr;
        }
        if (queued == Queued::stopping) {
            error = "Run-time shader translation stopped at exit";
            job->claimed = true;
            {
                const std::lock_guard lock(job->mutex);
                job->error = error;
                job->finished = true;
            }
            job->done.notify_all();
            return nullptr;
        }
    }
    // On the guest thread: the stage is ready (or has failed) when SetShaders returns.
    auto ready = job->wait();
    if (!ready) error = job->error;
    return ready;
}

bool validate_extended(void*, const NVNshaderDataExt& descriptor, std::string& error) {
    namespace format = shader_input::format;
    const auto address = reinterpret_cast<std::uint64_t>(descriptor.control);
    std::array<std::byte, format::control_prefix_size> prefix{};
    if (!read_memory(address, prefix) || word(prefix, format::control_size_offset) <
            format::extended_metadata_min_control_size) {
        error = "Extended shader control is unreadable or too short"; return false;
    }
    std::array<std::byte, format::extended_metadata_size> original{}, extended{};
    if (address > std::numeric_limits<std::uint64_t>::max() - format::extended_metadata_offset ||
        !read_memory(address + format::extended_metadata_offset, original) ||
        !read_memory(reinterpret_cast<std::uint64_t>(descriptor.extra), extended) || original != extended) {
        error = "Extended shader metadata is unreadable or changes the control contract"; return false;
    }
    return true;
}

bool resolve_package(void*, const DeviceState* device, std::span<const NVNshaderData> descriptors,
    std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 6>& stages,
    std::array<std::shared_ptr<PendingStage>, 6>& pending, std::string& error) {
    std::lock_guard lock(package_mutex);
    if (!shader_package && !runtime_stages().enabled) {
        error = "No offline shader package is configured and run-time preparation is off"; return false;
    }
    for (const auto& descriptor : descriptors) {
        std::array<std::byte, 0x24> prefix{};
        const auto control_address = reinterpret_cast<std::uint64_t>(descriptor.control);
        if (!read_memory(control_address, prefix)) { error = "Unreadable shader control prefix"; return false; }
        const auto control_size = word(prefix, 0x20);
        if (control_size < 0x718 || control_size > (1u << 20)) {
            error = "Shader control extent is outside supported bounds"; return false;
        }
        std::vector<std::byte> control(control_size);
        if (!read_memory(control_address, control) || std::memcmp(prefix.data(), control.data(), prefix.size())) {
            error = "Shader control changed or became unreadable"; return false;
        }
        const auto code_size = word(control, shader_input::format::code_size_field);
        if (!code_size || code_size > (4u << 20)) { error = "Invalid shader code extent"; return false; }
        std::vector<std::byte> code(code_size);
        if (!read_memory(descriptor.data, code)) { error = "Unreadable shader code"; return false; }
        const auto parsed = shader_input::parse_shader({
            {reinterpret_cast<const std::uint8_t*>(control.data()), control.size()},
            {reinterpret_cast<const std::uint8_t*>(code.data()), code.size()}, {}});
        if (const auto* failure = std::get_if<shader_input::ParseError>(&parsed)) {
            error = failure->message(); return false;
        }
        const auto& shader = std::get<shader_input::ParsedShader>(parsed);
        const auto index = static_cast<std::size_t>(shader.stage);
        if (index >= stages.size() || stages[index] || pending[index]) {
            error = "Duplicate or invalid shader stage"; return false;
        }
        const auto depth = device && device->depth_mode == 1
            ? shader::graphics_msl::VertexDepthConvention::ZeroToOne
            : shader::graphics_msl::VertexDepthConvention::NegativeOneToOne;
        // Identification reads each candidate's recorded program length from
        // guest memory at the program's address, as for the command-stream path.
        std::string missing = "The shader package does not hold the stage";
        if (shader_package) {
            if (shader_package->hoists_implicit_samples() != nvn::title_policy().hoist_implicit_samples) {
                missing = "The shader package was translated with a different sample hoisting than the title selects";
            } else {
                auto found = shader_package->lookup_stage_at(shader.stage, descriptor.data + shader.code_offset, depth);
                if (const auto* failure = std::get_if<graphics::CommandResult>(&found)) {
                    missing = std::string(failure->detail);
                } else {
                    stages[index] = std::get<std::shared_ptr<const platform::CompiledGraphicsStage>>(std::move(found));
                    continue;
                }
            }
        }
        if (!runtime_stages().enabled) {
            record_missing_stage(control, code, static_cast<unsigned>(shader.stage));
            error = std::move(missing); return false;
        }
        // A stage that fails to translate is recorded as missing by the translation.
        stages[index] = prepare_runtime_stage(std::move(control), std::move(code), shader.stage, depth, error,
                                              pending[index], shader_package ? &missing : nullptr);
        if (!stages[index] && !pending[index]) return false;
    }
    return true;
}
} // namespace
void set_program_package(std::shared_ptr<platform::MetalShaderPackage> package, const std::string& title_cache) {
    {
        std::lock_guard lock(package_mutex);
        shader_package = std::move(package);
        auto& runtime = runtime_stages();
        {
            const std::lock_guard lock(runtime.mutex);
            runtime.package = shader_package != nullptr;
        }
        std::filesystem::path directory;
        if (!runtime.root.empty())
            directory = runtime.root / (title_cache.empty() ? std::string("untitled")
                                                            : std::filesystem::path(title_cache).filename().string());
        else if (runtime.title_cache && !title_cache.empty())
            directory = std::filesystem::path(title_cache) / "shaders";
        if (!runtime.cache && !directory.empty()) {
            runtime.cache.emplace(directory);
            // Old translator versions and abandoned staging directories are
            // removed in the background so a cache never grows without bound.
            std::thread([cache = *runtime.cache] { cache.prune(std::chrono::minutes(10)); }).detach();
        }
        start_preload(runtime);
    }
    set_deferred_program_resolver(resolve_package, nullptr, validate_extended);
}
} // namespace armigo::nvn
