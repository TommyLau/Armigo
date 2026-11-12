// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "service_registry.h"

#include <armigo/compat/title_settings.h>
#include <armigo/graphics/engines.h>
#include <armigo/horizon/read_only_shared_memory.h>
#include <armigo/platform/metal_engine_renderer.h>
#include <armigo/platform/metal_shader_package.h>
#include <armigo/platform/pcm_sink.h>
#include <armigo/runtime/presentation.h>
#include <armigo/services/account.h>
#include <armigo/services/addon_content.h>
#include <armigo/services/applet.h>
#include <armigo/services/audio_device_manager.h>
#include <armigo/services/audio_renderer.h>
#include <armigo/services/audio_user.h>
#include <armigo/services/hardware_opus.h>
#include <armigo/services/nsd.h>
#include <armigo/services/sound_mixer.h>
#include <armigo/services/ssl.h>
#include <armigo/services/context_writer.h>
#include <armigo/services/file_system.h>
#include <armigo/services/hid.h>
#include <armigo/services/loader.h>
#include <armigo/services/log_manager.h>
#include <armigo/services/nfp.h>
#include <armigo/services/nvdrv.h>
#include <armigo/services/parental_control.h>
#include <armigo/services/settings.h>
#include <armigo/services/shared_font.h>
#include <armigo/services/audio_out.h>
#include <armigo/services/bsd.h>
#include <armigo/services/csrng.h>
#include <armigo/services/mm.h>
#include <armigo/services/nifm.h>
#include <armigo/services/performance.h>
#include <armigo/services/play_report.h>
#include <armigo/services/service_manager_session.h>
#include <armigo/services/shared_clock.h>
#include <armigo/services/time_session.h>
#include <armigo/services/video_interface.h>
#include <armigo/services/vsync_source.h>

#include "../platform/metal/nvn_install.h"

#if __has_include(<armigo/shader/expected_metal_chain.h>)
#include <armigo/shader/expected_metal_chain.h>
#define ARMIGO_HAS_EXPECTED_METAL_CHAIN 1
#endif

#include <armigo/runtime/shader_chain.h>

std::string_view armigo::runtime::expected_shader_chain_id() noexcept {
#if defined(ARMIGO_HAS_EXPECTED_METAL_CHAIN)
    return shader::expected_metal_chain_id;
#else
    return {};
#endif
}

#include <algorithm>
#include <cstdio>
#include <future>
#include <thread>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <cstdlib>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::runtime {

#ifdef ARMIGO_HAS_EXPECTED_METAL_CHAIN
class PendingShaderPackage {
public:
    std::future<platform::MetalShaderPackage::OpenResult> result;
};

namespace {

platform::MetalShaderPackage::OpenResult open_shader_package(const std::filesystem::path& directory) {
    // A package admits the stages the title uses; stages it lacks are
    // reported when a draw needs them.
    platform::MetalShaderPackage::Options options;
    options.expected_chain_id = shader::expected_metal_chain_id;
    options.allow_development_subset = true;
    // Stage reflection, program and library files are checked when a program
    // first selects the stage (nvnProgramSetShaders on the NVN path, a draw on
    // the command-stream path), so startup does not read the whole package.
    options.lazy_entries = true;
    // One preparing thread per core up to the performance cores of the large
    // chips: the first scenes select several hundred stages, and the guest
    // waits for each one that is not yet prepared.
    options.prefetch_threads = std::clamp(std::thread::hardware_concurrency(), 4u, 12u);
    return platform::MetalShaderPackage::open(directory, std::move(options));
}

}  // namespace
#else
class PendingShaderPackage {};
#endif

namespace {

constexpr std::uint32_t handle_exhausted = 0xd201;
constexpr std::uint32_t port_not_found = 0xf201;
constexpr std::uint64_t full_non_debug_fs_permissions = 0x3fff'ffff'ffff'ffffULL;

LaunchFailure make_failure(LaunchStage stage, LaunchError code,
                           std::string operation, std::string detail) {
    LaunchFailure failure;
    failure.stage = stage;
    failure.code = code;
    failure.operation = std::move(operation);
    failure.detail = std::move(detail);
    return failure;
}

LaunchStatus success() { return std::monostate{}; }

services::sm::ServiceName service_name(std::string_view text) {
    services::sm::ServiceName name{};
    const auto length = std::min(text.size(), name.size());
    std::transform(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(length), name.begin(),
                   [](char value) { return static_cast<std::byte>(value); });
    return name;
}

bool service_acl_allows(std::span<const std::byte> acl, std::string_view requested) {
    const auto name = service_name(requested);
    std::size_t offset = 0;
    while (offset < acl.size()) {
        const auto control = std::to_integer<std::uint8_t>(acl[offset]);
        const std::size_t length = (control & 7u) + 1;
        if (acl.size() - offset < length + 1)
            break;
        if ((control & 0x80u) == 0) {
            const auto bytes = acl.subspan(offset + 1, length);
            const bool wildcard = bytes.back() == std::byte{'*'};
            const std::size_t compared = wildcard ? length - 1 : length;
            bool equal = compared <= name.size();
            for (std::size_t index = 0; equal && index < compared; ++index)
                equal = bytes[index] == name[index];
            if (equal && (wildcard || std::all_of(
                    name.begin() + static_cast<std::ptrdiff_t>(compared), name.end(),
                    [](std::byte value) { return value == std::byte{}; })))
                return true;
        }
        offset += length + 1;
    }
    return false;
}

std::uint32_t read_u32(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index)
        value |= std::uint32_t{std::to_integer<std::uint8_t>(bytes[offset + index])}
                 << (8 * index);
    return value;
}

std::uint64_t read_u64(std::span<const std::byte> bytes, std::size_t offset) {
    return read_u32(bytes, offset) |
           (std::uint64_t{read_u32(bytes, offset + 4)} << 32);
}

std::optional<LaunchFailure> validate_file_system_access(const LaunchProfile& profile) {
    if (!service_acl_allows(profile.permissions.service_access_control, "fsp-srv"))
        return std::nullopt;
    const auto& bytes = profile.permissions.file_system_access_control;
    if (bytes.size() != 0x1c || read_u32(bytes, 0) == 0 ||
        read_u32(bytes, 0x0c) != 0x1c || read_u32(bytes, 0x10) != 0 ||
        read_u32(bytes, 0x14) != 0x1c || read_u32(bytes, 0x18) != 0) {
        return make_failure(
            LaunchStage::services, LaunchError::unsupported,
            "service_registry.file_system_access",
            "the file-system access profile is not the normalized owner-free 0x1c-byte shape");
    }
    // The profile derivation has already intersected ACI0 requests with ACID
    // grants and removed debug-only bits. No verified mapping yet relates
    // individual bits to the FS session's command catalogue, which holds only
    // the commands targeted titles use; a partial mask is admitted and noted.
    if (const auto mask = read_u64(bytes, 4); mask != full_non_debug_fs_permissions)
        armigo::diag::write(armigo::diag::Level::error,  "Armigo services: file-system permission mask %llx is not enforced per command.\n",
                     static_cast<unsigned long long>(mask));
    return std::nullopt;
}

std::optional<LaunchFailure> validate_service_access(const LaunchProfile& profile) {
    const auto& acl = profile.permissions.service_access_control;
    if (acl.size() > 0x200) {
        return make_failure(LaunchStage::services, LaunchError::invalid_input,
                            "service_registry.service_access",
                            "the service-access record exceeds the service-manager limit");
    }
    if (profile.origin != ProfileOrigin::synthetic_test)
        return std::nullopt;
    std::size_t offset = 0;
    while (offset < acl.size()) {
        const auto control = std::to_integer<std::uint8_t>(acl[offset]);
        const std::size_t length = (control & 7u) + 1;
        if (acl.size() - offset < length + 1) {
            return make_failure(
                LaunchStage::services, LaunchError::invalid_input,
                "service_registry.service_access",
                "a synthetic service-access entry is truncated");
        }
        offset += length + 1;
    }
    return std::nullopt;
}

struct Counters {
    std::atomic<std::uint64_t> named_ports{0};
    std::atomic<std::uint64_t> services{0};
    std::atomic<std::uint64_t> queued_frames{0};
    std::atomic<std::uint64_t> completed_frames{0};
    std::atomic<std::uint64_t> draws{0};
    std::atomic<std::uint64_t> clears{0};
};

class GuardedRenderer final : public std::enable_shared_from_this<GuardedRenderer> {
public:
    GuardedRenderer(graphics::RenderCallbacks inner,
                    std::function<LaunchResult<std::shared_ptr<void>>()> acquire,
                    std::shared_ptr<Counters> counters)
        : inner_(std::move(inner)), acquire_(std::move(acquire)),
          counters_(std::move(counters)) {}

    void enable_fenced_access() noexcept { fenced_access_ = true; }

    graphics::RenderCallbacks callbacks() {
        auto self = shared_from_this();
        graphics::RenderCallbacks callbacks{
            &GuardedRenderer::clear, this, &GuardedRenderer::draw,
            std::move(self), inner_.memory_changed ? &GuardedRenderer::memory_changed : nullptr,
            inner_.coherence ? &GuardedRenderer::coherence : nullptr};
        callbacks.ordered_effect = inner_.ordered_effect ? &GuardedRenderer::ordered_effect : nullptr;
        callbacks.read_coherence = inner_.read_coherence ? &GuardedRenderer::read_coherence : nullptr;
        callbacks.status = inner_.status ? &GuardedRenderer::status : nullptr;
        callbacks.ordered_write = inner_.ordered_write ? &GuardedRenderer::ordered_write : nullptr;
        callbacks.dispatch = inner_.dispatch ? &GuardedRenderer::dispatch : nullptr;
        callbacks.blit = inner_.blit ? &GuardedRenderer::blit : nullptr;
        callbacks.resolve_guest = inner_.resolve_guest ? &GuardedRenderer::resolve_guest : nullptr;
        return callbacks;
    }

    std::optional<LaunchFailure> failure() const {
        std::lock_guard lock(failure_mutex_);
        return failure_;
    }

private:
    void latch(LaunchFailure failure) noexcept {
        try {
            std::lock_guard lock(failure_mutex_);
            if (!failure_) failure_ = std::move(failure);
        } catch (...) {
        }
    }

    static graphics::CommandResult denied(std::string_view detail) noexcept {
        return {graphics::CommandStatus::Stopped, detail};
    }

    graphics::CommandResult completed(graphics::CommandResult result,
                                       const char* operation,
                                       std::atomic<std::uint64_t>* counter = nullptr) noexcept {
        if (result.ok()) {
            if (counter) counter->fetch_add(1, std::memory_order_relaxed);
        } else {
            try {
                const auto code = result.status == graphics::CommandStatus::Stopped ? LaunchError::host_failure :
                    result.status == graphics::CommandStatus::MemoryFault ? LaunchError::guest_fault :
                    result.status == graphics::CommandStatus::BudgetExceeded ? LaunchError::resource_limit :
                    result.status == graphics::CommandStatus::InvalidEntry ||
                    result.status == graphics::CommandStatus::InvalidHeader ? LaunchError::invalid_input :
                    LaunchError::unsupported;
                auto failure = make_failure(LaunchStage::graphics, code, operation,
                                            std::string(result.detail));
                if (result.method) failure.command = result.method;
                latch(std::move(failure));
            } catch (...) {
                // The failed command remains a failure even if its owned
                // diagnostic cannot be allocated during host exhaustion.
            }
        }
        return result;
    }

    LaunchResult<std::shared_ptr<void>> acquire() noexcept {
        // Pipelined GPU memory follows guest fence ownership. The synchronous
        // gate only serializes guest startup; it cannot park a running guest.
        if (fenced_access_) return std::shared_ptr<void>{};
        if (!acquire_)
            return make_failure(LaunchStage::graphics, LaunchError::unsupported,
                                "service_registry.graphics_access",
                                "no guest-execution exclusion provider is installed");
        try {
            auto result = acquire_();
            if (auto* owner = std::get_if<std::shared_ptr<void>>(&result); owner && !*owner)
                return make_failure(LaunchStage::graphics, LaunchError::host_failure,
                                    "service_registry.graphics_access",
                                    "the exclusion provider returned no owner");
            return result;
        } catch (...) {
            return make_failure(LaunchStage::graphics, LaunchError::host_failure,
                                "service_registry.graphics_access",
                                "the exclusion provider raised an exception");
        }
    }

    static graphics::CommandResult clear(void* context,
        const graphics::RenderStop& stop,
        const graphics::GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        auto owner = std::get<std::shared_ptr<void>>(std::move(access));
        static_cast<void>(owner);
        if (!self.inner_.clear)
            return self.completed(denied("the renderer has no clear callback"),
                                  "render-clear");
        const auto result = self.inner_.clear(self.inner_.context, stop, memory);
        return self.completed(result, "render-clear", &self.counters_->clears);
    }

    static graphics::CommandResult draw(void* context,
        const graphics::RenderStop& stop,
        const graphics::GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        auto owner = std::get<std::shared_ptr<void>>(std::move(access));
        static_cast<void>(owner);
        if (!self.inner_.draw)
            return self.completed(denied("the renderer has no draw callback"),
                                  "render-draw");
        const auto result = self.inner_.draw(self.inner_.context, stop, memory);
        return self.completed(result, "render-draw", &self.counters_->draws);
    }

    static graphics::CommandResult status(void* context) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        return self.completed(self.inner_.status(self.inner_.context), "render-status");
    }

    static graphics::CommandResult dispatch(void* context, const graphics::RenderStop& stop,
        const graphics::GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        return self.completed(self.inner_.dispatch(self.inner_.context, stop, memory), "render-dispatch");
    }

    static graphics::CommandResult blit(void* context, const graphics::RenderStop& stop,
        const graphics::GpuAddressSpace& memory) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        return self.completed(self.inner_.blit(self.inner_.context, stop, memory), "render-blit");
    }

    static graphics::CommandResult resolve_guest(void* context, std::uint64_t address,
        std::uint64_t size) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        return self.completed(self.inner_.resolve_guest(self.inner_.context, address, size), "render-resolve-guest");
    }

    static graphics::CommandResult memory_changed(void* context,
        std::span<const graphics::Mapping> removed,
        std::span<const std::shared_ptr<void>> released,
        graphics::MemoryCallbacks memory) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        auto owner = std::get<std::shared_ptr<void>>(std::move(access));
        static_cast<void>(owner);
        const auto result = self.inner_.memory_changed(
            self.inner_.context, removed, released, memory);
        return self.completed(result, "render-memory-changed");
    }

    static graphics::CommandResult coherence(void* context, const graphics::GpuAddressSpace& memory,
                                             std::uint64_t gpu_address, std::uint64_t size) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        auto owner = std::get<std::shared_ptr<void>>(std::move(access));
        static_cast<void>(owner);
        const auto result = self.inner_.coherence(self.inner_.context, memory, gpu_address, size);
        return self.completed(result, "render-coherence");
    }

    static graphics::CommandResult read_coherence(void* context, const graphics::GpuAddressSpace& memory,
                                                  std::uint64_t gpu_address, std::uint64_t size) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        auto owner = std::get<std::shared_ptr<void>>(std::move(access));
        static_cast<void>(owner);
        const auto result = self.inner_.read_coherence(self.inner_.context, memory, gpu_address, size);
        return self.completed(result, "render-read-coherence");
    }

    static graphics::CommandResult ordered_effect(void* context,
        const graphics::GpuAddressSpace& memory, graphics::GuestEffect effect) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        auto owner = std::get<std::shared_ptr<void>>(std::move(access));
        static_cast<void>(owner);
        return self.completed(self.inner_.ordered_effect(self.inner_.context, memory,
                                                       std::move(effect)), "render-ordered-effect");
    }

    static graphics::CommandResult ordered_write(void* context,
        const graphics::GpuAddressSpace& memory, std::uint64_t gpu_address,
        std::span<const std::byte> payload) noexcept {
        auto& self = *static_cast<GuardedRenderer*>(context);
        auto access = self.acquire();
        if (auto* failure = std::get_if<LaunchFailure>(&access)) {
            self.latch(*failure);
            return denied("renderer guest-access exclusion was not acquired");
        }
        auto owner = std::get<std::shared_ptr<void>>(std::move(access));
        static_cast<void>(owner);
        return self.completed(self.inner_.ordered_write(
            self.inner_.context, memory, gpu_address, payload), "render-ordered-write");
    }

    bool fenced_access_ = false;
    graphics::RenderCallbacks inner_;
    std::function<LaunchResult<std::shared_ptr<void>>()> acquire_;
    std::shared_ptr<Counters> counters_;
    mutable std::mutex failure_mutex_;
    std::optional<LaunchFailure> failure_;
};

#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
// VI invokes its synchronous presenter on the dispatch thread and does not
// permit presenter reentry. The binding may perform work on its own worker, but
// returns the result to this same callback before VI continues release logic.
// A request-local marker therefore correlates consumer completion with exactly
// one QueueBuffer transaction without conflating concurrent VI sessions.
thread_local bool* active_presentation_completion = nullptr;

std::uint32_t message_word(const horizon::IpcMessage& message,
                           std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index)
        value |= std::uint32_t{std::to_integer<std::uint8_t>(message[offset + index])}
                 << (8 * index);
    return value;
}

class CountingViSession final : public horizon::ClientSession {
public:
    CountingViSession(std::shared_ptr<services::vi::DisplaySession> inner,
                      std::shared_ptr<Counters> counters)
        : inner_(std::move(inner)), counters_(std::move(counters)) {}

    void set_memory_mapper(horizon::GuestMemoryMapper mapper) noexcept override {
        inner_->set_memory_mapper(std::move(mapper));
    }

    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override {
        return wrap(inner_->request(message, sender_pid));
    }

    horizon::IpcReply request_with_buffers(
        const horizon::IpcMessage& message, std::uint64_t sender_pid,
        const horizon::GuestBuffers& buffers) override {
        return track_completion(message, buffers, [&](const auto& forwarded) {
            return inner_->request_with_buffers(message, sender_pid, forwarded);
        });
    }

    bool receives_copied_objects() const noexcept override {
        return inner_->receives_copied_objects();
    }

    horizon::IpcReply request_with_handles(
        const horizon::IpcMessage& message, std::uint64_t sender_pid,
        const horizon::GuestBuffers& buffers,
        std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects) override {
        return track_completion(message, buffers, [&](const auto& forwarded) {
            return inner_->request_with_handles(
                message, sender_pid, forwarded, copied_objects);
        });
    }

private:
    class CompletionScope final {
    public:
        explicit CompletionScope(bool& completed) noexcept
            : previous_(active_presentation_completion) {
            active_presentation_completion = &completed;
        }
        ~CompletionScope() { active_presentation_completion = previous_; }

    private:
        bool* previous_ = nullptr;
    };

    class BufferProxy final {
    public:
        explicit BufferProxy(const horizon::GuestBuffers& source)
            : source_(&source), forwarded_(&BufferProxy::read,
                                           &BufferProxy::write, this) {
            for (std::size_t index = 0; index < source.size(); ++index)
                forwarded_.add(source[index]);
        }

        const horizon::GuestBuffers& forwarded() const noexcept {
            return forwarded_;
        }
        bool write_failed() const noexcept { return write_failed_; }

    private:
        static bool read(void* context, std::uint64_t address,
                         std::span<std::byte> output) noexcept {
            const auto& self = *static_cast<BufferProxy*>(context);
            return self.source_->read_memory(address, output);
        }
        static bool write(void* context, std::uint64_t address,
                          std::span<const std::byte> input) noexcept {
            auto& self = *static_cast<BufferProxy*>(context);
            const bool copied = self.source_->write_memory(address, input);
            self.write_failed_ = self.write_failed_ || !copied;
            return copied;
        }

        const horizon::GuestBuffers* source_ = nullptr;
        horizon::GuestBuffers forwarded_;
        bool write_failed_ = false;
    };

    bool is_queue_buffer(const horizon::IpcMessage& message) const noexcept {
        const auto cmif = horizon::locate_cmif(message);
        return cmif && *cmif <= message.size() - 24 &&
               message_word(message, *cmif + 8) <= 3 &&
               message_word(message, *cmif + 20) == 7;
    }

    template<class Request>
    horizon::IpcReply track_completion(const horizon::IpcMessage& message,
                                       const horizon::GuestBuffers& buffers,
                                       Request&& request) {
        if (!is_queue_buffer(message))
            return wrap(std::forward<Request>(request)(buffers));
        bool consumer_completed = false;
        CompletionScope completion_scope(consumer_completed);
        BufferProxy proxy(buffers);
        auto reply = wrap(std::forward<Request>(request)(proxy.forwarded()));
        // Consumer completion alone is earlier than VI release. A handled reply
        // proves release. For QueueBuffer the only output copy occurs after the
        // release, so a request-local failed write proves the documented
        // post-release transport failure without relying on mutable diagnostics.
        if (consumer_completed &&
            (reply.disposition == horizon::IpcDisposition::handled ||
             proxy.write_failed())) {
            counters_->completed_frames.fetch_add(1, std::memory_order_release);
        }
        return reply;
    }

    horizon::IpcReply wrap(horizon::IpcReply reply) {
        for (std::size_t index = 0; index < reply.transferred_count; ++index) {
            auto display = std::dynamic_pointer_cast<services::vi::DisplaySession>(
                reply.transferred[index]);
            if (display) {
                reply.transferred[index] = std::make_shared<CountingViSession>(
                    std::move(display), counters_);
            }
        }
        return reply;
    }

    const std::shared_ptr<services::vi::DisplaySession> inner_;
    const std::shared_ptr<Counters> counters_;
};
#endif

std::shared_ptr<GuardedRenderer> make_guarded_renderer(
    graphics::RenderCallbacks inner,
    std::function<LaunchResult<std::shared_ptr<void>>()> acquire,
    std::shared_ptr<Counters> counters) {
    return std::make_shared<GuardedRenderer>(
        std::move(inner), std::move(acquire), std::move(counters));
}

class Registry final : public ServiceRegistry,
                       public std::enable_shared_from_this<Registry> {
public:
    static LaunchResult<std::shared_ptr<ServiceRegistry>> create(
        ServiceRegistryDependencies dependencies) {
        if (!dependencies.process_id || !dependencies.handles ||
            !dependencies.memory || !dependencies.memory->process() ||
            !dependencies.clock || !dependencies.presentation_clock ||
            dependencies.presentation_clock->clock() != dependencies.clock ||
            !dependencies.presenter) {
            return make_failure(
                LaunchStage::services, LaunchError::invalid_input,
                "create_service_registry",
                "process identity, handles, memory, one shared clock, presentation clock, "
                "and presenter are required");
        }
        if (dependencies.profile.permissions.handle_table_capacity == 0 ||
            dependencies.handles->capacity() !=
                dependencies.profile.permissions.handle_table_capacity) {
            return make_failure(LaunchStage::services, LaunchError::invalid_input,
                                "create_service_registry",
                                "the handle table does not match the admitted process profile");
        }
        if (auto failure = validate_service_access(dependencies.profile))
            return std::move(*failure);
        if (auto failure = validate_file_system_access(dependencies.profile))
            return std::move(*failure);

        try {
            auto registry = std::shared_ptr<Registry>(
                new Registry(std::move(dependencies)));
            if (auto failure = registry->initialize())
                return std::move(*failure);
            return std::shared_ptr<ServiceRegistry>(std::move(registry));
        } catch (const std::bad_alloc&) {
            return make_failure(LaunchStage::services, LaunchError::resource_limit,
                                "create_service_registry",
                                "service registry allocation failed");
        } catch (const std::exception& error) {
            return make_failure(LaunchStage::services, LaunchError::host_failure,
                                "create_service_registry", error.what());
        } catch (...) {
            return make_failure(LaunchStage::services, LaunchError::host_failure,
                                "create_service_registry",
                                "service registry construction raised an exception");
        }
    }

    horizon::SessionConnection connect_named_port(std::string_view port) override {
        if (profile_.process.platform == GuestPlatform::switch2 && port != "sm:") {
            if (stopped_.load(std::memory_order_acquire))
                return {horizon::IpcDisposition::cancelled, 0, {}};
            latch(make_failure(LaunchStage::services, LaunchError::unsupported,
                               "service_registry.switch2_named_port",
                               "Switch 2 named port " + std::string(port) +
                               " requires a verified system-code contract."));
            return {horizon::IpcDisposition::unsupported, 0, {}};
        }
        if (port != "sm:")
            return {horizon::IpcDisposition::handled, port_not_found, {}};
        if (stopped_.load(std::memory_order_acquire))
            return {horizon::IpcDisposition::cancelled, 0, {}};
        try {
            const auto weak = weak_from_this();
            // The selected 23.0.1 reference and Switch 2 client use this same
            // ordinary-process TIPC subset. Service admission remains separate.
            auto session = std::make_shared<services::sm::ServiceManagerSession>(
                services::sm::ProcessConfig{process_id_, true,
                    profile_.permissions.service_access_control},
                [weak](std::uint64_t process_id, const services::sm::ServiceName& name) {
                    const auto registry = weak.lock();
                    if (!registry)
                        return services::sm::LookupResult{
                            services::sm::LookupKind::unsupported, 0, 0};
                    return registry->connect_service(process_id, name);
                });
            counters_->named_ports.fetch_add(1, std::memory_order_relaxed);
            return {horizon::IpcDisposition::handled, 0, std::move(session)};
        } catch (const std::bad_alloc&) {
            return {horizon::IpcDisposition::handled, handle_exhausted, {}};
        } catch (...) {
            latch(make_failure(LaunchStage::services, LaunchError::host_failure,
                               "service_registry.named_port",
                               "service-manager session construction raised an exception"));
            return {horizon::IpcDisposition::unsupported, 0, {}};
        }
    }

    std::optional<LaunchFailure> failure() const override {
        std::lock_guard lock(failure_mutex_);
        if (first_failure_)
            return first_failure_;
        if (nvn_driver_) {
            if (auto detail = platform::metal::nvn_driver::failure_detail(); !detail.empty())
                return make_failure(LaunchStage::graphics, LaunchError::host_failure,
                    "service_registry.nvn_replay", std::move(detail));
        }
        if (driver_) {
            if (auto error = driver_->gpu_failure())
                return make_failure(LaunchStage::graphics, LaunchError::host_failure,
                    "service_registry.gpu_worker", *error);
        }
        if (render_guard_) {
            if (auto failure = render_guard_->failure())
                return failure;
        }
#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
        if (auto failure = presentation_clock_->observation_failure()) return failure;
        if (presentation_) {
            if (auto failure = presentation_->failure())
                return failure;
        }
        if (vsync_) {
            if (auto failure = vsync_->failure())
                return failure;
        }
#endif
        return std::nullopt;
    }

    void request_stop() noexcept override {
        stopped_.store(true, std::memory_order_release);
        // Sound mixer streams hold views of guest memory: stop pumping them
        // and release the views before the process's memory goes away.
        std::shared_ptr<services::audsmx::Mixer> mixer;
        {
            const std::lock_guard lock(sound_mixer_mutex_);
            mixer = sound_mixer_;
        }
        if (mixer) mixer->stop();
        if (driver_) driver_->stop_gpu();
#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
        if (presentation_) presentation_->request_stop();
#endif
    }

    LaunchStatus drain(std::optional<ClockInstant> deadline) override {
        request_stop();
        // Execution errors remain visible through failure(). They must not
        // prevent presentation retirement and process-owner teardown.
        if (driver_) driver_->drain_gpu();
#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
        if (presentation_) {
            auto status = presentation_->drain(deadline);
            if (std::holds_alternative<LaunchFailure>(status)) return status;
        }
#else
        auto status = presenter_->drain(deadline);
        if (auto* failure = std::get_if<platform::PresentationFailure>(&status)) {
            auto converted = make_failure(LaunchStage::shutdown,
                failure->code == platform::PresentationError::deadline_expired
                    ? LaunchError::deadline_expired : LaunchError::host_failure,
                "service_registry.presenter_drain", failure->detail);
            return converted;
        }
#endif
        return success();
    }

    LaunchStatus finish_guest_shutdown(std::optional<ClockInstant> deadline) override {
        if (nvn_driver_ && !platform::metal::nvn_driver::stop_submission_workers(deadline))
            return make_failure(LaunchStage::shutdown, LaunchError::deadline_expired,
                                "service_registry.nvn_shutdown", "NVN submission workers did not stop before the deadline.");
        return success();
    }

    ServiceRegistryStatistics statistics() const noexcept override {
        return {
            counters_->named_ports.load(std::memory_order_relaxed),
            counters_->services.load(std::memory_order_relaxed),
            counters_->queued_frames.load(std::memory_order_relaxed),
            counters_->completed_frames.load(std::memory_order_acquire),
            counters_->draws.load(std::memory_order_relaxed),
            counters_->clears.load(std::memory_order_relaxed),
        };
    }

private:
    explicit Registry(ServiceRegistryDependencies dependencies)
        : profile_(std::move(dependencies.profile)),
          process_id_(dependencies.process_id), operation_mode_(dependencies.operation_mode),
          handles_(dependencies.handles),
          memory_(std::move(dependencies.memory)),
          clock_(std::move(dependencies.clock)),
          presentation_clock_(std::move(dependencies.presentation_clock)),
          presenter_(std::move(dependencies.presenter)),
          romfs_(std::move(dependencies.romfs)),
          addon_content_(std::move(dependencies.addon_content)),
          display_version_(std::move(dependencies.display_version)),
          shader_directory_(std::move(dependencies.shader_package_directory)),
          pending_shader_package_(std::move(dependencies.pending_shader_package)),
          nvn_driver_(dependencies.nvn_driver),
          nvn_pso_manifest_(std::move(dependencies.nvn_pso_manifest)),
          startup_user_(dependencies.startup_user),
          observe_frame_(std::move(dependencies.observe_frame)),
          wants_readback_(std::move(dependencies.wants_readback)),
          observe_display_(std::move(dependencies.observe_display)),
          acquire_graphics_access_(std::move(dependencies.acquire_graphics_access)),
          counters_(std::make_shared<Counters>()) {}

    // The title's cache directory (compat::title_cache_directory): the NVN
    // driver keeps its run-time shader stages and pipeline state there.
    std::string title_cache_directory() const {
        return compat::title_cache_directory(profile_.title_id).string();
    }

    std::optional<LaunchFailure> initialize() {
        const auto sample = clock_->sample();
        if (const auto* failure = std::get_if<LaunchFailure>(&sample))
            return *failure;
        services::shared_clock::Configuration clock_configuration;
        const auto counter = std::get<ClockSample>(sample).counter_time.count();
        clock_configuration.counter_nanoseconds = counter;
        clock_configuration.fallback_wall_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        for (std::size_t index = 0; index < 8; ++index)
            clock_configuration.steady_source_id[index] =
                static_cast<std::byte>((process_id_ >> (8 * index)) & 0xff);
        constexpr std::array<std::byte, 8> source_tag{
            std::byte{'A'}, std::byte{'R'}, std::byte{'M'}, std::byte{'I'},
            std::byte{'G'}, std::byte{'O'}, std::byte{'2'}, std::byte{'0'}};
        std::copy(source_tag.begin(), source_tag.end(),
                  clock_configuration.steady_source_id.begin() + 8);
        alignas(8) std::array<std::byte, services::shared_clock::page_size> page{};
        if (services::shared_clock::initialize(page, clock_configuration) !=
            services::shared_clock::InitializeResult::initialized) {
            return make_failure(LaunchStage::services, LaunchError::host_failure,
                                "service_registry.shared_clock",
                                "the process shared clock page could not be initialized");
        }
        shared_clock_ = std::make_shared<horizon::ReadOnlySharedMemory>(page);

        std::filesystem::path font_directory;
        bool fonts_required = false;
        if (const char* override = std::getenv("ARMIGO_SHARED_FONTS"); override && *override) {
            font_directory = override;
            fonts_required = true;
        } else if (!compat::packed_resources().empty()) {
            font_directory = compat::packed_resources() / "system";
        }
        if (fonts_required || std::filesystem::is_directory(font_directory)) {
            auto fonts = services::pl::load_shared_fonts(font_directory);
            if (const auto* error = std::get_if<std::string>(&fonts)) {
                return make_failure(LaunchStage::services, LaunchError::artifact_mismatch,
                                    "service_registry.shared_fonts", *error);
            }
            shared_fonts_ = std::get<services::pl::SharedFonts>(std::move(fonts));
        }
        // Development runs choose the system language and region the way a
        // packed player's settings do.
        if (const char* language = std::getenv("ARMIGO_SYSTEM_LANGUAGE"); language && *language &&
            !services::set::configure_console_language(language)) {
            return make_failure(LaunchStage::services, LaunchError::invalid_input, "service_registry.language",
                                std::string("ARMIGO_SYSTEM_LANGUAGE is not a system language code: ") + language);
        }
        if (const char* region = std::getenv("ARMIGO_SYSTEM_REGION"); region && *region) {
            const std::string_view text(region);
            if (text.size() != 1 || text.front() < '0' || text.front() > '5' ||
                !services::set::configure_console_region(static_cast<std::uint32_t>(text.front() - '0'))) {
                return make_failure(LaunchStage::services, LaunchError::invalid_input, "service_registry.region",
                                    std::string("ARMIGO_SYSTEM_REGION is not a region from 0 to 5: ") + region);
            }
        }

        platform::MetalEngineOptions options;
        // The persistent renderer runs guest command streams alongside running
        // guest threads; ARMIGO_METAL_RENDERER=legacy selects the old one.
        if (!std::getenv("ARMIGO_METAL_RENDERER")) options.mode = platform::MetalEngineMode::Persistent;
        // Guest-visible effects pass through the engine front end, which calls
        // coherence() first, so GPU results need not be written back per draw.
        options.defer_gpu_writeback = true;
        // The public runner enables page-level CPU write tracking so native
        // guest stores are observed instead of forcing a boundary epoch
        // revalidation. ARMIGO_CPU_WRITE_TRACKING=0 keeps the slow path.
        const auto* tracking = std::getenv("ARMIGO_CPU_WRITE_TRACKING");
        options.cpu_write_tracking = !(tracking && std::string_view{tracking} == "0");
        // Native execution maps guest memory into the host address space.
        options.guest_memory_host_addressable = true;
        if (!shader_directory_.empty()) {
#ifdef ARMIGO_HAS_EXPECTED_METAL_CHAIN
            auto opened = pending_shader_package_ ? pending_shader_package_->result.get()
                                                  : open_shader_package(shader_directory_);
            if (auto* error = std::get_if<platform::MetalShaderPackageError>(&opened)) {
                return make_failure(LaunchStage::graphics, LaunchError::artifact_mismatch,
                                    "service_registry.shader_package", error->detail);
            }
            shader_package_ = std::get<std::shared_ptr<platform::MetalShaderPackage>>(
                std::move(opened));
            if (nvn_driver_) {
                if (auto error = platform::metal::nvn_driver::install_driver(shader_package_, nvn_pso_manifest_, title_cache_directory()); !error.empty())
                    return make_failure(LaunchStage::graphics, LaunchError::host_failure,
                                        "service_registry.nvn_driver", std::move(error));
            }
            options.lookup = &platform::MetalShaderPackage::callback;
            options.compute_lookup = &platform::MetalShaderPackage::compute_callback;
            options.lookup_context = shader_package_.get();
            options.lookup_owner = shader_package_;
            // Package stages never change once loaded.
            options.immutable_stages = true;
#else
            return make_failure(
                LaunchStage::graphics, LaunchError::unsupported,
                "service_registry.shader_package",
                "the installed build has no trusted Metal compiler-chain identity");
#endif
        } else if (nvn_driver_) {
            // Without a package the NVN driver prepares every stage at run time.
            if (auto error = platform::metal::nvn_driver::install_driver(nullptr, nvn_pso_manifest_, title_cache_directory()); !error.empty())
                return make_failure(LaunchStage::graphics, LaunchError::host_failure,
                                    "service_registry.nvn_driver", std::move(error));
        }
        auto backend = platform::create_metal_engine_backend(std::move(options));
        if (auto* error = std::get_if<graphics::RendererError>(&backend)) {
            const auto command = error->command();
            return make_failure(LaunchStage::graphics, LaunchError::host_failure,
                                "service_registry.metal_backend",
                                std::string(command.detail));
        }
        auto guard = make_guarded_renderer(
            std::get<graphics::RenderCallbacks>(std::move(backend)),
            acquire_graphics_access_, counters_);
        render_guard_ = guard;
        driver_ = std::make_shared<services::nvdrv::DriverSession>();
        const auto rendering = guard->callbacks();
        driver_->set_render_backend(rendering);
        if (rendering.ordered_effect) {
            // These paired capabilities share the native memory state, but the
            // mapper lifetime does not retain the handle table containing nvdrv.
            const auto gpu_memory_owner = memory_->service_mapper().lifetime;
            if (!gpu_memory_owner)
                return make_failure(LaunchStage::graphics, LaunchError::host_failure,
                    "service_registry.gpu_memory", "GPU worker memory lifetime is unavailable");
            guard->enable_fenced_access();
            driver_->enable_async_gpu(memory_->ipc_buffers(), gpu_memory_owner);
        }

#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
        PresentationDependencies presentation_dependencies;
        graphics_access_ = driver_->graphics_backing_access(memory_);
        presentation_dependencies.backing = graphics_access_;
        presentation_dependencies.clock = presentation_clock_;
        presentation_dependencies.presenter = presenter_;
        presentation_dependencies.observe_frame = observe_frame_;
        presentation_dependencies.wants_readback = wants_readback_;
        auto binding = bind_presentation(std::move(presentation_dependencies));
        if (auto* failure = std::get_if<LaunchFailure>(&binding))
            return *failure;
        presentation_ = std::get<std::shared_ptr<PresentationBinding>>(std::move(binding));
        vsync_ = std::make_shared<services::vi::VsyncSource>(presentation_clock_);
#endif
        return std::nullopt;
    }

    // Renderer sound plays on the default output device in windowed runs and is
    // muted in headless runs, which never need an audio device. ARMIGO_AUDIO=0
    // mutes and ARMIGO_AUDIO=1 plays regardless of the presentation mode.
    // ARMIGO_AUDIO=silent opens that same device and clock and zero-fills the
    // samples after they are consumed. A device that cannot be opened leaves
    // that renderer muted with a diagnostic.
    // The process's sound mixer, created with its first session. Streams pin
    // the title's ring buffers through the process memory, which the mixer
    // does not keep alive. A mixer reached after request_stop is stopped, so
    // no stream opens once the stop has released the others.
    std::shared_ptr<services::audsmx::Mixer> sound_mixer() {
        const std::lock_guard lock(sound_mixer_mutex_);
        if (!sound_mixer_) sound_mixer_ = make_sound_mixer();
        if (stopped_.load(std::memory_order_acquire)) sound_mixer_->stop();
        return sound_mixer_;
    }

    std::shared_ptr<services::audsmx::Mixer> make_sound_mixer() {
        std::weak_ptr<GuestMemory> weak = memory_;
        auto pinner = [weak](std::uint64_t address, std::uint64_t size)
            -> std::optional<services::audsmx::StreamMemory> {
            const auto memory = weak.lock();
            if (!memory) return std::nullopt;
            auto pinned = memory->pin({address, size}, memory::Permission::read | memory::Permission::write);
            auto* pin = std::get_if<GuestMemoryPin>(&pinned);
            if (!pin || !*pin || (*pin)->writable_bytes().size() != size) return std::nullopt;
            return services::audsmx::StreamMemory{(*pin)->writable_bytes(), *pin};
        };
        return std::make_shared<services::audsmx::Mixer>(std::move(pinner), audio_output_factory());
    }

    services::audren::OutputFactory audio_output_factory() const {
        const bool windowed = presenter_ && presenter_->mode() == platform::PresentationMode::windowed;
        const auto kind = platform::pcm_output_kind_from_setting(std::getenv("ARMIGO_AUDIO"), windowed);
        if (kind == platform::PcmOutputKind::muted) return {};
        return [kind] {
            auto output = platform::make_pcm_sink_output(kind);
            if (!output)
                armigo::diag::write(armigo::diag::Level::error,  "[audio output] the default output device is unavailable; "
                                     "guest audio output is muted\n");
            return output;
        };
    }

    services::sm::LookupResult connect_service(
        std::uint64_t process_id, const services::sm::ServiceName& name) {
        if (process_id != process_id_) {
            latch(make_failure(LaunchStage::services, LaunchError::permission_denied,
                               "service_registry.process_identity",
                               "the service lookup belongs to a different process"));
            return {services::sm::LookupKind::unsupported, 0, 0};
        }
        if (stopped_.load(std::memory_order_acquire))
            return {services::sm::LookupKind::cancelled, 0, 0};
        auto handles = handles_.lock();
        if (!handles) {
            latch(make_failure(LaunchStage::services, LaunchError::host_failure,
                               "service_registry.handles",
                               "the process handle table is no longer available"));
            return {services::sm::LookupKind::unsupported, 0, 0};
        }

        std::shared_ptr<horizon::ClientSession> session;
        try {
            // Services admitted for Switch 2 so far. Each serves the title with
            // its existing session; contract differences are handled inside it.
            // Graphics services (nvdrv, vi) and ro, whose Switch 2 NRR layout
            // differs, stay outside until their contracts are admitted.
            static const std::array switch2_services{
                service_name("lm"), service_name("appletOE"), service_name("aoc:u"), service_name("ectx:aw"),
                service_name("pctl"), service_name("prepo:u"), service_name("time:u"), service_name("acc:u0"),
                service_name("fsp-srv"), service_name("set"), service_name("pl:u"), service_name("pl:s"),
                service_name("apm"), service_name("hid"), service_name("nifm:u"), service_name("mm:u"),
                service_name("bsd:u"), service_name("audout:u"), service_name("audren:u"), service_name("aud:u"),
                service_name("hwopus"), service_name("nfp:user"), service_name("auddmg:u"),
                service_name("audsmx:u"), service_name("ssl"), service_name("nsd:u"),
            };
            if (profile_.process.platform == GuestPlatform::switch2 &&
                std::find(switch2_services.begin(), switch2_services.end(), name) == switch2_services.end()) {
                std::string printable;
                for (const auto byte : name) {
                    const auto c = static_cast<char>(byte);
                    if (c == '\0') break;
                    printable += c >= 0x20 && c < 0x7f ? c : '?';
                }
                latch(make_failure(LaunchStage::services, LaunchError::unsupported,
                    "service_registry.switch2_service",
                    "Switch 2 service \"" + printable + "\" requires a verified service contract."));
                return {services::sm::LookupKind::unsupported, 0, 0};
            }
            if (name == service_name("lm"))
                session = std::make_shared<services::lm::LogManagerSession>(process_id_);
            else if (name == service_name("acc:u0"))
                session = std::make_shared<services::acc::AccountSession>(process_id_);
            else if (name == service_name("aoc:u"))
                session = std::make_shared<services::aoc::AddOnContentSession>(process_id_, *handles,
                                                                                addon_content_);
            else if (name == service_name("ectx:aw"))
                session = std::make_shared<services::err::ContextWriterSession>(process_id_);
            else if (name == service_name("pctl"))
                session = std::make_shared<services::pctl::PctlSession>(process_id_);
            else if (name == service_name("bsd:u"))
                session = std::make_shared<services::bsd::BsdSession>(process_id_, *handles);
            else if (name == service_name("nifm:u"))
                session = std::make_shared<services::nifm::Session>(
                    process_id_, profile_.process.platform == GuestPlatform::switch2);
            else if (name == service_name("mm:u"))
                session = std::make_shared<services::mm::Session>(process_id_);
            else if (name == service_name("csrng") && profile_.process.platform != GuestPlatform::switch2)
                session = std::make_shared<services::csrng::Session>(process_id_);
            else if (name == service_name("aud:u") && profile_.process.platform == GuestPlatform::switch2)
                session = std::make_shared<services::audio_user::UserClientSession>(process_id_,
                                                                                    audio_output_factory());
            else if (name == service_name("hwopus"))
                session = std::make_shared<services::hwopus::ManagerSession>(
                    process_id_, profile_.process.platform == GuestPlatform::switch2 ? services::hwopus::Console::switch2
                                                                                      : services::hwopus::Console::switch1);
            else if (name == service_name("auddmg:u") && profile_.process.platform == GuestPlatform::switch2)
                session = std::make_shared<services::auddmg::DeviceManagerSession>(process_id_);
            else if (name == service_name("nsd:u") && profile_.process.platform == GuestPlatform::switch2)
                session = std::make_shared<services::nsd::NsdSession>(process_id_);
            else if (name == service_name("ssl"))
                session = std::make_shared<services::ssl::SslSession>(process_id_);
            else if (name == service_name("audsmx:u") && profile_.process.platform == GuestPlatform::switch2)
                session = std::make_shared<services::audsmx::SoundMixerSession>(process_id_, sound_mixer());
            else if (name == service_name("audout:u"))
                session = std::make_shared<services::audout::ManagerSession>(process_id_, audio_output_factory());
            else if (name == service_name("set"))
                session = std::make_shared<services::set::SettingsSession>(process_id_);
            else if ((name == service_name("pl:u") || name == service_name("pl:s")) && shared_fonts_)
                session = std::make_shared<services::pl::SharedFontSession>(process_id_, *shared_fonts_);
            else if (name == service_name("pl:u") || name == service_name("pl:s")) {
                latch(make_failure(LaunchStage::services, LaunchError::unsupported,
                                   "service_registry.shared_fonts",
                                   "the title requires shared fonts but no packed font resources are available"));
                return {services::sm::LookupKind::unsupported, 0, 0};
            }
            else if (name == service_name("prepo:u"))
                session = std::make_shared<services::prepo::PlayReportSession>(process_id_);
            else if (name == service_name("time:u"))
                session = std::make_shared<services::time::TimeServiceSession>(
                    process_id_, *handles, shared_clock_, [clock = clock_]() -> std::optional<std::int64_t> {
                        const auto sample = clock->sample();
                        if (const auto* value = std::get_if<ClockSample>(&sample))
                            return value->counter_time.count();
                        return std::nullopt;
                    });
            else if (name == service_name("fsp-srv")) {
                auto file_system = std::make_shared<services::fs::FileSystemServiceSession>(
                    process_id_, 0x1000, romfs_, addon_content_);
                file_system->set_application_program_id(profile_.title_id);
                session = std::move(file_system);
            }
            else if (name == service_name("ro"))
                session = std::make_shared<services::ro::LoaderSession>(process_id_);
            else if (name == service_name("nvdrv"))
                session = std::make_shared<services::nvdrv::DriverSession>(*driver_);
            else if (name == service_name("nfp:user"))
                session = std::make_shared<services::nfp::ManagerSession>(process_id_);
            else if (name == service_name("audren:u"))
                session = std::make_shared<services::audren::ManagerSession>(
                    process_id_, audio_output_factory());
            else if (name == service_name("apm"))
                session = std::make_shared<services::apm::PerformanceSession>(
                    process_id_, 0, services::apm::PerformanceSession::Role::root,
                    operation_mode_);
            else if (name == service_name("appletOE")) {
                auto applet = std::make_shared<services::am::AppletServiceSession>(
                    process_id_, memory_->process(), *handles,
                    services::am::GameplayRecordingPolicy::Unavailable,
                    std::shared_ptr<services::am::ApplicationRunningState>{},
                    operation_mode_, profile_.process.platform == GuestPlatform::switch2
                        ? services::am::AppletContract::switch2
                        : services::am::AppletContract::switch1);
                if (profile_.process.platform == GuestPlatform::switch2) {
                    // The graphics system session's driver session shares the
                    // process's nvdrv client state. A weak reference keeps the
                    // handle table from owning the base driver.
                    services::gsv::GraphicsSystemDependencies graphics;
                    graphics.open_driver = [driver = std::weak_ptr<services::nvdrv::DriverSession>(driver_)]()
                        -> std::shared_ptr<horizon::ClientSession> {
                        const auto base = driver.lock();
                        if (!base) return nullptr;
                        auto session = std::make_shared<services::nvdrv::DriverSession>(*base);
                        session->refuse_gpu_devices("Switch 2 NVN boundary: the title's own NVN driver opens a GPU "
                                                    "device; Armigo serves graphics through its NVN 231.40 driver, "
                                                    "which the title's settings must select");
                        return session;
                    };
                    graphics.wait_refresh = [vsync = std::weak_ptr<services::vi::VsyncSource>(vsync_)](
                        std::uint64_t sequence, std::stop_token stop) -> std::optional<std::uint64_t> {
                        const auto source = vsync.lock();
                        if (!source) return std::nullopt;
                        const auto target = std::max(sequence, source->sequence()) + 1;
                        if (!source->wait_sequence(target, stop)) return std::nullopt;
                        return target;
                    };
#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
                    // Switch 2 has no vi:u session for the NVN driver's native
                    // windows to bind to; they present through a display of
                    // their own, paced by the same refresh.
                    if (nvn_driver_) graphics.native_display = make_display(*handles);
#endif
                    // Through the IPC buffer path, which keeps the write tracker
                    // informed of decoded pictures written into GPU memory.
                    graphics.write_memory = [memory = std::weak_ptr<GuestMemory>(memory_)](
                        std::uint64_t address, std::span<const std::byte> bytes) {
                        const auto guest = memory.lock();
                        return guest && !bytes.empty() && guest->ipc_buffers().write_memory(address, bytes);
                    };
                    graphics.read_memory = [memory = std::weak_ptr<GuestMemory>(memory_)](
                        std::uint64_t address, std::span<std::byte> bytes) {
                        const auto guest = memory.lock();
                        return guest && !bytes.empty() && guest->ipc_buffers().read_memory(address, bytes);
                    };
                    applet->set_graphics_system_dependencies(std::move(graphics));
                }
                if (startup_user_) applet->preselect_user(services::acc::AccountSession::local_user_id());
                if (display_version_) applet->set_display_version(*display_version_);
                applet->grant_foreground();
                session = std::move(applet);
            } else if (name == service_name("hid"))
                session = std::make_shared<services::hid::HidSession>(
                    process_id_, services::hid::SevenSixAxisSamplingState{},
                    profile_.process.platform == GuestPlatform::switch2);
            else if (name == service_name("vi:u") || name == service_name("vi:m") ||
                     name == service_name("vi:s")) {
#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
                session = std::make_shared<CountingViSession>(
                    make_display(*handles), counters_);
#else
                latch(make_failure(
                    LaunchStage::presentation, LaunchError::unsupported,
                    "service_registry.video_interface",
                    "runtime presentation support is not linked into this build"));
                return {services::sm::LookupKind::unsupported, 0, 0};
#endif
            } else {
                std::string printable;
                for (const auto byte : name) {
                    const auto c = static_cast<char>(byte);
                    if (c == '\0') break;
                    printable += c >= 0x20 && c < 0x7f ? c : '?';
                }
                latch(make_failure(LaunchStage::services, LaunchError::unsupported,
                                   "service_registry.connect",
                                   "the authorized service \"" + printable +
                                       "\" has no runtime implementation"));
                return {services::sm::LookupKind::unsupported, 0, 0};
            }
        } catch (const std::bad_alloc&) {
            return {services::sm::LookupKind::failed, 0, handle_exhausted};
        } catch (const std::exception& error) {
            latch(make_failure(LaunchStage::services, LaunchError::host_failure,
                               "service_registry.connect", error.what()));
            return {services::sm::LookupKind::unsupported, 0, 0};
        } catch (...) {
            latch(make_failure(LaunchStage::services, LaunchError::host_failure,
                               "service_registry.connect",
                               "service construction raised an exception"));
            return {services::sm::LookupKind::unsupported, 0, 0};
        }

        const auto handle = handles->insert(std::move(session));
        if (!handle)
            return {services::sm::LookupKind::failed, 0, handle_exhausted};
        counters_->services.fetch_add(1, std::memory_order_relaxed);
        return {services::sm::LookupKind::connected, *handle, 0};
    }

#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
    // A display whose frames reach the host presenter; native windows of the
    // NVN driver bind to the latest one.
    std::shared_ptr<services::vi::DisplaySession> make_display(horizon::HandleTable& handles) {
        auto display = std::make_shared<services::vi::DisplaySession>(handles, operation_mode_);
        display->set_vsync_source(vsync_);
        display->set_buffer_backing_lookup(graphics_access_.lookup);
        if (nvn_driver_) platform::metal::nvn_driver::bind_native_windows(display);
        if (observe_display_) observe_display_(display);
        auto callback = attach_presentation_observer(
            presentation_->callback(), presentation_clock_, memory_);
        auto counters = counters_;
        display->set_presenter(
            [callback = std::move(callback), counters](
                const services::GraphicsPresentation& frame) mutable {
                counters->queued_frames.fetch_add(1, std::memory_order_relaxed);
                auto result = callback(frame);
                if (result.completed) {
                    // A QueueBuffer request counts the frame once its reply
                    // proves release. A native-window present has no request:
                    // the presentation thread runs this callback and the frame
                    // is consumed when it returns.
                    if (active_presentation_completion)
                        *active_presentation_completion = true;
                    else if (frame.native_vsync_paced)
                        counters->completed_frames.fetch_add(
                            1, std::memory_order_release);
                }
                return result;
            });
        return display;
    }
#endif

    void latch(LaunchFailure failure) const {
        std::lock_guard lock(failure_mutex_);
        if (!first_failure_) first_failure_ = std::move(failure);
    }

    LaunchProfile profile_;
    std::uint64_t process_id_ = 0;
    services::OperationMode operation_mode_ = services::OperationMode::Console;
    std::weak_ptr<horizon::HandleTable> handles_;
    std::shared_ptr<GuestMemory> memory_;
    std::mutex sound_mixer_mutex_;
    std::shared_ptr<services::audsmx::Mixer> sound_mixer_;
    std::shared_ptr<GuestClock> clock_;
    std::shared_ptr<PresentationClock> presentation_clock_;
    std::shared_ptr<platform::Presenter> presenter_;
    std::shared_ptr<romfs::Storage> romfs_;
    std::shared_ptr<const services::AddOnContentCatalog> addon_content_;
    std::optional<std::string> display_version_;
    std::filesystem::path shader_directory_;
    std::shared_ptr<PendingShaderPackage> pending_shader_package_;
    bool nvn_driver_ = false;
    std::string nvn_pso_manifest_;
    bool startup_user_ = false;
    std::function<LaunchStatus(const platform::PresentedFrame&)> observe_frame_;
    std::function<bool(std::uint64_t)> wants_readback_;
    std::function<void(const std::shared_ptr<services::vi::DisplaySession>&)> observe_display_;
    std::function<LaunchResult<std::shared_ptr<void>>()> acquire_graphics_access_;
    std::shared_ptr<Counters> counters_;
    std::shared_ptr<horizon::ReadOnlySharedMemory> shared_clock_;
    std::optional<services::pl::SharedFonts> shared_fonts_;
    std::shared_ptr<platform::MetalShaderPackage> shader_package_;
    std::shared_ptr<GuardedRenderer> render_guard_;
    std::shared_ptr<services::nvdrv::DriverSession> driver_;
#ifdef ARMIGO_HAS_RUNTIME_PRESENTATION
    GraphicsBackingAccess graphics_access_;
    std::shared_ptr<PresentationBinding> presentation_;
    std::shared_ptr<services::vi::VsyncSource> vsync_;
#endif
    std::atomic<bool> stopped_{false};
    mutable std::mutex failure_mutex_;
    mutable std::optional<LaunchFailure> first_failure_;
};

} // namespace

services::GraphicsPresenter attach_presentation_observer(
    services::GraphicsPresenter callback,
    const std::shared_ptr<PresentationClock>& clock,
    const std::shared_ptr<GuestMemory>& memory) {
    if (!clock->observes_presents()) return callback;
    return [callback = std::move(callback), clock, memory = std::weak_ptr{memory}](
               const services::GraphicsPresentation& frame) {
        auto result = callback(frame);
        if (result.completed) {
            const auto observed = clock->observe_present(frame.frame_number, frame.swap_interval,
                [memory](std::uint64_t address, std::span<std::byte> out) {
                    const auto owner = memory.lock();
                    return owner && owner->copy_observation(address, out);
                });
            if (const auto* error = std::get_if<LaunchFailure>(&observed))
                return services::GraphicsPresentationResult{false, error->detail};
        }
        return result;
    };
}

horizon::SessionConnection ServiceRegistry::connector(
    void* context, std::string_view port) {
    if (!context)
        return {horizon::IpcDisposition::unsupported, 0, {}};
    return static_cast<ServiceRegistry*>(context)->connect_named_port(port);
}

std::shared_ptr<PendingShaderPackage> start_shader_package_open(
    const std::filesystem::path& directory) {
#ifdef ARMIGO_HAS_EXPECTED_METAL_CHAIN
    if (directory.empty()) return nullptr;
    auto pending = std::make_shared<PendingShaderPackage>();
    pending->result = std::async(std::launch::async, [directory] { return open_shader_package(directory); });
    return pending;
#else
    static_cast<void>(directory);
    return nullptr;
#endif
}

LaunchResult<std::shared_ptr<ServiceRegistry>> create_service_registry(
    ServiceRegistryDependencies dependencies) {
    return Registry::create(std::move(dependencies));
}

} // namespace armigo::runtime
