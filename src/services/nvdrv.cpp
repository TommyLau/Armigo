// ============================================================================
// Armigo Services - Initial Nvidia Driver Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/common/diagnostics.h>
#include <armigo/services/nvdrv.h>
#include <armigo/graphics/engines.h>
#include <armigo/runtime/presentation.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <optional>
#include <sstream>
#include <cstring>
#include <limits>
#include <new>
#include <deque>
#include <functional>
#include <thread>
#include <vector>

namespace armigo::services::nvdrv {

struct GpuChannelState {
    graphics::GpfifoChannel decoder;
    graphics::EngineFrontEnd engines;
    std::uint64_t submissions = 0;
    std::string stop_diagnostic;
};

// Each client has one FIFO across all channels sharing its address spaces.
// Requests own their entries, captured commands and immutable mappings. A
// request that cannot capture commands waits for live processing before returning.
// The queue never owns its descriptor table.
class GpuWorker {
public:
    using Work = std::function<void(std::stop_token)>;
    static constexpr std::size_t max_command_bytes = 4u << 20;
    static constexpr std::size_t max_inflight_bytes = 64u << 20;
    struct Admission {
        GpuWorker& worker;
        std::size_t bytes;
        ~Admission() {
            const std::lock_guard lock(worker.mutex);
            worker.inflight_bytes -= bytes;
            --worker.inflight_submissions;
            worker.ready.notify_all();
        }
    };
    GpuWorker(horizon::GuestBuffers memory, std::shared_ptr<void> owner, std::function<void()> poll)
        : memory(memory), owner(std::move(owner)), poll(std::move(poll)), thread([this](std::stop_token stop) {
            std::unique_lock lock(mutex);
            for (;;) {
                ready.wait_for(lock, std::chrono::milliseconds(2), [&] { return closing || !queue.empty(); });
                if (queue.empty() && closing) break;
                if (queue.empty()) {
                    active = true;
                    lock.unlock();
                    this->poll();
                    lock.lock();
                    active = false;
                    ready.notify_all();
                    continue;
                }
                auto work = std::move(queue.front());
                queue.pop_front();
                active = true;
                ready.notify_all();
                lock.unlock();
                work.run(stop);
                this->poll();
                // Destroy the payload before returning its capacity. Admission
                // release takes mutex, so it also precedes relocking below.
                work.run = {};
                work.admission.reset();
                lock.lock();
                active = false;
                ready.notify_all();
            }
        }) {}
    ~GpuWorker() { stop(); if (thread.joinable()) thread.join(); }
    std::unique_ptr<Admission> reserve(std::size_t bytes) {
        std::unique_lock lock(mutex);
        ready.wait(lock, [&] {
            return closing || (inflight_submissions < 64 &&
                               bytes <= max_inflight_bytes - inflight_bytes);
        });
        if (closing) return {};
        auto admission = std::unique_ptr<Admission>(new Admission{*this, bytes});
        ++inflight_submissions;
        inflight_bytes += bytes;
        return admission;
    }
    bool enqueue(Work work, std::unique_ptr<Admission> admission) {
        const std::lock_guard lock(mutex);
        if (closing) return false;
        queue.emplace_back();
        queue.back().run = std::move(work);
        queue.back().admission = std::move(admission);
        ready.notify_all();
        return true;
    }
    void drain() {
        std::unique_lock lock(mutex);
        ready.wait(lock, [&] { return queue.empty() && !active; });
    }
    void stop() noexcept {
        std::lock_guard lock(mutex);
        closing = true;
        thread.request_stop();
        ready.notify_all();
    }
    horizon::GuestBuffers memory;
    std::shared_ptr<void> owner;
private:
    std::mutex mutex;
    std::condition_variable ready;
    struct QueuedWork {
        Work run;
        std::unique_ptr<Admission> admission;
    };
    std::deque<QueuedWork> queue;
    // Includes the active job and a submitter capturing commands, not just the
    // waiting queue. Request serialization permits only one capturing submitter.
    std::size_t inflight_bytes = 0, inflight_submissions = 0;
    bool active = false, closing = false;
    std::function<void()> poll;
    std::jthread thread;
};

DescriptorTable::DescriptorTable() = default;
DescriptorTable::~DescriptorTable() {
    if (gpu) {
        gpu->stop();
        syncpoint_changed.notify_all();
        gpu.reset();
    }
}

namespace {

using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

// The driver reaches its GPU through this control node. Operations on it are
// implemented one at a time; anything else answers with the driver result word.
constexpr const char* nvgpu_control_device = "/dev/nvhost-ctrl-gpu";
// The GPU channel node owns the GPFIFO channel and is bound to an address space.
constexpr const char* nvgpu_channel_device = "/dev/nvhost-gpu";
// The channel binds the nvmap descriptor its allocations go through.
constexpr std::uint32_t nvgpu_channel_set_nvmap = 0x40044801u;
// GPFIFO allocation (the 0x20-byte extended form the driver uses).
constexpr std::uint32_t nvgpu_channel_alloc_gpfifo = 0xc020481au;
// The channel object-context and no-output setup operations.
constexpr std::uint32_t nvgpu_channel_alloc_obj_ctx = 0xc0104809u;
constexpr std::uint32_t nvgpu_channel_zcull_bind = 0xc010480bu;
constexpr std::uint32_t nvgpu_channel_set_error_notifier = 0xc018480cu;
constexpr std::uint32_t nvgpu_channel_set_priority = 0x4004480du;
constexpr std::uint32_t nvgpu_channel_set_timeout = 0x40044803u;
constexpr std::uint32_t nvgpu_channel_set_timeslice = 0xc004481du;
// Channel client data the driver stores on the GPFIFO channel.
constexpr std::uint32_t nvgpu_channel_set_client_data = 0x40084714u;
constexpr std::uint32_t nvgpu_channel_get_client_data = 0x80084715u;
// First command submission: the milestone the runtime stops at.
constexpr std::uint32_t nvgpu_channel_submit_gpfifo = 0xc018481bu;
// The address-space node carries the GPU page-table operations.
constexpr const char* nvgpu_address_space_device = "/dev/nvhost-as-gpu";
// The memory node owns the driver's memory handles.
constexpr const char* nvmap_device = "/dev/nvmap";
// Memory create: {u32 size, s32 handle}, answered with the new handle id.
constexpr std::uint32_t nvmap_create = 0xc0080101u;
// GetId: {u32 exported id, u32 client handle}.
constexpr std::uint32_t nvmap_get_id = 0xc008010eu;
// Memory allocate: {s32 handle, s32 heap mask, s32 flags, s32 align, s64 kind,
// u64 address}, answered with the address the guest reads and writes.
constexpr std::uint32_t nvmap_alloc = 0xc0200104u;
// Memory free: {s32 handle, s32 padding, u64 address, u32 size, s32 flags}.
constexpr std::uint32_t nvmap_free = 0xc0180105u;
// Memory parameter query: {s32 handle, s32 parameter, s32 value}.
constexpr std::uint32_t nvmap_param = 0xc00c0109u;
// Handle parameters the driver queries.
constexpr std::uint32_t nvmap_param_size = 1;
constexpr std::uint32_t nvmap_param_align = 2;
constexpr std::uint32_t nvmap_param_heap = 4;
constexpr std::uint32_t nvmap_param_kind = 5;
constexpr std::uint32_t nvmap_param_compression = 6;
// The console reports its memory as coming from the general system heap.
constexpr std::uint32_t nvmap_heap_mask = 0x40000000u;
// Driver memory is page aligned; the console allocates whole 4 KiB pages.
constexpr std::uint32_t nvmap_page_size = 0x1000;
// GPU control-node ioctls the driver issues while it brings the device up.
constexpr std::uint32_t nvgpu_get_characteristics = 0xc0b04705u;
constexpr std::uint32_t nvgpu_get_tpc_masks = 0xc0184706u;
// ZCull context size: a single u32 the driver reads before creating the channel.
constexpr std::uint32_t nvgpu_zcull_get_ctx_size = 0x80044701u;
// ZCull information the driver reads after the context size.
constexpr std::uint32_t nvgpu_zcull_get_info = 0x80284702u;
// The driver reports a one-byte zcull context; the value is not used to size
// guest memory before a zcull operation is submitted.
constexpr std::uint32_t nvgpu_zcull_ctx_size = 1;
// The console GPU has two texture processing clusters, so both mask bits are set.
constexpr std::uint32_t nvgpu_tpc_mask = 3;
// Address-space initialize: read/write arguments of 0x28 bytes.
constexpr std::uint32_t nvgpu_as_initialize = 0x40284109u;
// Address-space region query: read/write arguments of 0x40 bytes.
constexpr std::uint32_t nvgpu_as_get_va_regions = 0xc0404108u;
// The two windows a bound address space reports: system-memory small pages first,
// then the big-page window. The console keeps 4 KiB small pages.
constexpr std::uint32_t nvgpu_small_page_size = 0x1000;
constexpr std::size_t nvgpu_va_region_size = 0x18;
// Address-space allocation: {u32 pages, u32 page size, u32 flags, u32 padding,
// u64 offset or alignment}. The fixed-offset flag binds the run the guest names.
constexpr std::uint32_t nvgpu_as_alloc_space = 0xc0184102u;
constexpr std::uint32_t nvgpu_as_flag_fixed = 1u;
constexpr std::uint32_t nvgpu_as_flag_remap_subrange = 0x100u;
// The GPU address space is 40 bits on the console generation this baseline targets.
constexpr std::uint64_t nvgpu_address_space_limit = 1ull << 40;
// Address-space mapping: {u32 flags, u32 kind, s32 memory handle, u32 page size,
// u64 buffer offset, u64 mapping size, u64 offset or alignment}. The bound GPU
// address replaces the offset when the caller did not fix it.
constexpr std::uint32_t nvgpu_as_map_buffer = 0xc0284106u;
// UnmapBuffer: {u64 GPU offset}. An address outside all mappings is a no-op.
constexpr std::uint32_t nvgpu_as_unmap_buffer = 0xc0084105u;
// Address-space bind: one u32 naming the GPU channel descriptor.
constexpr std::uint32_t nvgpu_as_bind_channel = 0x40044101u;

// Driver result words. Zero is success; the rest are the values the driver
// client reports as NvOsDrv* failures. The same enumeration gives
// SharedMemoryTooSmall = 0x1000, which the client tests explicitly.
constexpr std::uint32_t nv_result_not_implemented = 1;
constexpr std::uint32_t nv_result_timeout = 5;
constexpr std::uint32_t nv_result_invalid_value = 11;
constexpr std::uint32_t nv_result_bad_parameter = 3;
constexpr std::uint32_t nv_result_insufficient_memory = 6;
constexpr std::uint32_t nv_result_invalid_address = 9;
constexpr std::uint32_t nv_result_invalid_size = 10;
// The AS adapter normalizes core errno to positive magnitude; those values
// take the public converter's default result rather than its negative-errno arms.
constexpr std::uint32_t nv_result_unrecognized_error = 0x3000f;
constexpr std::uint32_t nv_result_access_denied = 0x30010;

struct RetainedNvMapMapping {
    horizon::GuestMemoryMapper mapper;
    std::uint64_t address = 0;
    std::uint64_t size = 0;
    bool active = false;
    RetainedNvMapMapping(horizon::GuestMemoryMapper source,
                         std::uint64_t mapped_address,
                         std::uint64_t mapped_size)
        : mapper(std::move(source)), address(mapped_address), size(mapped_size) {}
    ~RetainedNvMapMapping() {
        if (active && mapper.unmap) mapper.unmap(mapper.context, address, size);
    }
};

// Imported presentation backing is a logical nvmap reference. Renderer caches
// retain the underlying mapping directly and do not extend that reference.
struct ImportedNvMapReference {
    std::shared_ptr<void> mapping;
};

class RetainedGraphicsPin final : public runtime::PinnedGuestMemory {
public:
    RetainedGraphicsPin(runtime::GuestMemoryPin pin, std::shared_ptr<void> export_owner)
        : export_owner_(std::move(export_owner)), pin_(std::move(pin)) {}

    horizon::AddressRange range() const noexcept override { return pin_->range(); }
    memory::Permission permissions() const noexcept override { return pin_->permissions(); }
    std::span<const std::byte> bytes() const noexcept override { return pin_->bytes(); }
    std::span<std::byte> writable_bytes() noexcept override {
        return pin_->writable_bytes();
    }

private:
    std::shared_ptr<void> export_owner_;
    runtime::GuestMemoryPin pin_;
};

runtime::LaunchFailure graphics_failure(runtime::LaunchError code,
                                        std::string operation,
                                        std::string detail) {
    return runtime::LaunchFailure{.stage = runtime::LaunchStage::graphics,
                                  .code = code,
                                  .operation = std::move(operation),
                                  .detail = std::move(detail)};
}

bool same_owner(const std::shared_ptr<void>& left,
                const std::shared_ptr<void>& right) noexcept {
    return left && right && !left.owner_before(right) && !right.owner_before(left);
}

std::optional<services::GraphicsBufferBacking> retain_exported_memory(
    const std::shared_ptr<DescriptorTable>& descriptors, std::uint32_t export_id) {
    const std::lock_guard lock(descriptors->mutex);
    const auto found = descriptors->memory_handles.find(export_id);
    if (found == descriptors->memory_handles.end() || !found->second.exported ||
        found->second.guest_address == 0 || !found->second.retained_mapping) {
        return std::nullopt;
    }
    const auto previous = descriptors->retained_exports.find(export_id);
    auto imported = previous != descriptors->retained_exports.end()
        ? previous->second.owner.lock() : std::shared_ptr<void>{};
    if (!imported) {
        auto reference = std::make_shared<ImportedNvMapReference>(
            ImportedNvMapReference{found->second.retained_mapping});
        imported = std::shared_ptr<void>(std::move(reference),
                                         found->second.retained_mapping.get());
    }
    services::GraphicsBufferBacking backing{
        export_id,
        found->second.alignment != 0 ? found->second.alignment : nvmap_page_size,
        found->second.guest_address,
        found->second.size,
        std::move(imported)};
    descriptors->retained_exports[export_id] = RetainedExportState{
        backing.alignment, backing.guest_address, backing.size, backing.owner};
    return backing;
}

// The GPU a Horizon console reports: a Tegra X1 (GM20B) with one GPC and
// two TPCs. The guest SDK selects its GPU code paths from these values, so they
// describe the emulated console rather than the host GPU.
struct GpuCharacteristics {
    std::int32_t architecture;
    std::int32_t implementation;
    std::int32_t revision;
    std::int32_t gpc_count;
    std::int64_t l2_cache_size;
    std::int64_t on_board_video_memory_size;
    std::int32_t tpc_per_gpc;
    std::int32_t bus_type;
    std::int32_t big_page_size;
    std::int32_t compression_page_size;
    std::int32_t pde_coverage_bit_count;
    std::int32_t available_big_page_sizes;
    std::int32_t gpc_mask;
    std::int32_t sm_architecture_version;
    std::int32_t sm_architecture_spa_version;
    std::int32_t sm_warp_count;
    std::int32_t virtual_address_bit_count;
    std::int32_t reserved;
    std::int64_t flags;
    std::int32_t twod_class;
    std::int32_t threed_class;
    std::int32_t compute_class;
    std::int32_t gpfifo_class;
    std::int32_t inline_to_memory_class;
    std::int32_t dma_copy_class;
    std::int32_t max_fbps_count;
    std::int32_t fbp_enable_mask;
    std::int32_t max_ltc_per_fbp;
    std::int32_t max_lts_per_ltc;
    std::int32_t max_tex_per_tpc;
    std::int32_t max_gpc_count;
    std::int32_t rop_l2_enable_mask0;
    std::int32_t rop_l2_enable_mask1;
    std::int64_t chip_name;
    std::int64_t gr_compbit_store_base_hw;
};
static_assert(sizeof(GpuCharacteristics) == 0xa0,
              "the driver expects a 0xa0-byte GPU description");

GpuCharacteristics tegra_x1_gpu_characteristics() {
    GpuCharacteristics characteristics{};
    characteristics.architecture = 0x120;
    characteristics.implementation = 0xb;
    characteristics.revision = 0xa1;
    characteristics.gpc_count = 1;
    characteristics.l2_cache_size = 0x40000;
    characteristics.tpc_per_gpc = 2;
    characteristics.bus_type = 0x20;
    characteristics.big_page_size = 0x20000;
    characteristics.compression_page_size = 0x20000;
    characteristics.pde_coverage_bit_count = 0x1b;
    characteristics.available_big_page_sizes = 0x30000;
    characteristics.gpc_mask = 1;
    characteristics.sm_architecture_version = 0x503;
    characteristics.sm_architecture_spa_version = 0x503;
    characteristics.sm_warp_count = 0x80;
    characteristics.virtual_address_bit_count = 0x28;
    characteristics.flags = 0x55;
    characteristics.twod_class = 0x902d;
    characteristics.threed_class = 0xb197;
    characteristics.compute_class = 0xb1c0;
    characteristics.gpfifo_class = 0xb06f;
    characteristics.inline_to_memory_class = 0xa140;
    characteristics.dma_copy_class = 0xb0b5;
    characteristics.max_fbps_count = 1;
    characteristics.max_ltc_per_fbp = 2;
    characteristics.max_lts_per_ltc = 1;
    characteristics.max_gpc_count = 1;
    characteristics.rop_l2_enable_mask0 = 0x21d70;
    // "gm20b" as the driver's chip name string, low byte first.
    characteristics.chip_name = 0x6230326d67;
    return characteristics;
}

template <std::size_t N>
std::uint64_t read_u64(const std::array<std::byte, N>& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned index = 0; index != 8; ++index)
        value |= std::uint64_t{std::to_integer<unsigned char>(bytes[offset + index])}
                 << (8 * index);
    return value;
}

template <std::size_t N>
void write_u64(std::array<std::byte, N>& bytes, std::size_t offset, std::uint64_t value) {
    for (unsigned index = 0; index != 8; ++index)
        bytes[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
}

using namespace horizon::cmif;


// A driver reply whose single result word follows the CMIF reply header. The
// client reads the SFCO word and its result directly after the two-word reply
// header, so the header sits at byte16 and the data starts at byte32.
IpcReply driver_reply(std::uint32_t first, std::uint32_t second, bool two_words) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = two_words ? 40 : 36;
    write(reply.bytes, 4, (reply.size - 8) / 4);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 32, first);
    if (two_words) write(reply.bytes, 36, second);
    return reply;
}


// The driver serializes every request with its descriptor mutex. A publication
// may also run on the renderer pump, where it must try (never wait for) that
// mutex. This scope records only ownership already established by the caller.
thread_local SyncpointTimeline* locked_timeline = nullptr;
thread_local std::unique_lock<std::mutex>* descriptor_lock = nullptr;
class DescriptorLockScope {
public:
    explicit DescriptorLockScope(const std::shared_ptr<DescriptorTable>& table)
        : DescriptorLockScope(table->timeline.get()) {}
    explicit DescriptorLockScope(SyncpointTimeline* timeline)
        : previous_(locked_timeline) { locked_timeline = timeline; }
    ~DescriptorLockScope() { locked_timeline = previous_; }
private:
    SyncpointTimeline* previous_;
};

// The request gate prevents mutations while its state lock is released. Worker
// callbacks can publish fences during backpressure, drains and memory changes.
bool is_syncpoint_request(const IpcMessage& message) {
    const auto type = read(message, 0) & 0xffffu;
    if (type != 4 && type != 6) return false;
    for (std::size_t offset = 16; offset + 24 <= message.size(); offset += 4) {
        if (read(message, offset) != sfci) continue;
        const auto command = read(message, offset + 8);
        if (command != 1 && command != 11 && command != 12) return false;
        const auto operation = read(message, offset + 20);
        switch (operation) {
        case 0xc0080014: case 0xc008001a: case 0x40040015:
        case 0x4004001f: case 0x40040020: case 0xc004001f: case 0xc0040020:
        case 0xc0100019: case 0xc010001d: case 0xc010001e: return true;
        default: return false;
        }
    }
    return false;
}

class RequestLock {
public:
    RequestLock(DescriptorTable& table, const IpcMessage& message)
        : request(table.requests, std::defer_lock), state(table.mutex, std::defer_lock) {
        // Fence progress remains possible while map changes or backpressure wait.
        if (!is_syncpoint_request(message)) request.lock();
        state.lock();
        descriptor_lock = &state;
    }
    ~RequestLock() { descriptor_lock = nullptr; }
private:
    std::unique_lock<std::mutex> request;
    std::unique_lock<std::mutex> state;
};

class DescriptorUnlock {
public:
    DescriptorUnlock() : lock(descriptor_lock), previous(locked_timeline) {
        locked_timeline = nullptr;
        if (lock) lock->unlock();
    }
    ~DescriptorUnlock() {
        if (lock) lock->lock();
        locked_timeline = previous;
    }
private:
    std::unique_lock<std::mutex>* lock;
    SyncpointTimeline* previous;
};

bool syncpoint_expired(const SyncpointTimeline& client, std::uint32_t id,
                       std::uint32_t threshold) noexcept {
    if (id >= client.syncpoints.size()) return false;
    const auto& state = client.syncpoints[id];
    const auto delta = static_cast<std::uint32_t>(state.completed - threshold);
    const bool signed_comparison = id > 96 || ((0xffeull >> (id & 63u)) & 1u) != 0;
    if (signed_comparison) return (delta & 0x80000000u) == 0;
    return static_cast<std::uint32_t>(state.reserved - threshold) >= delta;
}

bool increment_syncpoint(void* raw, std::uint32_t id) noexcept {
    auto& client = *static_cast<SyncpointTimeline*>(raw);
    std::unique_lock lock(client.mutex, std::defer_lock);
    if (locked_timeline != &client) lock.lock();
    if (id >= client.syncpoints.size()) return false;
    const auto completed = ++client.syncpoints[id].completed;
    for (auto& [fd, slots] : client.events) {
        for (auto& event : slots) {
            if (event.pending && event.syncpoint == id &&
                ((completed - event.threshold) & 0x80000000u) == 0) {
                event.pending = false;
                event.event->signal();
            }
        }
    }
    client.changed.notify_all();
    return true;
}

graphics::GuestPublication retain_syncpoint_increment(void* raw, std::uint32_t id,
                                                        std::uint32_t count) {
    const auto owner = static_cast<SyncpointTimeline*>(raw)->weak_from_this();
    const auto client = owner.lock();
    if (!client || client.get() != raw || id >= client->syncpoints.size()) return {};
    return [owner, id, count]() -> std::optional<graphics::CommandResult> {
        const auto client = owner.lock();
        // Driver teardown cancels publication; no callback retains the driver
        // and renderer in a cycle or touches destroyed synchronization state.
        if (!client) return graphics::CommandResult{};
        std::unique_lock lock(client->mutex, std::defer_lock);
        if (locked_timeline != client.get() && !lock.try_lock()) return std::nullopt;
        const DescriptorLockScope scope(client.get());
        for (std::uint32_t i = 0; i != count; ++i) increment_syncpoint(client.get(), id);
        return graphics::CommandResult{};
    };
}

bool expired_syncpoint(void* raw, std::uint32_t id, std::uint32_t threshold) noexcept {
    auto& client = *static_cast<SyncpointTimeline*>(raw);
    std::unique_lock lock(client.mutex, std::defer_lock);
    if (locked_timeline != &client) lock.lock();
    return syncpoint_expired(client, id, threshold);
}

// Multimedia engine adapters. Engines run on their own threads: device memory
// goes through the pins (their own lock) and the persistent guest memory
// access; syncpoints through the timeline.
bool is_multimedia(const std::string& path) {
    return path == "/dev/nvhost-nvdec" || path == "/dev/nvhost-vic";
}

std::optional<std::uint64_t> host1x_guest_address(DescriptorTable& table, std::uint64_t address,
                                                  std::size_t size) {
    const std::lock_guard lock(table.host1x_pins_mutex);
    for (const auto& [handle, pin] : table.host1x_pins) {
        if (address < pin.address || address - pin.address >= pin.size) continue;
        if (size > pin.size - (address - pin.address)) return std::nullopt;
        return pin.guest_address + (address - pin.address);
    }
    return std::nullopt;
}

bool host1x_tracing() {
    static const bool enabled = [] {
        const char* value = std::getenv("ARMIGO_HOST1X_TRACE");
        return value && value[0] == '1';
    }();
    return enabled;
}

bool host1x_read(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
    auto& table = *static_cast<DescriptorTable*>(raw);
    const auto guest = table.gpu ? host1x_guest_address(table, address, out.size()) : std::nullopt;
    const bool read = guest && table.gpu->memory.read_memory(*guest, out);
    if (!read && host1x_tracing())
        armigo::diag::opt_in( "host1x read failed: device 0x%llx size 0x%zx memory %d guest 0x%llx\n",
                     static_cast<unsigned long long>(address), out.size(), table.gpu ? 1 : 0,
                     static_cast<unsigned long long>(guest.value_or(0)));
    return read;
}

bool host1x_write(void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
    auto& table = *static_cast<DescriptorTable*>(raw);
    if (!table.gpu) return false;
    const auto guest = host1x_guest_address(table, address, in.size());
    return guest && table.gpu->memory.write_memory(*guest, in);
}

void host1x_increment(void* raw, std::uint32_t id) noexcept {
    (void)increment_syncpoint(static_cast<DescriptorTable*>(raw)->timeline.get(), id);
}

bool host1x_wait(void* raw, std::uint32_t id, std::uint32_t threshold, std::stop_token stop) noexcept {
    auto& timeline = *static_cast<DescriptorTable*>(raw)->timeline;
    std::unique_lock lock(timeline.mutex);
    while (!syncpoint_expired(timeline, id, threshold)) {
        if (stop.stop_requested() || timeline.stopped) return false;
        timeline.changed.wait_for(lock, std::chrono::milliseconds(10));
    }
    return true;
}

void host1x_fail(void* raw, std::uint32_t syncpoint, std::string detail) noexcept {
    auto& timeline = *static_cast<DescriptorTable*>(raw)->timeline;
    const std::lock_guard lock(timeline.mutex);
    if (syncpoint < timeline.channel_failures.size() && timeline.channel_failures[syncpoint].empty())
        timeline.channel_failures[syncpoint] = std::move(detail);
    timeline.failed.store(true, std::memory_order_release);
    for (auto& [fd, slots] : timeline.events)
        for (auto& event : slots)
            if (event.pending && event.syncpoint == syncpoint) event.event->signal();
    timeline.changed.notify_all();
}

std::uint64_t gpu_ticks(void*) noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    return (ns / 625) * 384 + (ns % 625) * 384 / 625;
}

struct EngineDispatch {
    graphics::EngineFrontEnd& engines;
    const graphics::GpuAddressSpace& memory;
    graphics::CompletionCallbacks completion;
};

graphics::CommandResult dispatch_engine(void* raw, const graphics::MethodCall& call,
                                         std::uint32_t engine) noexcept {
    auto& context = *static_cast<EngineDispatch*>(raw);
    return context.engines.call(call, engine, context.memory, context.completion);
}

bool read_gpu_backing(void* raw, std::uint64_t address,
                      std::span<std::byte> out) noexcept {
    return static_cast<const horizon::GuestBuffers*>(raw)->read_memory(address, out);
}

bool write_gpu_backing(void* raw, std::uint64_t address,
                       std::span<const std::byte> in) noexcept {
    return static_cast<const horizon::GuestBuffers*>(raw)->write_memory(address, in);
}

// Command fetch is a front-end read too. Keep its coherence separate from the
// borrowed backing callbacks used by render preparation, which must not reenter
// the renderer. Read coherence leaves clean write-generation certificates and
// unrelated open GPU work intact.
struct CommandFetch {
    const graphics::GpuAddressSpace& memory;
    const graphics::CompletionCallbacks& completion;
    graphics::CommandResult failure;
    // Fetches walk command buffers sequentially: the previous fetch's
    // mapping is tried before scanning the view.
    const graphics::Mapping* last = nullptr;
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> bytes) noexcept {
        auto& self = *static_cast<CommandFetch*>(raw);
        const auto contains = [&](const graphics::Mapping& mapping) {
            const auto base = mapping.guest_address + mapping.buffer_offset;
            return address >= base && address - base < mapping.mapped_size &&
                   bytes.size() <= mapping.mapped_size - (address - base);
        };
        const auto mappings = self.memory.mappings();
        if (!self.last || !contains(*self.last)) {
            self.last = nullptr;
            // Mapping positions sorted by guest start, kept per thread for the
            // view generation (same generation, same mappings in the same order).
            thread_local std::vector<std::pair<std::uint64_t, std::uint32_t>> by_guest;
            // Running maximum of mapping ends over the sorted starts: the walk
            // below stops once no earlier mapping can reach the address.
            thread_local std::vector<std::uint64_t> reach;
            thread_local std::uint64_t indexed_generation = 0;
            thread_local std::size_t indexed_count = 0;
            if (indexed_generation != self.memory.generation() || indexed_count != mappings.size()) {
                by_guest.clear();
                by_guest.reserve(mappings.size());
                for (std::uint32_t i = 0; i < mappings.size(); ++i)
                    by_guest.emplace_back(mappings[i].guest_address + mappings[i].buffer_offset, i);
                std::sort(by_guest.begin(), by_guest.end());
                reach.resize(by_guest.size());
                std::uint64_t furthest = 0;
                for (std::size_t i = 0; i < by_guest.size(); ++i) {
                    furthest = std::max(furthest, by_guest[i].first + mappings[by_guest[i].second].mapped_size);
                    reach[i] = furthest;
                }
                indexed_generation = self.memory.generation();
                indexed_count = mappings.size();
            }
            auto it = std::upper_bound(by_guest.begin(), by_guest.end(),
                                       std::pair{address, std::numeric_limits<std::uint32_t>::max()});
            while (it != by_guest.begin()) {
                --it;
                if (reach[std::size_t(it - by_guest.begin())] <= address) break;
                if (contains(mappings[it->second])) {
                    self.last = &mappings[it->second];
                    break;
                }
            }
        }
        if (!self.last) return false;
        const auto& mapping = *self.last;
        const auto base = mapping.guest_address + mapping.buffer_offset;
        const auto& rendering = self.completion.rendering;
        self.failure = rendering.read_coherence
            ? rendering.read_coherence(rendering.context, self.memory,
                mapping.gpu_base + (address - base), bytes.size())
            : rendering.coherence(rendering.context, self.memory, 0, 0);
        if (!self.failure.ok()) return false;
        const auto callbacks = self.memory.backing_callbacks();
        return callbacks.read && callbacks.read(callbacks.context, address, bytes);
    }
};


graphics::Mapping render_mapping(const AddressMapping& mapping) {
    return {mapping.offset, mapping.size, mapping.buffer_offset,
            mapping.guest_address, mapping.backing_size, mapping.page_size,
            mapping.retained_mapping};
}

void trace_mapping(const char* operation, const DescriptorTable* table,
                   const Device& device, const AddressMapping& mapping) noexcept {
    if (![] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }()) return;
    armigo::diag::opt_in(
        "nvdrv_mapping operation=%s table=%p space=%u handle=%u gpu=%#llx size=%#llx "
        "offset=%#llx guest=%#llx backing=%#llx owner=%p sparse=%u page=%#llx kind=%#x\n",
        operation, static_cast<const void*>(table), device.fd, mapping.memory_handle,
        static_cast<unsigned long long>(mapping.offset),
        static_cast<unsigned long long>(mapping.size),
        static_cast<unsigned long long>(mapping.buffer_offset),
        static_cast<unsigned long long>(mapping.guest_address),
        static_cast<unsigned long long>(mapping.backing_size),
        mapping.retained_mapping.get(), unsigned(mapping.sparse_remap),
        static_cast<unsigned long long>(mapping.page_size), unsigned(mapping.kind));
}


void poll_backend(DescriptorTable& client) noexcept {
    try {
        graphics::RenderCallbacks rendering;
        {
            const std::lock_guard lock(client.mutex);
            if (!client.timeline->failure.empty()) return;
            rendering = client.rendering;
        }
        if (!rendering.status) return;
        const auto result = rendering.status(rendering.context);
        if (result.ok()) return;
        const std::lock_guard lock(client.mutex);
        if (client.timeline->failure.empty()) client.timeline->failure = result.detail;
        client.timeline->failed.store(true, std::memory_order_release);
        for (auto& [fd, slots] : client.events)
            for (auto& event : slots)
                if (event.pending) event.event->signal();
        client.syncpoint_changed.notify_all();
    } catch (...) {
        // Status observation must never terminate the queue worker.
    }
}

void drain_worker(DescriptorTable& client) {
    if (client.gpu) {
        client.gpu->drain();
        const auto rendering = client.rendering;
        if (rendering.memory_changed) {
            const auto result = rendering.memory_changed(rendering.context, {}, {},
                {read_gpu_backing, write_gpu_backing, &client.gpu->memory});
            if (!result.ok()) {
                const std::lock_guard lock(client.mutex);
                if (client.timeline->failure.empty()) client.timeline->failure = result.detail;
                client.timeline->failed.store(true, std::memory_order_release);
                for (auto& [fd, slots] : client.events)
                    for (auto& event : slots)
                        if (event.pending) event.event->signal();
                client.syncpoint_changed.notify_all();
            }
        }
    }
}

// Call with the timeline locked, after leaving any renderer callback. Both
// submission paths allow fence waiters to register while processing runs.
void record_channel_failure(DescriptorTable& client, GpuChannelState& state,
                            std::uint32_t syncpoint_id, std::string failure) {
    client.timeline->channel_failures[syncpoint_id] = failure;
    state.stop_diagnostic = std::move(failure);
    client.timeline->failed.store(true, std::memory_order_release);
    for (auto& [fd, slots] : client.events)
        for (auto& event : slots)
            if (event.pending && event.syncpoint == syncpoint_id) event.event->signal();
    client.syncpoint_changed.notify_all();
}

std::string process_submission(GpuChannelState& state, const graphics::GpuAddressSpace& memory,
    EngineDispatch& context, std::span<const std::uint64_t> captured, std::uint32_t flags,
    std::uint32_t channel, std::uint32_t syncpoint_id, std::uint64_t submission,
    SyncpointState syncpoint,
    std::optional<std::span<const std::byte>> captured_commands = std::nullopt) {
    CommandFetch fetch{memory, context.completion, {}};
    std::optional<graphics::GpuAddressSpace> command_memory;
    if (!captured_commands &&
        (context.completion.rendering.ordered_effect || context.completion.rendering.ordered_write) &&
        (context.completion.rendering.read_coherence || context.completion.rendering.coherence)) {
        auto made = graphics::make_address_space(memory.mappings(), {CommandFetch::read, nullptr, &fetch});
        if (auto* view = std::get_if<graphics::GpuAddressSpace>(&made)) command_memory = std::move(*view);
        else return "GPU command fetch view could not be retained";
    }
    auto outcome = state.decoder.process(captured, command_memory ? *command_memory : memory,
        dispatch_engine, &context, graphics::GpfifoChannel::default_word_budget, captured_commands);
    if (!fetch.failure.ok()) {
        outcome.status = fetch.failure.status;
        outcome.detail = fetch.failure.detail;
    }
    if (!outcome.ok()) {
        std::ostringstream text;
        text << "GPU processing stopped: channel " << channel << " submission " << submission
             << " engine 0x" << std::hex << outcome.engine << " method 0x" << outcome.method
             << " argument 0x" << outcome.argument << " at 0x" << outcome.address
             << std::dec << " subchannel " << outcome.subchannel << ": " << outcome.detail
             << "; syncpoint " << syncpoint_id << " completed " << syncpoint.completed
             << " reserved " << syncpoint.reserved;
        if (const auto& stopped = state.engines.rendering_stop(); stopped) {
            const char* kind = "unknown";
            switch (stopped->kind) {
            case graphics::RenderKind::Clear: kind = "clear"; break;
            case graphics::RenderKind::Draw: kind = "draw"; break;
            case graphics::RenderKind::Compute: kind = "compute dispatch"; break;
            case graphics::RenderKind::Blit: kind = "2D blit"; break;
            }
            text << "; first rendering operation " << kind
                 << " engine 0x" << std::hex << stopped->engine << " method 0x" << stopped->original.method
                 << " argument 0x" << stopped->original.argument;
            for (std::size_t index = 0; index < stopped->color_targets.size(); ++index) {
                const auto& target = stopped->color_targets[index];
                text << "; color[" << std::dec << index << "] address 0x" << std::hex << target.address
                     << " format 0x" << target.format << " layout 0x" << target.tile_mode
                     << std::dec << " size " << target.width << 'x' << target.height
                     << " depth " << target.depth << " pitch " << target.array_pitch
                     << " layer " << target.base_layer;
            }
            const auto& depth = stopped->depth;
            text << "; depth address 0x" << std::hex << depth.address << " format 0x" << depth.format
                 << " layout 0x" << depth.tile_mode << std::dec << " size " << depth.width << 'x' << depth.height
                 << " pitch " << depth.array_pitch << " enabled " << depth.enabled;
            for (std::size_t index = 0; index < stopped->programs.size(); ++index) {
                const auto& program = stopped->programs[index];
                text << "; program[" << index << "] address 0x" << std::hex << program.address
                     << " offset 0x" << program.offset << " control 0x" << program.control
                     << std::dec << " enabled " << program.enabled;
            }
            for (std::size_t stage = 0; stage < stopped->constant_buffers.size(); ++stage) {
                for (std::size_t slot = 0; slot < stopped->constant_buffers[stage].size(); ++slot) {
                    const auto& binding = stopped->constant_buffers[stage][slot];
                    if (binding.enabled)
                        text << "; constant buffer[" << std::dec << stage << "][" << slot
                             << "] address 0x" << std::hex << binding.address
                             << std::dec << " size " << binding.size;
                }
            }
            if (stopped->has_compute_descriptor)
                text << "; compute descriptor address 0x" << std::hex << stopped->compute_descriptor_address;
            const auto& regs = stopped->registers;
            text << "; clear values 0x" << std::hex << regs[0x360] << ",0x" << regs[0x361]
                 << ",0x" << regs[0x362] << ",0x" << regs[0x363] << " depth 0x" << regs[0x364]
                 << " stencil 0x" << regs[0x368] << "; draw begin 0x" << regs[0x586]
                 << " index format 0x" << regs[0x5f6] << " first " << std::dec << regs[0x5f7]
                 << " count " << regs[0x5f8];
            const auto index_address = (static_cast<std::uint64_t>(regs[0x5f2]) << 32) | regs[0x5f3];
            text << "; index address 0x" << std::hex << index_address
                 << "; surface clip 0x" << regs[0x3fd] << ",0x" << regs[0x3fe]
                 << "; render targets 0x" << regs[0x487] << " color mask[0] 0x" << regs[0x680];
            for (std::size_t index = 0; index < 16; ++index) {
                const auto scissor = 0x380 + index * 4;
                text << "; scissor[" << std::dec << index << "] 0x" << std::hex << regs[scissor]
                     << ",0x" << regs[scissor + 1] << ",0x" << regs[scissor + 2];
            }
            text << "; viewport[0]";
            for (std::size_t index = 0; index < 6; ++index)
                text << (index == 0 ? " 0x" : ",0x") << std::hex << regs[0x280 + index];
            text << "; backend work not completed successfully";
        }
        return text.str();
    }
    if ((flags & 2u) != 0) {
        if (state.decoder.has_pending() || state.engines.has_pending_memory_work()) {
            return "Automatic GPU fence increments require complete packet and memory work";
        }
        // The fence makes this submission's results guest-visible.
        if (const auto coherent = state.engines.increment_syncpoint(
                memory, context.completion, syncpoint_id, 2);
            !coherent.ok()) {
            return std::string("Automatic GPU fence coherence failed: ") + std::string(coherent.detail);
        }
    }
    return {};
}

}  // namespace

void AddressWindow::configure(std::uint64_t start, std::uint64_t size,
                              std::uint64_t page_size) noexcept {
    start_ = start;
    size_ = size;
    page_size_ = page_size;
    cursor_ = start;
    used_.clear();
}

std::uint64_t AddressWindow::next(std::uint64_t address, std::uint64_t alignment) const noexcept {
    if (alignment <= page_size_) return address;
    const auto remainder = address % alignment;
    return remainder == 0 ? address : address + (alignment - remainder);
}

std::optional<std::uint64_t> AddressWindow::allocate(std::uint64_t pages,
                                                     std::uint64_t alignment) noexcept {
    if (page_size_ == 0 || pages == 0) return std::nullopt;
    const auto size = pages * page_size_;
    std::uint64_t candidate = next(start_, alignment);
    for (const auto& [offset, used] : used_) {
        if (candidate + size <= offset) break;
        const auto after = offset + used * page_size_;
        if (after > candidate) candidate = next(after, alignment);
    }
    if (candidate + size > end()) return std::nullopt;
    if (candidate > cursor_) cursor_ = candidate;
    used_[candidate] = pages;
    return candidate;
}

bool AddressWindow::allocate_at(std::uint64_t offset, std::uint64_t pages) noexcept {
    if (page_size_ == 0 || pages == 0) return false;
    if (offset < start_ || offset % page_size_ != 0) return false;
    const auto size = pages * page_size_;
    if (offset + size > end()) return false;
    for (const auto& [used_offset, used] : used_) {
        const auto used_end = used_offset + used * page_size_;
        if (used_end <= offset) continue;
        if (used_offset >= offset + size) break;
        return false;
    }
    used_[offset] = pages;
    return true;
}

bool AddressWindow::release(std::uint64_t offset, std::uint64_t pages) noexcept {
    const auto found = used_.find(offset);
    if (found == used_.end() || found->second != pages) return false;
    used_.erase(found);
    return true;
}

DriverSession::DriverSession() = default;

void DriverSession::set_render_backend(graphics::RenderCallbacks rendering) {
    const std::lock_guard request(descriptors_->requests);
    drain_worker(*descriptors_);
    const std::lock_guard lock(descriptors_->mutex);
    descriptors_->rendering = rendering;
}

void DriverSession::enable_async_gpu(horizon::GuestBuffers memory, std::shared_ptr<void> owner) {
    const std::lock_guard request(descriptors_->requests);
    const std::lock_guard lock(descriptors_->mutex);
    if (!owner || descriptors_->gpu) return;
    descriptors_->gpu = std::make_unique<GpuWorker>(memory, std::move(owner),
        [client = descriptors_.get()] { poll_backend(*client); });
}

void DriverSession::stop_gpu() noexcept {
    const std::lock_guard lock(descriptors_->mutex);
    if (descriptors_->gpu) descriptors_->gpu->stop();
    descriptors_->timeline->stopped = true;
    descriptors_->syncpoint_changed.notify_all();
}

void DriverSession::drain_gpu() {
    const std::lock_guard request(descriptors_->requests);
    drain_worker(*descriptors_);
}

bool DriverSession::notify_memory_changed(
    std::span<const graphics::Mapping> removed,
    std::span<const std::shared_ptr<void>> released,
    const horizon::GuestBuffers& buffers) noexcept {
    if (descriptors_->gpu) {
        const DescriptorUnlock unlocked;
        descriptors_->gpu->drain();
    }
    const auto callback = descriptors_->rendering.memory_changed;
    if (!callback || (!descriptors_->gpu && removed.empty() && released.empty())) return true;
    const auto result = [&] {
        const DescriptorUnlock unlocked;
        return callback(descriptors_->rendering.context, removed, released,
            {read_gpu_backing, write_gpu_backing, const_cast<horizon::GuestBuffers*>(&buffers)});
    }();
    if (result.ok()) return true;
    try {
        ioctl_diagnostic_.assign(result.detail);
        if (ioctl_diagnostic_.empty())
            ioctl_diagnostic_ = "Renderer rejected a driver memory change";
        diagnostic_ = ioctl_diagnostic_.c_str();
    } catch (const std::bad_alloc&) {
        diagnostic_ = "Renderer rejected a driver memory change and its diagnostic could not be retained";
    }
    return false;
}

DriverSession::DriverSession(const DriverSession& parent)
    : converted_(parent.converted_), descriptors_(parent.descriptors_),
      mapper_(parent.mapper_), work_memory_(parent.work_memory_) {}

void DriverSession::refuse_gpu_devices(const char* reason) noexcept { refused_gpu_devices_ = reason; }

IpcReply DriverSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply DriverSession::request(const IpcMessage& message, std::uint64_t) {
    const RequestLock lock(*descriptors_, message);
    const DescriptorLockScope publication_scope(descriptors_);
    diagnostic_ = nullptr;
    return domain_request(message, {});
}

IpcReply DriverSession::request_with_buffers(const IpcMessage& message, std::uint64_t,
                                             const horizon::GuestBuffers& buffers) {
    const RequestLock lock(*descriptors_, message);
    const DescriptorLockScope publication_scope(descriptors_);
    diagnostic_ = nullptr;
    return domain_request(message, buffers);
}

IpcReply DriverSession::domain_request(const IpcMessage& message,
                                       const horizon::GuestBuffers& buffers) {
    // The message type is the leading halfword; the rest of the first word carries the
    // HIPC descriptor counts, so comparing the whole word rejects every such request.
    const auto type = read(message, 0) & 0xffffu;
    if (type == 5 || type == 7) {
        // Session control: report the actual pointer-buffer capacity of this host
        // session, convert to a domain, and clone onto the same port.
        if (read(message, 16) != sfci) return unsupported("Driver control magic is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_reply(0, 2);
        if (command == 0) {
            if (converted_) return unsupported("Repeated driver domain conversion is unsupported");
            converted_ = true;
            return control_reply(1, 4);
        }
        if (command == 2 || command == 4) {
            // Clone, and its tagged variant used by CloneNvDrvForRealtime: both hand
            // the caller a second session on the same port, and the tagged form also
            // carries a u32 the client keeps for its own bookkeeping.
            if (command == 4) clone_tag_ = read(message, 32);
            auto clone = std::make_shared<DriverSession>(*this);
            IpcReply reply = control_reply(0, 4);
            reply.transferred[0] = std::move(clone);
            reply.transferred_count = 1;
            return reply;
        }
        return unsupported("Driver control command is unsupported");
    }
    if (type != 4 && type != 6) return unsupported("Driver request shape is unsupported");
    // The CMIF header sits after the request's handle, buffer and domain areas, so
    // its position follows the request shape: byte16 for a plain request, byte32
    // when it carries handles or one pointer buffer, byte48 for the PID variant and
    // byte64 once counted HIPC descriptors precede it. The magic itself is the
    // reliable marker, so the header is found by scanning the fixed message for it.
    std::size_t cmif = 0;
    for (std::size_t offset = 16; offset + 12 <= message.size(); offset += 4) {
        if (read(message, offset) == sfci) {
            cmif = offset;
            break;
        }
    }
    if (cmif == 0) return unsupported("Driver request shape is unsupported");
    const auto candidate = cmif == 16 ? 0 : read(message, cmif - 16);
    const bool domain_header = cmif != 16 && (candidate & 0xffff) == 1 && (candidate >> 16) >= 16;
    if (domain_header) {
        const auto size = candidate >> 16;
        const auto id = read(message, cmif - 12);
        if (size > 64 || id != 1) return unsupported("Driver domain header is unsupported");
    }
    const auto command = read(message, cmif + 8);
    switch (command) {
    case 3: {
        // Initialize: the client hands over the current-process pseudo handle and
        // the transfer memory that backs the driver work area. The service records
        // them and answers with a plain success; the dispatcher places its header
        // after the request's handle array.
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 48;
        write(reply.bytes, 4, 10);
        write(reply.bytes, 16, sfco);
        work_memory_ = read(message, 16);
        return reply;
    }
    case 13: {
        // Command 13 reaches the driver through the same virtual interface as the
        // binding command, with an eight-byte payload and no output of its own
        // (sdk sub_466FE0 copies the payload and calls the command sender). The
        // session records the requested mode so later commands can be attributed.
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 48;
        write(reply.bytes, 4, 10);
        write(reply.bytes, 16, sfco);
        driver_mode_ = read(message, 32);
        return reply;
    }
    case 8: {
        // Command 8 reaches the driver through its virtual interface with a second
        // session handle and one inline word (sdk sub_466DFC hands eight payload
        // bytes to the command sender). The session records the binding so later
        // commands can be attributed to it.
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 48;
        write(reply.bytes, 4, 10);
        write(reply.bytes, 16, sfco);
        bound_session_ = read(message, 20);
        bound_value_ = read(message, 48);
        return reply;
    }
    case 0: return open_device(message, cmif, buffers);
    case 4: return query_event(message, cmif);
    case 12: return ioctl3(message, cmif, buffers);
    // Ioctl2 carries the same request word and argument buffer as Ioctl; the
    // submission consumes its separate inline entry buffer.
    case 11: return ioctl(message, cmif, buffers);
    case 1: return ioctl(message, cmif, buffers);
    case 2: return close_device(message, cmif, buffers);
    default: return unsupported("Driver command is not implemented");
    }
}

IpcReply DriverSession::open_device(const IpcMessage& message, std::size_t cmif,
                                    const horizon::GuestBuffers& buffers) {
    // Open carries the device path. The generated client passes it as a string
    // buffer, so the guest buffer is authoritative and the inline area is the
    // fallback for a short path that travels with the message.
    std::string path;
    if (!buffers.empty()) {
        // The path arrives in the request buffer without a terminator, so take the printable
        // run rather than requiring a C string.
        std::vector<std::byte> raw(buffers[0].size);
        if (buffers.read(0, raw)) {
            for (const auto value : raw) {
                const auto character = std::to_integer<char>(value);
                if (character == 0 || character < 32 || character > 126) break;
                path.push_back(character);
            }
        }
    }
    if (path.empty()) {
        // The path can sit anywhere after the CMIF header: the client's request layout
        // shifts with the handle and descriptor areas it needs. Look for the device node
        // prefix and take the run of printable bytes that follows it.
        for (std::size_t offset = cmif + 16; offset + 5 < message.size(); ++offset) {
            if (std::to_integer<char>(message[offset]) != '/') continue;
            if (std::to_integer<char>(message[offset + 1]) != 'd') continue;
            if (std::to_integer<char>(message[offset + 2]) != 'e') continue;
            if (std::to_integer<char>(message[offset + 3]) != 'v') continue;
            if (std::to_integer<char>(message[offset + 4]) != '/') continue;
            for (; offset < message.size(); ++offset) {
                const auto value = std::to_integer<char>(message[offset]);
                if (value == 0 || value < 32 || value > 126) break;
                path.push_back(value);
            }
            break;
        }
    }
    if (path.empty()) return unsupported("Driver open needs a device path");
    if (refused_gpu_devices_ && path.size() > 4 && path.ends_with("-gpu")) return unsupported(refused_gpu_devices_);
    if (descriptors_->next_fd == std::numeric_limits<std::uint32_t>::max())
        return unsupported("Driver descriptors are exhausted");
    const auto fd = descriptors_->next_fd++;
    descriptors_->devices[fd] = Device{.path = path, .fd = fd};
    if (is_multimedia(path)) {
        // A multimedia channel owns a syncpoint and an engine from its open.
        auto& device = descriptors_->devices[fd];
        if (descriptors_->next_gpu_syncpoint >= descriptors_->syncpoints.size()) {
            descriptors_->devices.erase(fd);
            return unsupported("Multimedia channel syncpoints are exhausted");
        }
        device.syncpoint_id = descriptors_->next_gpu_syncpoint++;
        if (!descriptors_->host1x) {
            auto* table = descriptors_.get();
            descriptors_->host1x = std::make_unique<host1x::Hub>(
                host1x::DeviceMemory{table, host1x_read, host1x_write},
                host1x::Syncpoints{table, host1x_increment, host1x_wait},
                host1x::FailureSink{table, host1x_fail});
        }
        descriptors_->host1x->start_device(fd, path == "/dev/nvhost-nvdec" ? host1x::ChannelType::video_decoder
                                                                            : host1x::ChannelType::video_compositor,
                                           device.syncpoint_id);
    }
    // Open answers with the descriptor and a separate zero result word; the
    // client reads both.
    return driver_reply(fd, 0, true);
}

IpcReply DriverSession::close_device(const IpcMessage& message, std::size_t cmif,
                                     const horizon::GuestBuffers& buffers) {
    const auto fd = read(message, cmif + 16);
    auto found = descriptors_->devices.find(fd);
    if (found == descriptors_->devices.end())
        return unsupported("Driver close names an unknown descriptor");
    try {
        std::vector<graphics::Mapping> removed;
        std::vector<std::shared_ptr<void>> released;
        removed.reserve(found->second.mappings.size());
        released.reserve(found->second.mappings.size());
        for (const auto& mapping : found->second.mappings) {
            removed.push_back(render_mapping(mapping));
            const auto& owner = mapping.retained_mapping;
            if (!owner) continue;
            const bool already_released = std::any_of(
                released.begin(), released.end(), [&](const auto& candidate) {
                    return same_owner(candidate, owner);
                });
            if (already_released) continue;
            const bool has_handle = std::any_of(
                descriptors_->memory_handles.begin(), descriptors_->memory_handles.end(),
                [&](const auto& entry) {
                    return same_owner(entry.second.retained_mapping, owner);
                });
            const bool has_other_mapping = std::any_of(
                descriptors_->devices.begin(), descriptors_->devices.end(),
                [&](const auto& entry) {
                    if (entry.first == fd) return false;
                    return std::any_of(entry.second.mappings.begin(), entry.second.mappings.end(),
                        [&](const AddressMapping& candidate) {
                            return same_owner(candidate.retained_mapping, owner);
                        });
                });
            if (!has_handle && !has_other_mapping) released.push_back(owner);
        }
        if (!notify_memory_changed(removed, released, buffers)) return {};
    } catch (const std::bad_alloc&) {
        return unsupported("Driver close could not prepare renderer memory notification");
    }
    for (const auto& mapping : found->second.mappings)
        trace_mapping("close", descriptors_.get(), found->second, mapping);
    if (is_multimedia(found->second.path) && descriptors_->host1x) {
        // The engine thread may need the descriptor state to finish; release it
        // while the engine stops.
        auto* hub = descriptors_->host1x.get();
        const auto stopped_fd = found->first;
        {
            const DescriptorUnlock unlocked;
            hub->stop_device(stopped_fd);
        }
        found = descriptors_->devices.find(stopped_fd);
        if (found == descriptors_->devices.end()) return unsupported("Driver close lost its device");
    }
    // A control descriptor's event slots close with it; the events stay alive
    // for the handles the guest holds.
    if (found->second.path == "/dev/nvhost-ctrl") descriptors_->events.erase(found->first);
    descriptors_->devices.erase(found);
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 48;
    write(reply.bytes, 4, 10);
    write(reply.bytes, 32, sfco);
    return reply;
}

IpcReply DriverSession::query_event(const IpcMessage& message, std::size_t cmif) {
    const auto fd = read(message, cmif + 16);
    const auto encoded = read(message, cmif + 20);
    const auto device = descriptors_->devices.find(fd);
    if (device == descriptors_->devices.end()) return driver_reply(nv_result_invalid_value, 0, false);
    if (device->second.path == "/dev/nvhost-ctrl") {
        const bool allocated = (encoded & 0x10000000u) != 0;
        const auto slot = allocated ? encoded & 0xffffu : encoded & 0xfu;
        const auto syncpoint = allocated ? (encoded >> 16) & 0xfffu : encoded >> 4;
        const auto slots = descriptors_->events.find(fd);
        if (slots == descriptors_->events.end() || slot >= slots->second.size() || !slots->second[slot].event ||
            slots->second[slot].syncpoint != syncpoint)
            return driver_reply(nv_result_invalid_value, 0, false);
        auto reply = driver_reply(0, 0, false);
        reply.transferred[0] = slots->second[slot].event;
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
        return reply;
    }
    char text[160]{};
    std::snprintf(text, sizeof text, "Driver event %u on descriptor %u is not implemented", encoded, fd);
    ioctl_diagnostic_ = text;
    diagnostic_ = ioctl_diagnostic_.c_str();
    return driver_reply(nv_result_not_implemented, 0, false);
}

IpcReply DriverSession::ioctl3(const IpcMessage& message, std::size_t cmif,
                              const horizon::GuestBuffers& buffers) {
    return ioctl(message, cmif, buffers);
}

IpcReply DriverSession::ioctl(const IpcMessage& message, std::size_t cmif,
                              const horizon::GuestBuffers& buffers) {
    // The request number and the descriptor it targets are both part of the
    // refusal, so a run names the exact device operation that is still missing.
    // Reporting success here would let the guest continue against an
    // uninitialised device.
    const auto fd = read(message, cmif + 16);
    const auto request = read(message, cmif + 20);
    const auto found = descriptors_->devices.find(fd);
    if (found == descriptors_->devices.end())
        return unsupported("Driver ioctl names an unknown descriptor");
    if (found->second.path == "/dev/nvhost-nvdec" || found->second.path == "/dev/nvhost-vic")
        return multimedia_ioctl(found->second, request, buffers);
    if (found->second.path == "/dev/nvhost-ctrl") {
        switch (request) {
        case 0xc0080014: case 0xc008001a: case 0x40040015:
        case 0x4004001f: case 0x40040020:
        case 0xc0100019: case 0xc010001d: case 0xc010001e:
            return syncpoint_ioctl(fd, request, buffers);
        // EventRegister and EventUnregister also come as read-write requests;
        // their argument is the same event slot.
        case 0xc004001f: case 0xc0040020:
            return syncpoint_ioctl(fd, request & ~0x80000000u, buffers);
        default: break;
        }
    }
    if (found->second.path == nvgpu_control_device) {
        if (request == nvgpu_get_characteristics) return gpu_characteristics(buffers);
        if (request == nvgpu_get_tpc_masks) return gpu_tpc_masks(buffers);
        if (request == nvgpu_zcull_get_ctx_size) return gpu_zcull_get_ctx_size(buffers);
        if (request == nvgpu_zcull_get_info) return gpu_zcull_get_info(buffers);
    }
    if (found->second.path == nvmap_device) {
        if (request == nvmap_create) return memory_create(buffers);
        if (request == nvmap_get_id) return memory_get_id(buffers);
        if (request == nvmap_alloc) return memory_allocate(buffers);
        if (request == nvmap_free) return memory_free(buffers);
        if (request == nvmap_param) return memory_param(buffers);
    }
    if (found->second.path == nvgpu_address_space_device) {
        if (request == nvgpu_as_initialize) return address_space_initialize(found->second, buffers);
        if (request == nvgpu_as_get_va_regions)
            return address_space_va_regions(found->second, buffers);
        if (request == nvgpu_as_alloc_space) return address_space_allocate(found->second, buffers);
        if (request == nvgpu_as_map_buffer) return address_space_map(found->second, buffers);
        if (request == nvgpu_as_unmap_buffer) return address_space_unmap(found->second, buffers);
        if ((request & 0xc000ffffu) == 0xc0004114u)
            return address_space_remap(found->second, request, buffers);
        if (request == nvgpu_as_bind_channel)
            return address_space_bind_channel(found->second, buffers);
    }
    if (found->second.path == nvgpu_channel_device) {
        if (request == nvgpu_channel_set_nvmap)
            return gpu_channel_set_nvmap(found->second, buffers);
        if (request == nvgpu_channel_alloc_gpfifo)
            return gpu_channel_alloc_gpfifo(found->second, buffers);
        if (request == nvgpu_channel_submit_gpfifo)
            return gpu_channel_submit_gpfifo(found->second, buffers);
        if (request == nvgpu_channel_set_client_data)
            return gpu_channel_set_client_data(found->second, buffers);
        if (request == nvgpu_channel_get_client_data)
            return gpu_channel_get_client_data(found->second, buffers);
        if (request == nvgpu_channel_alloc_obj_ctx || request == nvgpu_channel_zcull_bind ||
            request == nvgpu_channel_set_error_notifier || request == nvgpu_channel_set_priority ||
            request == nvgpu_channel_set_timeout || request == nvgpu_channel_set_timeslice)
            return driver_reply(0, 0, false);
    }
    // An operation the service does not implement is answered with the driver
    // result word the firmware uses, not with a transport failure: the guest
    // decides whether the operation is optional. The diagnostic keeps the exact
    // device and request visible to the host.
    char text[160]{};
    std::snprintf(text, sizeof text, "Driver ioctl %#x on %s is not implemented",
                  request, found->second.path.c_str());
    ioctl_diagnostic_ = text;
    diagnostic_ = ioctl_diagnostic_.c_str();
    // The guest continues, so the host log names each missing operation once.
    if (reported_ioctls_.insert(found->second.path + ':' + std::to_string(request)).second)
        armigo::diag::write(armigo::diag::Level::warning,  "Armigo nvdrv: %s.\n", text);
    return driver_reply(nv_result_not_implemented, 0, false);
}

IpcReply DriverSession::gpu_zcull_get_ctx_size(const horizon::GuestBuffers& buffers) {
    if (buffers.empty() || buffers[0].size < 4) return driver_reply(nv_result_invalid_size, 0, false);
    const std::array<std::byte, 4> value{
        static_cast<std::byte>(nvgpu_zcull_ctx_size),
        static_cast<std::byte>(nvgpu_zcull_ctx_size >> 8),
        static_cast<std::byte>(nvgpu_zcull_ctx_size >> 16),
        static_cast<std::byte>(nvgpu_zcull_ctx_size >> 24)};
    if (!buffers.write(0, value)) return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::gpu_zcull_get_info(const horizon::GuestBuffers& buffers) {
    if (buffers.empty() || buffers[0].size < 0x28)
        return driver_reply(nv_result_invalid_size, 0, false);
    // The Tegra X1 zcull surface geometry the driver reports to the client.
    const std::uint32_t values[10] = {0x20, 0x20, 0x400, 0x800, 0x20, 0x20, 0xc0, 0x20, 0x40, 0x10};
    std::array<std::byte, 0x28> raw{};
    for (std::size_t index = 0; index != 10; ++index) {
        for (unsigned byte = 0; byte != 4; ++byte)
            raw[index * 4 + byte] = static_cast<std::byte>((values[index] >> (8 * byte)) & 255);
    }
    if (!buffers.write(0, raw)) return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::gpu_channel_alloc_gpfifo(Device& device,
                                                const horizon::GuestBuffers& buffers) {
    if (buffers.empty() || buffers[0].size < 0x20)
        return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 0x20> arguments{};
    if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto entries = static_cast<std::uint32_t>(std::to_integer<unsigned char>(arguments[0])) |
                         (static_cast<std::uint32_t>(std::to_integer<unsigned char>(arguments[1])) << 8) |
                         (static_cast<std::uint32_t>(std::to_integer<unsigned char>(arguments[2])) << 16) |
                         (static_cast<std::uint32_t>(std::to_integer<unsigned char>(arguments[3])) << 24);
    const auto jobs = static_cast<std::uint32_t>(read_u64(arguments, 0) >> 32);
    if (entries < 16 || (entries & (entries - 1)) != 0 || jobs == 0)
        return driver_reply(nv_result_invalid_value, 0, false);
    if (device.syncpoint_id == 0) {
        if (descriptors_->next_gpu_syncpoint >= descriptors_->syncpoints.size())
            return driver_reply(nv_result_insufficient_memory, 0, false);
        device.syncpoint_id = descriptors_->next_gpu_syncpoint++;
    }
    device.gpfifo_entries = entries;
    // The extended allocation answers with the channel's current syncpoint fence
    // at byte12: {s32 id, u32 value}.
    for (unsigned byte = 0; byte != 4; ++byte) {
        arguments[12 + byte] = static_cast<std::byte>((device.syncpoint_id >> (8 * byte)) & 255);
        arguments[16 + byte] = static_cast<std::byte>((descriptors_->syncpoints[device.syncpoint_id].reserved >> (8 * byte)) & 255);
    }
    if (!buffers.write(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::gpu_channel_set_client_data(Device& device,
                                                   const horizon::GuestBuffers& buffers) {
    if (buffers.empty() || buffers[0].size < 8) return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 8> raw{};
    if (!buffers.read(0, raw)) return driver_reply(nv_result_invalid_address, 0, false);
    std::uint64_t value = 0;
    for (unsigned index = 0; index != 8; ++index)
        value |= std::uint64_t{std::to_integer<unsigned char>(raw[index])} << (8 * index);
    device.client_data = value;
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::gpu_channel_get_client_data(Device& device,
                                                   const horizon::GuestBuffers& buffers) {
    if (buffers.empty() || buffers[0].size < 8) return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 8> raw{};
    for (unsigned index = 0; index != 8; ++index)
        raw[index] = static_cast<std::byte>((device.client_data >> (8 * index)) & 255);
    if (!buffers.write(0, raw)) return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::gpu_channel_submit_gpfifo(Device& device,
                                                 const horizon::GuestBuffers& buffers) {
    if (buffers.empty() || buffers[0].size < 0x18)
        return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 0x18> arguments{};
    if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto get_word = [&](std::size_t offset) {
        return static_cast<std::uint32_t>(read_u64(arguments, offset));
    };
    const auto entries = get_word(8);
    const auto flags = get_word(12);
    const auto fence_id = get_word(16);
    // The final field is four bytes, so do not use the eight-byte reader here.
    std::uint32_t fence_value = 0;
    for (unsigned byte = 0; byte != 4; ++byte)
        fence_value |= std::uint32_t{std::to_integer<unsigned char>(arguments[20 + byte])} << (8 * byte);
    if (descriptors_->gpu) {
        const DescriptorUnlock unlocked;
        poll_backend(*descriptors_);
    }
    if (descriptors_->gpu && !descriptors_->timeline->failure.empty()) {
        ioctl_diagnostic_ = descriptors_->timeline->failure;
        diagnostic_ = ioctl_diagnostic_.c_str();
        return {};
    }
    if (device.gpu_state && !device.gpu_state->stop_diagnostic.empty()) {
        ioctl_diagnostic_ = device.gpu_state->stop_diagnostic;
        diagnostic_ = ioctl_diagnostic_.c_str();
        return {};
    }
    if (device.gpfifo_entries < 16 || entries > device.gpfifo_entries - 3 ||
        entries > graphics::GpfifoChannel::max_submission_entries)
        return driver_reply(nv_result_insufficient_memory, 0, false);
    if ((flags & ~0x117u) != 0 || (flags & 0x101u) == 0x101u)
        return driver_reply(nv_result_invalid_value, 0, false);
    if (descriptors_->gpu && (flags & 1u) && fence_id >= descriptors_->syncpoints.size())
        return driver_reply(nv_result_invalid_value, 0, false);
    if (device.syncpoint_id == 0 || device.bound_address_space == 0)
        return unsupported("GPU submission requires an allocated channel and bound address space");
    const auto address_space = descriptors_->devices.find(device.bound_address_space);
    if (address_space == descriptors_->devices.end())
        return unsupported("GPU channel address space was closed");
    std::vector<std::uint64_t> captured;
    if (entries != 0) {
        std::size_t inline_index = buffers.size();
        const auto bytes = static_cast<std::uint64_t>(entries) * 8;
        for (std::size_t index = 1; index < buffers.size(); ++index) {
            if (buffers[index].address != buffers[0].address && buffers[index].size >= bytes) {
                inline_index = index;
                break;
            }
        }
        if (inline_index == buffers.size())
            return driver_reply(nv_result_invalid_size, 0, false);
        std::vector<std::byte> raw(static_cast<std::size_t>(bytes));
        if (!buffers.read(inline_index, raw))
            return driver_reply(nv_result_invalid_address, 0, false);
        captured.resize(entries);
        for (std::size_t index = 0; index < captured.size(); ++index)
            for (unsigned byte = 0; byte != 8; ++byte)
                captured[index] |= std::uint64_t{std::to_integer<unsigned char>(raw[index * 8 + byte])} << (8 * byte);
    }
    std::vector<graphics::Mapping> mappings;
    for (const auto& mapping : address_space->second.mappings) {
        if (mapping.guest_address == 0 || !mapping.retained_mapping)
            return unsupported("GPU address mapping has no live guest backing");
        mappings.push_back({mapping.offset, mapping.size, mapping.buffer_offset,
                            mapping.guest_address, mapping.backing_size, mapping.page_size,
                            mapping.retained_mapping});
    }
    if (!address_space->second.view_generation)
        address_space->second.view_generation = graphics::allocate_address_space_generation();
    auto resolved = graphics::make_address_space(mappings,
        {read_gpu_backing, write_gpu_backing, descriptors_->gpu ? &descriptors_->gpu->memory
            : const_cast<horizon::GuestBuffers*>(&buffers)},
        address_space->second.view_generation);
    if (const auto* error = std::get_if<graphics::MemoryResult>(&resolved)) {
        ioctl_diagnostic_ = std::string(error->detail);
        diagnostic_ = ioctl_diagnostic_.c_str();
        return {};
    }
    auto& memory = std::get<graphics::GpuAddressSpace>(resolved);
    if (!device.gpu_state) device.gpu_state = std::make_shared<GpuChannelState>();
    auto& state = *device.gpu_state;
    std::unique_ptr<GpuWorker::Admission> admission;
    std::optional<std::vector<std::byte>> command_snapshot;
    if (descriptors_->gpu) {
        std::size_t command_bytes = 0;
        bool capturable = true;
        for (const auto entry : captured) {
            const auto words = graphics::GpfifoChannel::entry_words(entry);
            const auto bytes = static_cast<std::size_t>(words) * 4;
            if (words > graphics::GpfifoChannel::max_entry_words ||
                bytes > GpuWorker::max_command_bytes - command_bytes) {
                capturable = false;
                break;
            }
            command_bytes += bytes;
        }
        // Reserve before allocating/copying. Descriptors and commands consume
        // the same bounded capacity through capture, queuing and execution.
        {
            const DescriptorUnlock unlocked;
            admission = descriptors_->gpu->reserve(captured.size() * sizeof(std::uint64_t) +
                                                   (capturable ? command_bytes : 0));
        }
        if (!admission) return unsupported("GPU submission queue is stopped");
        if (capturable) {
            try {
                command_snapshot.emplace(command_bytes);
                auto remaining = std::span(*command_snapshot);
                for (const auto entry : captured) {
                    const auto bytes = static_cast<std::size_t>(graphics::GpfifoChannel::entry_words(entry)) * 4;
                    if (bytes == 0) continue;
                    if (!memory.read(graphics::GpfifoChannel::entry_address(entry), remaining.first(bytes))) {
                        command_snapshot.reset();
                        break;
                    }
                    remaining = remaining.subspan(bytes);
                }
            } catch (const std::bad_alloc&) {
                command_snapshot.reset();
            }
        }
    }
    EngineDispatch context{state.engines, memory,
        {increment_syncpoint, expired_syncpoint, gpu_ticks, descriptors_->timeline.get(),
         descriptors_->rendering, retain_syncpoint_increment}};
    if (!descriptors_->gpu && (flags & 1u) != 0 && fence_id < descriptors_->syncpoints.size() &&
        !syncpoint_expired(*descriptors_->timeline, fence_id, fence_value)) {
        // Earlier submissions can still own the increments satisfying this
        // fence. Drain their ordered effects before deciding the wait failed.
        const auto coherent = [&] {
            const DescriptorUnlock unlocked;
            return state.engines.make_coherent(memory, context.completion);
        }();
        if (!coherent.ok()) {
            record_channel_failure(*descriptors_, state, device.syncpoint_id,
                std::string("Input GPU fence coherence failed: ") + std::string(coherent.detail));
            ioctl_diagnostic_ = state.stop_diagnostic;
            diagnostic_ = ioctl_diagnostic_.c_str();
            return {};
        }
        if (!syncpoint_expired(*descriptors_->timeline, fence_id, fence_value))
            return unsupported("GPU submission is waiting for an incomplete input fence");
    }
    auto& syncpoint = descriptors_->syncpoints[device.syncpoint_id];
    const auto increment = (flags & 2u) + ((flags & 0x100u) != 0 ? fence_value : 0u);
    syncpoint.reserved += increment;
    // Unsigned syncpoint expiry also depends on reservations across wraparound.
    if (increment != 0) descriptors_->syncpoint_changed.notify_all();
    ++state.submissions;
    if ([] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }())
        armigo::diag::opt_in( "nvdrv_submit table=%p channel=%u space=%u submission=%llu engine=%p mappings=%zu\n",
            static_cast<const void*>(descriptors_.get()), device.fd, device.bound_address_space,
            static_cast<unsigned long long>(state.submissions),
            static_cast<const void*>(&state.engines), memory.mappings().size());
    if (descriptors_->gpu) {
        const bool synchronous = !command_snapshot;
        auto* client = descriptors_.get();
        const auto channel = device.fd;
        const auto syncpoint_id = device.syncpoint_id;
        const auto submission = state.submissions;
        const auto state_owner = device.gpu_state;
        auto rendering = descriptors_->rendering;
        auto job = [client, channel, syncpoint_id, submission, state_owner,
                    rendering = std::move(rendering), memory = std::move(memory),
                    captured = std::move(captured), commands = std::move(command_snapshot),
                    flags, fence_id, fence_value](std::stop_token stop) mutable {
            std::string failure;
            try {
                EngineDispatch dispatch{state_owner->engines, memory,
                    {increment_syncpoint, expired_syncpoint, gpu_ticks, client->timeline.get(),
                     rendering, retain_syncpoint_increment}};
                if ((flags & 1u) && !expired_syncpoint(client->timeline.get(), fence_id, fence_value)) {
                    const auto coherent = state_owner->engines.make_coherent(memory, dispatch.completion);
                    if (!coherent.ok()) failure = std::string("Input GPU fence coherence failed: ") +
                        std::string(coherent.detail);
                }
                SyncpointState snapshot;
                {
                    std::unique_lock lock(client->mutex);
                    if (!state_owner->stop_diagnostic.empty()) return;
                    if (failure.empty() && (flags & 1u) && fence_id < client->syncpoints.size()) {
                        while (!stop.stop_requested() && client->timeline->failure.empty() &&
                               client->timeline->channel_failures[fence_id].empty() &&
                               !syncpoint_expired(*client->timeline, fence_id, fence_value)) {
                            client->syncpoint_changed.wait_for(lock, std::chrono::milliseconds(2));
                            lock.unlock();
                            poll_backend(*client);
                            lock.lock();
                        }
                    }
                    if (stop.stop_requested()) return;
                    if (!client->timeline->failure.empty()) failure = client->timeline->failure;
                    if ((flags & 1u) && fence_id < client->timeline->channel_failures.size() &&
                        !client->timeline->channel_failures[fence_id].empty())
                        failure = client->timeline->channel_failures[fence_id];
                    snapshot = client->syncpoints[syncpoint_id];
                }
                if (failure.empty()) failure = process_submission(*state_owner, memory, dispatch, captured, flags,
                    channel, syncpoint_id, submission, snapshot,
                    commands ? std::optional<std::span<const std::byte>>{*commands} : std::nullopt);
            } catch (const std::exception& error) {
                failure = std::string("GPU worker failed: ") + error.what();
            } catch (...) {
                failure = "GPU worker failed with an unknown exception";
            }
            if (!failure.empty()) {
                const std::lock_guard lock(client->mutex);
                record_channel_failure(*client, *state_owner, syncpoint_id, std::move(failure));
            }
        };
        bool accepted;
        {
            const DescriptorUnlock unlocked;
            accepted = descriptors_->gpu->enqueue(std::move(job), std::move(admission));
            // An uncapturable submission retains live command-fetch semantics,
            // but cannot return while the guest might still reuse its words.
            // Running it in the same FIFO preserves input waits and cancellation.
            if (accepted && synchronous) descriptors_->gpu->drain();
        }
        if (!accepted) {
            syncpoint.reserved -= increment;
            return unsupported("GPU submission queue is stopped");
        }
        if (synchronous) {
            if (!state.stop_diagnostic.empty() || !descriptors_->timeline->failure.empty()) {
                ioctl_diagnostic_ = !state.stop_diagnostic.empty()
                    ? state.stop_diagnostic : descriptors_->timeline->failure;
                diagnostic_ = ioctl_diagnostic_.c_str();
                return {};
            }
            if (descriptors_->timeline->stopped)
                return unsupported("GPU submission queue is stopped");
        }
    } else {
        // The request gate keeps channel state and mappings stable. Release the
        // timeline lock just as the GPU worker does: threaded callbacks may wait
        // for an earlier effect whose publication needs that lock. Snapshot the
        // diagnostic counters before allowing concurrent fence publication.
        const auto snapshot = syncpoint;
        auto failure = [&] {
            const DescriptorUnlock unlocked;
            return process_submission(state, memory, context, captured, flags,
                device.fd, device.syncpoint_id, state.submissions, snapshot);
        }();
        if (!failure.empty()) {
            record_channel_failure(*descriptors_, state, device.syncpoint_id, std::move(failure));
            ioctl_diagnostic_ = state.stop_diagnostic;
            diagnostic_ = ioctl_diagnostic_.c_str();
            return {};
        }
    }
    const auto put_word = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned byte = 0; byte != 4; ++byte)
            arguments[offset + byte] = static_cast<std::byte>((value >> (8 * byte)) & 255);
    };
    put_word(12, 0);
    put_word(16, device.syncpoint_id);
    put_word(20, syncpoint.reserved);
    if (!buffers.write(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::multimedia_ioctl(Device& device, std::uint32_t request,
                                        const horizon::GuestBuffers& buffers) {
    // The request word packs direction (31..30), argument size (29..16), group
    // (15..8) and command (7..0). MapBuffer and UnmapBuffer carry a variable
    // entry list after their fixed part, so they are matched without the size.
    const auto group = (request >> 8) & 0xffu;
    const auto command = request & 0xffu;
    std::vector<std::byte> args;
    const auto load = [&](std::size_t minimum) {
        if (buffers.empty() || buffers[0].size < minimum) return false;
        args.resize(buffers[0].size);
        return buffers.read(0, args);
    };
    const auto word = [&](std::size_t offset) {
        std::uint32_t value = 0;
        for (unsigned byte = 0; byte != 4; ++byte)
            value |= std::uint32_t{std::to_integer<unsigned char>(args[offset + byte])} << (8 * byte);
        return value;
    };
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned byte = 0; byte != 4; ++byte)
            args[offset + byte] = static_cast<std::byte>((value >> (8 * byte)) & 255);
    };
    const auto answer = [&]() {
        if (!buffers.write(0, args)) return driver_reply(nv_result_invalid_address, 0, false);
        return driver_reply(0, 0, false);
    };
    if (request == 0x40044801) {  // SetNvmapFd(s32 fd)
        if (!load(4)) return driver_reply(nv_result_invalid_size, 0, false);
        device.nvmap_fd = word(0);
        device.nvmap_bound = true;
        return driver_reply(0, 0, false);
    }
    if (group == 0 && command == 0x02) {  // GetSyncpoint(u32 index) -> u32 syncpoint id
        if (!load(8)) return driver_reply(nv_result_invalid_size, 0, false);
        put(4, device.syncpoint_id);
        return answer();
    }
    if (group == 0 && command == 0x03) {  // GetWaitbase(u32 index) -> u32 0: no wait bases
        if (!load(8)) return driver_reply(nv_result_invalid_size, 0, false);
        put(4, 0);
        return answer();
    }
    if (group == 0 && command == 0x07) {  // SetSubmitTimeout(u32 milliseconds)
        if (!load(4)) return driver_reply(nv_result_invalid_size, 0, false);
        device.submit_timeout = word(0);
        return driver_reply(0, 0, false);
    }
    if (group == 0 && (command == 0x09 || command == 0x0a)) {
        // MapBuffer / UnmapBuffer({u32 count, u32 reserved, u32 attach} then count
        // {u32 handle, u32 device address}): MapBuffer pins each NVMap handle at a
        // 32-bit device address and reports it; UnmapBuffer releases one pin each.
        if (!load(12)) return driver_reply(nv_result_invalid_size, 0, false);
        const auto count = word(0);
        if (count > (args.size() - 12) / 8) return driver_reply(nv_result_invalid_size, 0, false);
        for (std::uint32_t index = 0; index != count; ++index) {
            const auto at = 12 + std::size_t{index} * 8;
            const auto handle = word(at);
            const auto memory = descriptors_->memory_handles.find(handle);
            if (memory == descriptors_->memory_handles.end() || memory->second.guest_address == 0)
                return driver_reply(nv_result_bad_parameter, 0, false);
            const std::lock_guard pins(descriptors_->host1x_pins_mutex);
            auto pin = descriptors_->host1x_pins.find(handle);
            if (command == 0x09) {
                if (pin == descriptors_->host1x_pins.end()) {
                    const auto size = (std::uint64_t{memory->second.size} + 0xfff) & ~std::uint64_t{0xfff};
                    // First fit among released ranges, then the top of the space.
                    auto& released = descriptors_->host1x_free;
                    auto fit = std::find_if(released.begin(), released.end(),
                                            [&](const auto& range) { return range.second >= size; });
                    std::uint32_t address = 0;
                    if (fit != released.end()) {
                        address = fit->first;
                        const auto remaining = fit->second - static_cast<std::uint32_t>(size);
                        released.erase(fit);
                        if (remaining) released.emplace(address + static_cast<std::uint32_t>(size), remaining);
                    } else {
                        if (size > 0xffffffffull - descriptors_->next_host1x_address)
                            return driver_reply(nv_result_insufficient_memory, 0, false);
                        address = descriptors_->next_host1x_address;
                        descriptors_->next_host1x_address += static_cast<std::uint32_t>(size);
                    }
                    pin = descriptors_->host1x_pins.emplace(
                        handle, DescriptorTable::Host1xPin{address, 0,
                                                           memory->second.size, memory->second.guest_address}).first;
                }
                ++pin->second.references;
                put(at + 4, pin->second.address);
            } else {
                if (pin == descriptors_->host1x_pins.end() || pin->second.references == 0)
                    return driver_reply(nv_result_bad_parameter, 0, false);
                // The device address stays mapped after the last unpin: work
                // already submitted still reads through it, and pinning the
                // handle again reuses it. Freeing the handle releases it.
                --pin->second.references;
                put(at + 4, 0);
            }
        }
        if (command == 0x0a) put(0, 0);
        return answer();
    }
    if (group == 0 && command == 0x01) return multimedia_submit(device, buffers);
    char text[160]{};
    std::snprintf(text, sizeof text, "Driver ioctl %#x on %s is not implemented", request, device.path.c_str());
    ioctl_diagnostic_ = text;
    diagnostic_ = ioctl_diagnostic_.c_str();
    if (reported_ioctls_.insert(device.path + ':' + std::to_string(request)).second)
        armigo::diag::write(armigo::diag::Level::warning,  "Armigo nvdrv: %s.\n", text);
    return driver_reply(nv_result_not_implemented, 0, false);
}

IpcReply DriverSession::multimedia_submit(Device& device, const horizon::GuestBuffers& buffers) {
    // {u32 command buffers, u32 relocations, u32 syncpoint increments, u32 fences}
    // followed by command buffers {s32 handle, u32 offset, s32 words}, relocations
    // (16 bytes each), relocation shifts (u32 each), syncpoint increments
    // {u32 id, u32 increments, 3 reserved words} and fence thresholds (u32 each).
    // The arrays are written back with each increment's fence threshold.
    if (buffers.empty() || buffers[0].size < 16) return driver_reply(nv_result_invalid_size, 0, false);
    std::vector<std::byte> args(buffers[0].size);
    if (!buffers.read(0, args)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto word = [&](std::size_t offset) {
        std::uint32_t value = 0;
        for (unsigned byte = 0; byte != 4; ++byte)
            value |= std::uint32_t{std::to_integer<unsigned char>(args[offset + byte])} << (8 * byte);
        return value;
    };
    const auto put = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned byte = 0; byte != 4; ++byte)
            args[offset + byte] = static_cast<std::byte>((value >> (8 * byte)) & 255);
    };
    const std::uint64_t commands = word(0), relocations = word(4), increments = word(8), fences = word(12);
    const auto commands_at = std::uint64_t{16};
    const auto relocations_at = commands_at + commands * 12;
    const auto shifts_at = relocations_at + relocations * 16;
    const auto increments_at = shifts_at + relocations * 4;
    const auto fences_at = increments_at + increments * 20;
    if (commands > 64 || relocations > 4096 || increments > 64 || fences > 64 ||
        fences_at + fences * 4 > args.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    // A refused submission ends the run, so the host log names why.
    const auto refuse = [&](std::string detail) {
        armigo::diag::write(armigo::diag::Level::error,  "Armigo nvdrv: Submit on %s: %s.\n", device.path.c_str(), detail.c_str());
        ioctl_diagnostic_ = std::move(detail);
        diagnostic_ = ioctl_diagnostic_.c_str();
        return IpcReply{};
    };
    if (!descriptors_->host1x) return refuse("the channel has no engine");
    const auto syncpoint = device.syncpoint_id;
    if (!descriptors_->timeline->channel_failures[syncpoint].empty())
        return refuse(descriptors_->timeline->channel_failures[syncpoint]);
    // Read every command buffer before reserving fences, so a bad request
    // changes nothing.
    std::vector<std::vector<std::uint32_t>> lists;
    for (std::uint64_t index = 0; index != commands; ++index) {
        const auto at = commands_at + index * 12;
        const auto handle = word(at);
        const auto offset = word(at + 4);
        const auto count = word(at + 8);
        const auto memory = descriptors_->memory_handles.find(handle);
        if (memory == descriptors_->memory_handles.end() || memory->second.guest_address == 0 ||
            count > (memory->second.size / 4) || offset > memory->second.size - count * 4ull)
            return driver_reply(nv_result_bad_parameter, 0, false);
        std::vector<std::uint32_t> words(count);
        if (count != 0 && !buffers.read_memory(memory->second.guest_address + offset,
                                               std::as_writable_bytes(std::span(words))))
            return driver_reply(nv_result_invalid_address, 0, false);
        lists.push_back(std::move(words));
    }
    for (std::uint64_t index = 0; index != increments; ++index) {
        const auto at = increments_at + index * 20;
        const auto id = word(at);
        if (id >= descriptors_->syncpoints.size()) return driver_reply(nv_result_bad_parameter, 0, false);
    }
    for (std::uint64_t index = 0; index != increments; ++index) {
        const auto at = increments_at + index * 20;
        auto& state = descriptors_->syncpoints[word(at)];
        state.reserved += word(at + 4);
        if (index < fences) put(fences_at + index * 4, state.reserved);
    }
    if (host1x_tracing()) {
        armigo::diag::opt_in( "host1x submit fd %u lists %zu increments %llu", device.fd, lists.size(),
                     static_cast<unsigned long long>(increments));
        for (std::uint64_t index = 0; index != increments; ++index)
            armigo::diag::opt_in( " [id %u count %u fence %u]", word(increments_at + index * 20),
                         word(increments_at + index * 20 + 4), index < fences ? word(fences_at + index * 4) : 0u);
        armigo::diag::newline();
    }
    for (auto& list : lists)
        if (!descriptors_->host1x->push(device.fd, std::move(list))) return refuse("the channel has no engine");
    if (!buffers.write(0, args)) return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::syncpoint_ioctl(std::uint32_t fd, std::uint32_t request,
                                       const horizon::GuestBuffers& buffers) {
    if (descriptors_->gpu) {
        const DescriptorUnlock unlocked;
        poll_backend(*descriptors_);
    }
    std::size_t required = 0;
    switch (request) {
    case 0xc0080014: case 0xc008001a: required = 8; break;
    case 0x40040015: case 0x4004001f: case 0x40040020: required = 4; break;
    case 0xc0100019: case 0xc010001d: case 0xc010001e: required = 16; break;
    default: return unsupported("GPU syncpoint controller ioctl is not implemented");
    }
    if (buffers.empty() || buffers[0].size < required)
        return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 16> args{};
    if (!buffers.read(0, std::span(args).first(required)))
        return driver_reply(nv_result_invalid_address, 0, false);
    const auto get_word = [&](std::size_t offset) {
        std::uint32_t value = 0;
        for (unsigned byte = 0; byte != 4; ++byte)
            value |= std::uint32_t{std::to_integer<unsigned char>(args[offset + byte])} << (8 * byte);
        return value;
    };
    const auto put_word = [&](std::size_t offset, std::uint32_t value) {
        for (unsigned byte = 0; byte != 4; ++byte)
            args[offset + byte] = static_cast<std::byte>((value >> (8 * byte)) & 255);
    };
    const auto answer = [&](std::uint32_t result) {
        if (!buffers.write(0, std::span<const std::byte>(args).first(required)))
            return driver_reply(nv_result_invalid_address, 0, false);
        return driver_reply(result, 0, false);
    };
    const auto id = get_word(0);
    auto& events = descriptors_->events[fd];
    if (request == 0x4004001f || request == 0x40040020) {
        if (id >= events.size()) return driver_reply(nv_result_invalid_value, 0, false);
        auto& slot = events[id];
        if (slot.pending) return unsupported("GPU event slot still owns an incomplete wait");
        if (request == 0x4004001f) slot = {std::make_shared<horizon::Event>(false), 0, 0, false};
        else slot = {};
        return driver_reply(0, 0, false);
    }
    if (id >= descriptors_->syncpoints.size()) return driver_reply(nv_result_invalid_value, 0, false);
    if (descriptors_->gpu && !descriptors_->timeline->failure.empty()) {
        ioctl_diagnostic_ = descriptors_->timeline->failure;
        diagnostic_ = ioctl_diagnostic_.c_str();
        return {};
    }
    // Preserve raw counter inspection in the synchronous adapter, but reject
    // waits on a failed channel in either mode instead of arming a dead event.
    const bool counter_query = request == 0xc0080014 || request == 0xc008001a;
    if (!descriptors_->timeline->channel_failures[id].empty() &&
        (descriptors_->gpu || !counter_query)) {
        ioctl_diagnostic_ = descriptors_->timeline->channel_failures[id];
        diagnostic_ = ioctl_diagnostic_.c_str();
        return {};
    }
    if (request == 0xc0080014 || request == 0xc008001a) {
        const auto& state = descriptors_->syncpoints[id];
        put_word(4, request == 0xc0080014 ? state.completed : state.reserved);
        return answer(0);
    }
    if (request == 0x40040015) {
        increment_syncpoint(descriptors_->timeline.get(), id);
        return driver_reply(0, 0, false);
    }
    const auto threshold = get_word(4);
    const auto timeout = get_word(8);
    const auto requested_slot = get_word(12);
    if (request != 0xc010001e) put_word(12, 0);
    if (syncpoint_expired(*descriptors_->timeline, id, threshold)) {
        put_word(12, descriptors_->syncpoints[id].completed);
        return answer(0);
    }
    if (timeout == 0) return answer(nv_result_timeout);
    if (request == 0xc0100019)
        return unsupported("Blocking GPU syncpoint wait needs a resumable execution gate");
    if (request == 0xc010001e && requested_slot >= 16)
        return unsupported("Registered GPU event slots above 15 need additional event-id routing");
    std::size_t index = requested_slot;
    if (request == 0xc010001d) {
        index = 0;
        while (index < events.size() && events[index].pending) ++index;
    }
    if (index >= events.size()) return driver_reply(nv_result_invalid_value, 0, false);
    auto& slot = events[index];
    if (slot.pending || (request == 0xc010001e && !slot.event))
        return driver_reply(nv_result_invalid_value, 0, false);
    const auto encoded = request == 0xc010001d
        ? static_cast<std::uint32_t>(index) | ((id & 0xfffu) << 16) | 0x10000000u
        : static_cast<std::uint32_t>(index) | (id << 4);
    put_word(12, encoded);
    if (!buffers.write(0, args)) return driver_reply(nv_result_invalid_address, 0, false);
    if (!slot.event) slot.event = std::make_shared<horizon::Event>(false);
    slot.event->reset();
    slot.syncpoint = id;
    slot.threshold = threshold;
    slot.pending = true;
    return driver_reply(nv_result_timeout, 0, false);
}

IpcReply DriverSession::gpu_channel_set_nvmap(Device& device,
                                              const horizon::GuestBuffers& buffers) {
    if (buffers.empty() || buffers[0].size < 4) return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 4> raw{};
    if (!buffers.read(0, raw)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto fd = static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[0])) |
                    (static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[1])) << 8) |
                    (static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[2])) << 16) |
                    (static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[3])) << 24);
    const auto found = descriptors_->devices.find(fd);
    if (found == descriptors_->devices.end() || found->second.path != nvmap_device)
        return driver_reply(nv_result_bad_parameter, 0, false);
    device.nvmap_fd = fd;
    device.nvmap_bound = true;
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::address_space_bind_channel(Device& device,
                                                  const horizon::GuestBuffers& buffers) {
    if (descriptors_->gpu) {
        const DescriptorUnlock unlocked;
        descriptors_->gpu->drain();
    }
    if (buffers.empty() || buffers[0].size < 4) return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 4> raw{};
    if (!buffers.read(0, raw)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto channel = static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[0])) |
                         (static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[1])) << 8) |
                         (static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[2])) << 16) |
                         (static_cast<std::uint32_t>(std::to_integer<unsigned char>(raw[3])) << 24);
    const auto found = descriptors_->devices.find(channel);
    if (found == descriptors_->devices.end() || found->second.path != nvgpu_channel_device)
        return driver_reply(nv_result_bad_parameter, 0, false);
    if (found->second.bound_address_space != 0) {
        const auto previous = descriptors_->devices.find(found->second.bound_address_space);
        if (previous != descriptors_->devices.end() && previous->second.bound_channel == channel)
            previous->second.bound_channel = 0;
    }
    device.bound_channel = channel;
    found->second.bound_address_space = device.fd;
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::gpu_characteristics(const horizon::GuestBuffers& buffers) {
    // GetCharacteristics passes its argument structure in the first buffer: a
    // 64-bit output size and a 64-bit guest address. The direction is read/write,
    // so the argument buffer also appears in the receive list, and the description
    // is delivered through the receive buffer that is large enough for it.
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 16> header{};
    if (!buffers.read(0, header)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto size = read_u64(header, 0);
    if (size < sizeof(GpuCharacteristics)) return driver_reply(nv_result_invalid_size, 0, false);
    // The argument buffer is listed in the send and receive lists alike, so the
    // description buffer is the separate receive buffer large enough to hold it.
    std::size_t description = buffers.size();
    for (std::size_t index = 1; index < buffers.size(); ++index) {
        if (buffers[index].address == buffers[0].address) continue;
        if (buffers[index].size < sizeof(GpuCharacteristics)) continue;
        description = index;
        break;
    }
    if (description == buffers.size()) return driver_reply(nv_result_invalid_size, 0, false);
    const GpuCharacteristics characteristics = tegra_x1_gpu_characteristics();
    std::array<std::byte, sizeof(GpuCharacteristics)> contents{};
    std::memcpy(contents.data(), &characteristics, sizeof(GpuCharacteristics));
    if (!buffers.write(description, contents))
        return driver_reply(nv_result_invalid_address, 0, false);
    write_u64(header, 0, sizeof(GpuCharacteristics));
    if (!buffers.write(0, header)) return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::gpu_tpc_masks(const horizon::GuestBuffers& buffers) {
    // GetTpcMasks passes a header {u32 mask buffer size, u32 reserved, u64 mask
    // buffer address} and receives the cluster mask in that buffer and in the
    // argument word the driver reads back.
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    std::array<std::byte, 24> arguments{};
    if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto mask_size = static_cast<std::uint32_t>(read_u64(arguments, 0));
    const auto mask_address = read_u64(arguments, 8);
    write_u64(arguments, 16, nvgpu_tpc_mask);
    if (mask_size != 0) {
        std::size_t target = buffers.size();
        for (std::size_t index = 1; index < buffers.size(); ++index) {
            if (buffers[index].address == buffers[0].address) continue;
            if (buffers[index].address == mask_address) {
                target = index;
                break;
            }
            if (target == buffers.size() && buffers[index].size >= 4) target = index;
        }
        if (target == buffers.size() || buffers[target].size < 4)
            return driver_reply(nv_result_invalid_address, 0, false);
        std::array<std::byte, 4> mask{};
        for (unsigned index = 0; index != 4; ++index)
            mask[index] = static_cast<std::byte>((nvgpu_tpc_mask >> (8 * index)) & 255);
        if (!buffers.write(target, mask)) return driver_reply(nv_result_invalid_address, 0, false);
    }
    if (!buffers.write(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::address_space_initialize(Device& device,
                                               const horizon::GuestBuffers& buffers) {
    // Initialize binds the address space the guest selected: {u32 flags, u32
    // reserved, u32 big page size, u32 reserved, u64 range start, u64 range end,
    // u64 range split}. The console keeps the guest's own range and page-size
    // selection, so the arguments are written back unchanged and the driver
    // continues with the range it asked for. An unspecified range falls back to the
    // console defaults derived from the address-space page size.
    std::array<std::byte, 0x28> arguments{};
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    if (buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    device.big_page_size = static_cast<std::uint32_t>(read_u64(arguments, 8));
    device.va_range_start = read_u64(arguments, 16);
    device.va_range_end = read_u64(arguments, 24);
    device.va_range_split = read_u64(arguments, 32);
    if (device.big_page_size == 0) device.big_page_size = 0x20000;
    if (device.va_range_start == 0)
        device.va_range_start = static_cast<std::uint64_t>(device.big_page_size) << 10;
    if (device.va_range_split == 0) device.va_range_split = 1ull << 34;
    if (device.va_range_end == 0) device.va_range_end = 1ull << 37;
    device.small_pages.configure(device.va_range_start,
                                device.va_range_split - device.va_range_start,
                                nvgpu_small_page_size);
    device.big_pages.configure(device.va_range_split,
                               device.va_range_end - device.va_range_split,
                               device.big_page_size);
    device.address_space_bound = true;
    if (!buffers.write(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::memory_create(const horizon::GuestBuffers& buffers) {
    // Create reserves one memory handle for the requested size and answers with its
    // id. The console allocates whole pages, and the driver reads the id back from
    // the same argument buffer.
    std::array<std::byte, 8> arguments{};
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    if (buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    const auto size = static_cast<std::uint32_t>(read_u64(arguments, 0));
    if (size == 0) return driver_reply(nv_result_invalid_size, 0, false);
    if (descriptors_->next_memory_handle + 1 > std::numeric_limits<std::uint32_t>::max() / 4)
        return driver_reply(nv_result_insufficient_memory, 0, false);
    ++descriptors_->next_memory_handle;
    const auto id = descriptors_->next_memory_handle * 4;
    const auto aligned = (static_cast<std::uint64_t>(size) + nvmap_page_size - 1) &
                         ~static_cast<std::uint64_t>(nvmap_page_size - 1);
    if (aligned > std::numeric_limits<std::size_t>::max())
        return driver_reply(nv_result_insufficient_memory, 0, false);
    NvMapHandle handle{};
    handle.size = static_cast<std::uint32_t>(aligned);
    handle.memory.resize(static_cast<std::size_t>(aligned));
    descriptors_->memory_handles[id] = std::move(handle);
    write_u64(arguments, 0, (static_cast<std::uint64_t>(id) << 32) | size);
    if (!buffers.write(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::memory_get_id(const horizon::GuestBuffers& buffers) {
    std::array<std::byte, 8> arguments{};
    if (buffers.empty() || buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    std::uint32_t handle = 0;
    for (unsigned byte = 0; byte != 4; ++byte)
        handle |= std::uint32_t{std::to_integer<unsigned char>(arguments[4 + byte])}
                  << (byte * 8);
    if (handle == 0)
        return driver_reply(nv_result_invalid_value, 0, false);
    const auto found = descriptors_->memory_handles.find(handle);
    if (found == descriptors_->memory_handles.end())
        return driver_reply(nv_result_access_denied, 0, false);
    for (unsigned byte = 0; byte != 4; ++byte)
        arguments[byte] = static_cast<std::byte>((handle >> (byte * 8)) & 255);
    if (!buffers.write(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    found->second.exported = true;
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::memory_allocate(const horizon::GuestBuffers& buffers) {
    // Alloc answers with the guest address the driver reads and writes the handle
    // through. The driver names that address, so the runtime maps the handle's
    // memory there and the guest and the GPU share the same pages.
    std::array<std::byte, 0x20> arguments{};
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    if (buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto id = static_cast<std::uint32_t>(read_u64(arguments, 0));
    const auto found = descriptors_->memory_handles.find(id);
    if (found == descriptors_->memory_handles.end()) return driver_reply(nv_result_invalid_address, 0, false);
    const auto requested_alignment = static_cast<std::uint32_t>(read_u64(arguments, 12));
    if (requested_alignment != 0 && (requested_alignment & (requested_alignment - 1)) != 0)
        return driver_reply(nv_result_invalid_value, 0, false);
    const auto alignment = std::max(requested_alignment, nvmap_page_size);
    for (unsigned byte = 0; byte != 4; ++byte)
        arguments[12 + byte] = static_cast<std::byte>((alignment >> (byte * 8)) & 255);
    const auto address = read_u64(arguments, 24);
    if (found->second.guest_address != 0) {
        if (address == 0) {
            auto flags = static_cast<std::uint32_t>(read_u64(arguments, 8));
            flags |= 2;
            for (unsigned byte = 0; byte != 4; ++byte)
                arguments[8 + byte] = static_cast<std::byte>((flags >> (byte * 8)) & 255);
        }
        if (!buffers.write(0, arguments))
            return driver_reply(nv_result_invalid_address, 0, false);
        return driver_reply(0, 0, false);
    }
    if (address == 0) return driver_reply(nv_result_invalid_address, 0, false);
    std::shared_ptr<RetainedNvMapMapping> retained;
    if (mapper_.lifetime) {
        try {
            retained = std::make_shared<RetainedNvMapMapping>(mapper_, address,
                                                              found->second.size);
        } catch (const std::bad_alloc&) {
            return driver_reply(nv_result_insufficient_memory, 0, false);
        }
    }
    if (!mapper_.map || !mapper_.map(mapper_.context, address, found->second.size)) {
        char text[160]{};
        std::snprintf(text, sizeof text,
                      "Driver memory allocate needs %u guest bytes at %#llx",
                      found->second.size, static_cast<unsigned long long>(address));
        ioctl_diagnostic_ = text;
        diagnostic_ = ioctl_diagnostic_.c_str();
        return driver_reply(nv_result_not_implemented, 0, false);
    }
    found->second.alignment = alignment;
    found->second.guest_address = address;
    found->second.allocation_flags = static_cast<std::uint32_t>(read_u64(arguments, 8));
    if (retained) {
        retained->active = true;
        found->second.retained_mapping = std::move(retained);
    }
    write_u64(arguments, 24, address);
    if (!buffers.write(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::memory_param(const horizon::GuestBuffers& buffers) {
    // Param reports one property of a handle. The driver asks for the size, the
    // alignment, the heap and the compression state; a property the console does
    // not answer stays explicit.
    std::array<std::byte, 0x0c> arguments{};
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    if (buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto id = static_cast<std::uint32_t>(read_u64(arguments, 0));
    const auto parameter = static_cast<std::uint32_t>(read_u64(arguments, 4));
    const auto found = descriptors_->memory_handles.find(id);
    if (found == descriptors_->memory_handles.end()) return driver_reply(nv_result_invalid_address, 0, false);
    std::uint32_t value = 0;
    switch (parameter) {
    case nvmap_param_size: value = found->second.size; break;
    case nvmap_param_align:
        value = found->second.alignment != 0 ? found->second.alignment : nvmap_page_size;
        break;
    case nvmap_param_heap: value = nvmap_heap_mask; break;
    case nvmap_param_kind: value = 0; break;
    case nvmap_param_compression: value = 0; break;
    default: return driver_reply(nv_result_invalid_size, 0, false);
    }
    for (unsigned byte = 0; byte != 4; ++byte)
        arguments[8 + byte] = static_cast<std::byte>((value >> (byte * 8)) & 255);
    if (!buffers.write(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::memory_free(const horizon::GuestBuffers& buffers) {
    std::array<std::byte, 0x18> arguments{};
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    if (buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto id = static_cast<std::uint32_t>(read_u64(arguments, 0));
    const auto found = descriptors_->memory_handles.find(id);
    // Free treats a missing handle as a successful no-op and preserves the request.
    if (found == descriptors_->memory_handles.end()) return driver_reply(0, 0, false);
    std::array<std::shared_ptr<void>, 1> released{};
    std::size_t released_count = 0;
    const auto& owner = found->second.retained_mapping;
    bool retained_reference = false;
    if (owner) {
        const bool has_other_handle = std::any_of(
            descriptors_->memory_handles.begin(), descriptors_->memory_handles.end(),
            [&](const auto& entry) {
                return entry.first != id && same_owner(entry.second.retained_mapping, owner);
            });
        const bool has_mapping = std::any_of(
            descriptors_->devices.begin(), descriptors_->devices.end(),
            [&](const auto& entry) {
                return std::any_of(entry.second.mappings.begin(), entry.second.mappings.end(),
                    [&](const AddressMapping& mapping) {
                        return same_owner(mapping.retained_mapping, owner);
                    });
            });
        if (!has_other_handle && !has_mapping) released[released_count++] = owner;
        retained_reference = has_other_handle || has_mapping;
    }
    const auto imported = descriptors_->retained_exports.find(id);
    if (imported != descriptors_->retained_exports.end() && !imported->second.owner.expired())
        retained_reference = true;
    if (!notify_memory_changed({}, {released.data(), released_count}, buffers)) return {};
    // Supported Alloc requests use caller-supplied storage. Firmware's internally
    // allocated storage state (which leaves these fields untouched) is not reached
    // by the unsupported zero-address allocation path.
    write_u64(arguments, 8, retained_reference ? 0 : found->second.guest_address);
    write_u64(arguments, 16, found->second.size |
        (std::uint64_t{found->second.allocation_flags & 1u} << 32));
    if (!buffers.write(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    // Owned mappings release through their lease, which may outlive the handle.
    if (!found->second.retained_mapping && found->second.guest_address != 0 && mapper_.unmap) {
        mapper_.unmap(mapper_.context, found->second.guest_address, found->second.size);
    }
    if ([] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }()) {
        armigo::diag::opt_in( "nvdrv_free table=%p handle=%u guest=%#llx size=%#x owner=%p released=%zu\n",
            static_cast<const void*>(descriptors_.get()), id,
            static_cast<unsigned long long>(found->second.guest_address),
            found->second.size, found->second.retained_mapping.get(), released_count);
        for (const auto& [fd, device] : descriptors_->devices)
            for (const auto& mapping : device.mappings)
                if (same_owner(mapping.retained_mapping, owner))
                    trace_mapping("free-live", descriptors_.get(), device, mapping);
    }
    {
        // A freed handle loses its engine device address, which a later pin
        // can reuse; the released range merges with released neighbours.
        const std::lock_guard pins(descriptors_->host1x_pins_mutex);
        if (const auto pin = descriptors_->host1x_pins.find(found->first); pin != descriptors_->host1x_pins.end()) {
            auto& released = descriptors_->host1x_free;
            auto address = pin->second.address;
            auto bytes = static_cast<std::uint32_t>((std::uint64_t{pin->second.size} + 0xfff) & ~std::uint64_t{0xfff});
            if (auto next = released.find(address + bytes); next != released.end()) {
                bytes += next->second;
                released.erase(next);
            }
            if (auto previous = released.lower_bound(address); previous != released.begin()) {
                --previous;
                if (previous->first + previous->second == address) {
                    address = previous->first;
                    bytes += previous->second;
                    released.erase(previous);
                }
            }
            released.emplace(address, bytes);
            descriptors_->host1x_pins.erase(pin);
        }
    }
    descriptors_->memory_handles.erase(found);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::address_space_allocate(Device& device,
                                              const horizon::GuestBuffers& buffers) {
    // AllocSpace reserves a run of GPU virtual addresses in the window that matches
    // the requested page size and answers with the bound address. The driver maps
    // its buffers into the returned range afterwards, so an address it received is
    // kept until the guest frees it.
    std::array<std::byte, 0x18> arguments{};
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    if (buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    if (!device.address_space_bound) return driver_reply(nv_result_invalid_size, 0, false);
    const auto pages = static_cast<std::uint64_t>(read_u64(arguments, 0) & 0xffffffffu);
    const auto page_size = static_cast<std::uint64_t>(read_u64(arguments, 0) >> 32);
    const auto flags = static_cast<std::uint32_t>(read_u64(arguments, 8));
    const auto offset = read_u64(arguments, 16);
    AddressWindow* window = nullptr;
    if (page_size == nvgpu_small_page_size) window = &device.small_pages;
    else if (page_size == device.big_page_size) window = &device.big_pages;
    if (window == nullptr || pages == 0) return driver_reply(nv_result_invalid_size, 0, false);
    const bool sparse = (flags & 2u) != 0;
    if (sparse && page_size != device.big_page_size)
        return unsupported("Sparse GPU reservations require the configured big-page size");
    std::optional<std::uint64_t> bound;
    if ((flags & nvgpu_as_flag_fixed) != 0) {
        if (!window->allocate_at(offset, pages))
            return driver_reply(nv_result_invalid_address, 0, false);
        bound = offset;
    } else {
        bound = window->allocate(pages, offset == 0 ? page_size : offset);
        if (!bound) return driver_reply(nv_result_insufficient_memory, 0, false);
    }
    if (sparse) device.sparse_reservations[*bound] = pages * page_size;
    write_u64(arguments, 16, *bound);
    if (!buffers.write(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::address_space_remap(Device& device, std::uint32_t request,
                                           const horizon::GuestBuffers& buffers) {
    if (descriptors_->gpu) {
        const DescriptorUnlock unlocked;
        descriptors_->gpu->drain();
    }
    // Horizon validates the complete batch before publishing page-table changes.
    if (!device.address_space_bound)
        return driver_reply(nv_result_invalid_value, 0, false);
    const std::size_t bytes = (request >> 16) & 0x3fff;
    if (bytes < 20 || bytes % 20 != 0)
        return driver_reply(nv_result_unrecognized_error, 0, false);
    if (buffers.empty() || buffers[0].size < bytes)
        return unsupported("GPU remap argument descriptor is shorter than the encoded batch");
    try {
        std::vector<std::byte> arguments(bytes);
        if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
        const auto word = [&](std::size_t offset) {
            std::uint32_t value = 0;
            for (unsigned index = 0; index < 4; ++index)
                value |= std::uint32_t{std::to_integer<unsigned char>(arguments[offset + index])} << (index * 8);
            return value;
        };
        if ((word(0) & 0x1000u) != 0 || word(4) == 0)
            return unsupported("Small-page and sparse-restoration remaps are not implemented");
        const std::uint64_t page = device.big_page_size;
        const std::uint64_t first_address = std::uint64_t{word(12)} * page;
        auto reservation = device.sparse_reservations.upper_bound(first_address);
        if (reservation == device.sparse_reservations.begin())
            return driver_reply(nv_result_unrecognized_error, 0, false);
        --reservation;
        std::vector<AddressMapping> replacements;
        for (std::size_t offset = 0; offset < bytes; offset += 20) {
            const auto flags = static_cast<std::uint16_t>(word(offset));
            const auto requested_kind = static_cast<std::uint16_t>(word(offset) >> 16);
            const auto handle = word(offset + 4);
            if ((flags & 0x1000u) != 0 || handle == 0)
                return unsupported("Small-page and sparse-restoration remaps are not implemented");
            const auto found = descriptors_->memory_handles.find(handle);
            if (found == descriptors_->memory_handles.end())
                return driver_reply(nv_result_unrecognized_error, 0, false);
            const std::uint64_t source = std::uint64_t{word(offset + 8)} * page;
            const std::uint64_t destination = std::uint64_t{word(offset + 12)} * page;
            const std::uint64_t size = std::uint64_t{word(offset + 16)} * page;
            if (size == 0 || source >= found->second.size || size > found->second.size - source)
                return driver_reply(nv_result_unrecognized_error, 0, false);
            // The host has no compression-tag allocation. Use the firmware's
            // uncompressed fallback for the supported compression-capable kinds.
            std::uint16_t kind = requested_kind;
            switch (kind) {
            case 0: case 1: case 0x46: case 0x7b: case 0xc3: case 0xfe: break;
            case 7: kind = 1; break;
            case 0x51: kind = 0x46; break;
            case 0x86: kind = 0x7b; break;
            case 0xce: kind = 0xc3; break;
            case 0xdb: case 0xe9: case 0xf5: kind = 0xfe; break;
            default: return unsupported("GPU remap kind is outside the implemented subset");
            }
            if (kind != requested_kind) {
                arguments[offset + 2] = static_cast<std::byte>(kind & 0xff);
                arguments[offset + 3] = static_cast<std::byte>(kind >> 8);
                // Firmware can expose normalization even if a later record or
                // destination check fails; only mapping publication is atomic.
                if (!buffers.write(0, arguments))
                    return driver_reply(nv_result_invalid_address, 0, false);
            }
            if (destination < reservation->first || destination - reservation->first >= reservation->second ||
                size > reservation->second - (destination - reservation->first))
                return driver_reply(nv_result_unrecognized_error, 0, false);
            if (found->second.guest_address == 0)
                return driver_reply(nv_result_unrecognized_error, 0, false);
            if (!found->second.retained_mapping)
                return unsupported("GPU remap requires owned guest backing");
            replacements.push_back({.offset = destination, .size = size,
                .memory_handle = handle, .page_size = device.big_page_size,
                .buffer_offset = source, .kind = kind, .flags = flags,
                .guest_address = found->second.guest_address,
                .backing_size = found->second.size,
                .retained_mapping = found->second.retained_mapping,
                .sparse_remap = true});
        }
        auto updated = device.mappings;
        for (const auto& replacement : replacements) {
            std::vector<AddressMapping> next;
            for (auto existing : updated) {
                const auto begin = std::max(existing.offset, replacement.offset);
                const auto end = std::min(existing.offset + existing.size, replacement.offset + replacement.size);
                if (begin >= end) {
                    next.push_back(existing);
                    continue;
                }
                if (existing.offset < begin) {
                    auto left = existing;
                    left.size = begin - existing.offset;
                    next.push_back(left);
                }
                if (end < existing.offset + existing.size) {
                    existing.buffer_offset += end - existing.offset;
                    existing.size = existing.offset + existing.size - end;
                    existing.offset = end;
                    next.push_back(existing);
                }
            }
            next.push_back(replacement);
            updated = std::move(next);
        }
        std::vector<graphics::Mapping> removed;
        std::vector<std::shared_ptr<void>> released;
        removed.reserve(device.mappings.size());
        released.reserve(device.mappings.size());
        for (const auto& existing : device.mappings) {
            const bool replaced = std::any_of(
                replacements.begin(), replacements.end(),
                [&](const AddressMapping& replacement) {
                    return existing.offset < replacement.offset + replacement.size &&
                           replacement.offset < existing.offset + existing.size;
                });
            if (!replaced) continue;
            removed.push_back(render_mapping(existing));
            const auto& owner = existing.retained_mapping;
            if (!owner) continue;
            const bool already_released = std::any_of(
                released.begin(), released.end(), [&](const auto& candidate) {
                    return same_owner(candidate, owner);
                });
            if (already_released) continue;
            const bool has_handle = std::any_of(
                descriptors_->memory_handles.begin(), descriptors_->memory_handles.end(),
                [&](const auto& entry) {
                    return same_owner(entry.second.retained_mapping, owner);
                });
            const bool has_updated_mapping = std::any_of(
                updated.begin(), updated.end(), [&](const AddressMapping& mapping) {
                    return same_owner(mapping.retained_mapping, owner);
                });
            const bool has_other_mapping = std::any_of(
                descriptors_->devices.begin(), descriptors_->devices.end(),
                [&](const auto& entry) {
                    if (&entry.second == &device) return false;
                    return std::any_of(entry.second.mappings.begin(), entry.second.mappings.end(),
                        [&](const AddressMapping& mapping) {
                            return same_owner(mapping.retained_mapping, owner);
                        });
                });
            if (!has_handle && !has_updated_mapping && !has_other_mapping)
                released.push_back(owner);
        }
        // Copyout precedes publication so callback failure cannot expose a
        // partially installed host mapping batch.
        if (!buffers.write(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
        if (!notify_memory_changed(removed, released, buffers)) return {};
        for (const auto& mapping : device.mappings)
            trace_mapping("remap-before", descriptors_.get(), device, mapping);
        device.mappings = std::move(updated);
        device.view_generation = 0;
        for (const auto& mapping : device.mappings)
            trace_mapping("remap-after", descriptors_.get(), device, mapping);
        return driver_reply(0, 0, false);
    } catch (const std::bad_alloc&) {
        return driver_reply(nv_result_unrecognized_error, 0, false);
    }
}

IpcReply DriverSession::address_space_map(Device& device,
                                         const horizon::GuestBuffers& buffers) {
    if (descriptors_->gpu) {
        const DescriptorUnlock unlocked;
        descriptors_->gpu->drain();
    }
    // MapBufferEx maps a memory handle or changes an existing mapping's kind.
    // Ordinary maps use the offset word as a fixed address or an alignment,
    // then return the bound address in the same word.
    std::array<std::byte, 0x28> arguments{};
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    if (buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    if (!device.address_space_bound) return driver_reply(nv_result_invalid_size, 0, false);
    const auto flags = static_cast<std::uint32_t>(read_u64(arguments, 0));
    const auto kind = static_cast<std::uint32_t>(read_u64(arguments, 4));
    const auto memory_handle = static_cast<std::uint32_t>(read_u64(arguments, 8));
    const auto page_size = static_cast<std::uint32_t>(read_u64(arguments, 12));
    const auto buffer_offset = read_u64(arguments, 16);
    const auto mapping_size = read_u64(arguments, 24);
    const auto offset = read_u64(arguments, 32);
    if ([] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }()) {
        armigo::diag::opt_in(
            "nvdrv_map_request table=%p space=%u handle=%u flags=%#x page=%#x "
            "offset=%#llx size=%#llx gpu=%#llx\n",
            static_cast<const void*>(descriptors_.get()), device.fd, memory_handle,
            flags, page_size, static_cast<unsigned long long>(buffer_offset),
            static_cast<unsigned long long>(mapping_size), static_cast<unsigned long long>(offset));
    }
    if ((flags & nvgpu_as_flag_remap_subrange) != 0) {
        // RemapSubRange selects an existing mapping, independently of the
        // supplied handle, page size, and fixed-offset flag.
        const auto mapping = std::find_if(device.mappings.begin(), device.mappings.end(),
            [offset](const AddressMapping& candidate) {
                return !candidate.sparse_remap && candidate.offset == offset;
            });
        if (mapping == device.mappings.end())
            return driver_reply(nv_result_unrecognized_error, 0, false);
        const auto page = mapping->page_size;
        if (page == 0 || buffer_offset % page != 0 || mapping_size % page != 0 ||
            buffer_offset >= mapping->size || mapping_size > mapping->size - buffer_offset)
            return driver_reply(nv_result_unrecognized_error, 0, false);
        if (mapping_size != 0 && (buffer_offset != 0 || mapping_size != mapping->size))
            return unsupported("GPU kind remap currently requires the whole mapping");
        // Kind does not alter byte translation or backing lifetime. Preserve
        // the mapping object and its VA reservation, and publish after copyout.
        if (!buffers.write(0, arguments))
            return driver_reply(nv_result_invalid_address, 0, false);
        // A valid zero-sized span is a firmware no-op, including its kind.
        if (mapping_size != 0) {
            mapping->kind = kind;
            trace_mapping("kind-remap", descriptors_.get(), device, *mapping);
        }
        return driver_reply(0, 0, false);
    }
    const auto found = descriptors_->memory_handles.find(memory_handle);
    if (found == descriptors_->memory_handles.end()) return driver_reply(nv_result_invalid_address, 0, false);
    if (found->second.guest_address == 0)
        return driver_reply(nv_result_unrecognized_error, 0, false);
    // The allocation's original mapper owns its pages. A later request cannot
    // promote a borrowed allocation or substitute a different deallocator.
    if (!found->second.retained_mapping)
        return unsupported("GPU mapping requires owned guest backing");
    // A big-page mapping covers whole pages, so a smaller request rounds up to one.
    const auto requested_size = mapping_size != 0 ? mapping_size : found->second.size;
    const auto page = page_size != 0 ? page_size : nvgpu_small_page_size;
    const auto alignment = static_cast<std::uint64_t>(std::max(found->second.alignment, nvmap_page_size));
    const auto padded_backing = (static_cast<std::uint64_t>(found->second.size) + alignment - 1) / alignment * alignment;
    if (buffer_offset > padded_backing || requested_size > padded_backing - buffer_offset)
        return driver_reply(nv_result_invalid_value, 0, false);
    if (buffer_offset > found->second.size) {
        ioctl_diagnostic_ = "GPU mapping starts in allocation padding without modeled guest backing";
        diagnostic_ = ioctl_diagnostic_.c_str();
        return driver_reply(nv_result_not_implemented, 0, false);
    }
    const auto size = (requested_size + page - 1) / page * page;
    AddressWindow* window = nullptr;
    if (page == nvgpu_small_page_size) window = &device.small_pages;
    else if (page == device.big_page_size) window = &device.big_pages;
    if (window == nullptr || size == 0 || size % page != 0 || buffer_offset % page != 0) {
        char text[180]{};
        std::snprintf(text, sizeof text,
                      "Driver mapping rejected: handle %u flags %#x page %#x size %#llx offset %#llx",
                      memory_handle, flags, page, (unsigned long long)size,
                      (unsigned long long)offset);
        ioctl_diagnostic_ = text;
        diagnostic_ = ioctl_diagnostic_.c_str();
        return driver_reply(nv_result_invalid_size, 0, false);
    }
    std::optional<std::uint64_t> bound;
    try {
        if (device.mappings.size() == device.mappings.capacity())
            device.mappings.reserve(std::max<std::size_t>(1, device.mappings.capacity() * 2));
    } catch (const std::bad_alloc&) {
        return driver_reply(nv_result_insufficient_memory, 0, false);
    }
    if ((flags & nvgpu_as_flag_fixed) != 0) {
        // A fixed mapping names the address directly: the driver reserves GPU
        // virtual addresses with AllocSpace and then maps memory into them, so the
        // range must lie in the GPU address space and must not already hold a
        // mapping.
        if (size > nvgpu_address_space_limit || offset > nvgpu_address_space_limit - size ||
            offset % page != 0)
            return driver_reply(nv_result_invalid_address, 0, false);
        for (const auto& mapping : device.mappings) {
            if (mapping.offset < offset + size && offset < mapping.offset + mapping.size)
                return driver_reply(nv_result_invalid_address, 0, false);
        }
        bound = offset;
    } else {
        bound = window->allocate(size / page, offset == 0 ? page : offset);
        if (!bound) return driver_reply(nv_result_insufficient_memory, 0, false);
    }
    write_u64(arguments, 32, *bound);
    if (!buffers.write(0, arguments)) {
        if ((flags & nvgpu_as_flag_fixed) == 0) window->release(*bound, size / page);
        return driver_reply(nv_result_invalid_address, 0, false);
    }
    device.view_generation = 0;
    device.mappings.push_back({.offset = *bound,
                               .size = size,
                               .memory_handle = memory_handle,
                               .page_size = page,
                               .buffer_offset = buffer_offset,
                               .kind = kind,
                               .guest_address = found->second.guest_address,
                               .backing_size = found->second.size,
                               .retained_mapping = found->second.retained_mapping,
                               .automatically_allocated = (flags & nvgpu_as_flag_fixed) == 0});
    trace_mapping("map", descriptors_.get(), device, device.mappings.back());
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::address_space_unmap(Device& device,
                                           const horizon::GuestBuffers& buffers) {
    if (descriptors_->gpu) {
        const DescriptorUnlock unlocked;
        descriptors_->gpu->drain();
    }
    if (!device.address_space_bound)
        return driver_reply(nv_result_invalid_value, 0, false);
    std::array<std::byte, 8> arguments{};
    if (buffers.empty() || buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!buffers.read(0, arguments)) return driver_reply(nv_result_invalid_address, 0, false);
    const auto offset = read_u64(arguments, 0);
    const auto found = std::find_if(device.mappings.begin(), device.mappings.end(),
        [offset](const AddressMapping& mapping) {
            return !mapping.sparse_remap && offset >= mapping.offset &&
                   offset - mapping.offset < mapping.size;
        });
    if (found == device.mappings.end()) return driver_reply(0, 0, false);
    std::array<graphics::Mapping, 1> removed{render_mapping(*found)};
    std::array<std::shared_ptr<void>, 1> released{};
    std::size_t released_count = 0;
    const auto& owner = found->retained_mapping;
    if (owner) {
        const bool has_handle = std::any_of(
            descriptors_->memory_handles.begin(), descriptors_->memory_handles.end(),
            [&](const auto& entry) {
                return same_owner(entry.second.retained_mapping, owner);
            });
        const bool has_other_mapping = std::any_of(
            descriptors_->devices.begin(), descriptors_->devices.end(),
            [&](const auto& entry) {
                return std::any_of(entry.second.mappings.begin(), entry.second.mappings.end(),
                    [&](const AddressMapping& mapping) {
                        return &mapping != &*found &&
                               same_owner(mapping.retained_mapping, owner);
                    });
            });
        if (!has_handle && !has_other_mapping) released[released_count++] = owner;
    }
    if (!notify_memory_changed(removed, {released.data(), released_count}, buffers)) return {};
    // Fixed mappings borrow AllocSpace's reservation. Sparse Remap updates the
    // reservation's page table separately and does not create an UnmapBuffer object.
    if (found->automatically_allocated) {
        auto& window = found->page_size == nvgpu_small_page_size
            ? device.small_pages : device.big_pages;
        window.release(found->offset, found->size / found->page_size);
    }
    trace_mapping("unmap", descriptors_.get(), device, *found);
    device.mappings.erase(found);
    device.view_generation = 0;
    return driver_reply(0, 0, false);
}

IpcReply DriverSession::address_space_va_regions(Device& device,
                                                const horizon::GuestBuffers& buffers) {
    // GetVaRegions reports the bound range as two windows: small pages below the
    // split and big pages above it. Each region is {u64 offset, u32 page size, u32
    // padding, u64 pages}, and the argument buffer starts with {u64 unused, u32
    // buffer size, u32 padding}.
    std::array<std::byte, 0x40> arguments{};
    if (buffers.empty()) return driver_reply(nv_result_invalid_size, 0, false);
    if (buffers[0].size < arguments.size())
        return driver_reply(nv_result_invalid_size, 0, false);
    if (!device.address_space_bound) return driver_reply(nv_result_invalid_size, 0, false);
    const auto small_pages = (device.va_range_split - device.va_range_start) /
                             nvgpu_small_page_size;
    const auto big_pages = (device.va_range_end - device.va_range_split) /
                           device.big_page_size;
    write_u64(arguments, 0x08, nvgpu_va_region_size * 2);
    write_u64(arguments, 0x10, device.va_range_start);
    write_u64(arguments, 0x18, nvgpu_small_page_size);
    write_u64(arguments, 0x20, small_pages);
    write_u64(arguments, 0x28, device.va_range_split);
    write_u64(arguments, 0x30, device.big_page_size);
    write_u64(arguments, 0x38, big_pages);
    if (!buffers.write(0, arguments))
        return driver_reply(nv_result_invalid_address, 0, false);
    return driver_reply(0, 0, false);
}

void DriverSession::set_memory_mapper(horizon::GuestMemoryMapper mapper) noexcept {
    const std::lock_guard lock(descriptors_->mutex);
    mapper_ = mapper;
}

std::optional<services::GraphicsBufferBacking>
DriverSession::retain_exported_memory(std::uint32_t export_id) {
    return nvdrv::retain_exported_memory(descriptors_, export_id);
}

runtime::GraphicsBackingAccess DriverSession::graphics_backing_access(
    std::shared_ptr<runtime::GuestMemory> guest_memory) const {
    const auto descriptors = descriptors_;
    runtime::GraphicsBackingAccess access;
    access.lookup = [descriptors](std::uint32_t export_id) {
        return nvdrv::retain_exported_memory(descriptors, export_id);
    };
    access.pin = [descriptors, memory_owner = std::weak_ptr{guest_memory}](
                     const services::GraphicsBufferBacking& backing,
                     std::uint64_t offset,
                     std::uint64_t length) -> runtime::LaunchResult<runtime::GuestMemoryPin> {
        // GuestMemory owns the process handle table, which can own VI and this
        // binding. Borrow the authority for this call; retained pins own only
        // its cycle-free backing state, independently of the handle table.
        const auto guest_memory = memory_owner.lock();
        if (!guest_memory) {
            return graphics_failure(runtime::LaunchError::invalid_input,
                                    "nvdrv.pin_exported_memory",
                                    "guest memory is unavailable");
        }
        std::shared_ptr<void> retained;
        {
            const std::lock_guard lock(descriptors->mutex);
            const auto found = descriptors->retained_exports.find(backing.export_id);
            if (found == descriptors->retained_exports.end()) {
                return graphics_failure(runtime::LaunchError::permission_denied,
                                        "nvdrv.pin_exported_memory",
                                        "the export does not belong to this driver client");
            }
            retained = found->second.owner.lock();
            if (!same_owner(retained, backing.owner) ||
                found->second.alignment != backing.alignment ||
                found->second.guest_address != backing.guest_address ||
                found->second.size != backing.size) {
                return graphics_failure(runtime::LaunchError::permission_denied,
                                        "nvdrv.pin_exported_memory",
                                        "the retained export identity or metadata does not match");
            }
        }
        if (length == 0 || offset > backing.size || length > backing.size - offset ||
            backing.guest_address > std::numeric_limits<std::uint64_t>::max() - offset) {
            return graphics_failure(runtime::LaunchError::invalid_input,
                                    "nvdrv.pin_exported_memory",
                                    "the surface range is outside the retained export");
        }
        const auto address = backing.guest_address + offset;
        if (address > std::numeric_limits<std::uint64_t>::max() - length) {
            return graphics_failure(runtime::LaunchError::invalid_input,
                                    "nvdrv.pin_exported_memory",
                                    "the surface range overflows the guest address space");
        }
        auto pinned = guest_memory->pin({address, length}, memory::Permission::read);
        if (auto* failure = std::get_if<runtime::LaunchFailure>(&pinned)) {
            return std::move(*failure);
        }
        auto guest_pin = std::get<runtime::GuestMemoryPin>(std::move(pinned));
        if (!guest_pin) {
            return graphics_failure(runtime::LaunchError::host_failure,
                                    "nvdrv.pin_exported_memory",
                                    "guest memory returned an empty surface pin");
        }
        try {
            return std::make_shared<RetainedGraphicsPin>(std::move(guest_pin),
                                                         std::move(retained));
        } catch (const std::bad_alloc&) {
            return graphics_failure(runtime::LaunchError::resource_limit,
                                    "nvdrv.pin_exported_memory",
                                    "the retained surface pin could not be allocated");
        }
    };
    access.wait_fence = [descriptors](const services::GraphicsFence& fence,
                                      std::optional<runtime::ClockInstant> deadline,
                                      std::stop_token stop) -> runtime::LaunchStatus {
        if (fence.syncpoint < 0 ||
            static_cast<std::size_t>(fence.syncpoint) >= descriptors->syncpoints.size()) {
            return graphics_failure(runtime::LaunchError::invalid_input,
                                    "nvdrv.wait_fence",
                                    "the producer fence syncpoint is invalid");
        }
        if (stop.stop_requested()) {
            return graphics_failure(runtime::LaunchError::cancelled,
                                    "nvdrv.wait_fence",
                                    "the producer fence wait was cancelled");
        }
        poll_backend(*descriptors);
        const auto id = static_cast<std::uint32_t>(fence.syncpoint);
        bool stop_requested = false;
        std::stop_callback cancellation(stop, [&] {
            const std::lock_guard guard(descriptors->mutex);
            stop_requested = true;
            descriptors->syncpoint_changed.notify_all();
        });
        std::unique_lock lock(descriptors->mutex);
        const auto channel_failure = [&]() -> const std::string* {
            if (!descriptors->timeline->failure.empty()) return &descriptors->timeline->failure;
            const auto& failure = descriptors->timeline->channel_failures[id];
            return failure.empty() ? nullptr : &failure;
        };
        const auto finished = [&] {
            return stop_requested || descriptors->timeline->stopped || channel_failure() ||
                   syncpoint_expired(*descriptors->timeline, id, fence.threshold);
        };
        bool awakened = true;
        if (deadline) {
            awakened = descriptors->syncpoint_changed.wait_until(lock, *deadline, finished);
        } else {
            descriptors->syncpoint_changed.wait(lock, finished);
        }
        if (stop_requested || descriptors->timeline->stopped) {
            return graphics_failure(runtime::LaunchError::cancelled,
                                    "nvdrv.wait_fence",
                                    "the producer fence wait was cancelled");
        }
        if (const auto* failure = channel_failure())
            return graphics_failure(runtime::LaunchError::host_failure,
                                    "nvdrv.wait_fence", *failure);
        if (!awakened) {
            return graphics_failure(runtime::LaunchError::deadline_expired,
                                    "nvdrv.wait_fence",
                                    "the producer fence deadline expired");
        }
        return std::monostate{};
    };
    return access;
}

std::size_t DriverSession::domain_object_count() const {
    const std::lock_guard lock(descriptors_->mutex);
    return converted_ ? 1 : 0;
}

std::optional<std::string> DriverSession::gpu_failure() const {
    // Every guest supervisor call polls this: answer without the descriptor
    // lock while nothing has been recorded for this client.
    if (!descriptors_->timeline->failed.load(std::memory_order_acquire)) return std::nullopt;
    const std::lock_guard lock(descriptors_->mutex);
    if (!descriptors_->timeline->failure.empty()) return descriptors_->timeline->failure;
    for (const auto& failure : descriptors_->timeline->channel_failures)
        if (!failure.empty()) return failure;
    for (const auto& [fd, device] : descriptors_->devices)
        if (device.gpu_state && !device.gpu_state->stop_diagnostic.empty())
            return device.gpu_state->stop_diagnostic;
    return std::nullopt;
}

const char* DriverSession::diagnostic() const {
    const std::lock_guard lock(descriptors_->mutex);
    return diagnostic_;
}

std::uint32_t DriverSession::device_bound_channel(std::uint32_t fd) const {
    const std::lock_guard lock(descriptors_->mutex);
    const auto found = descriptors_->devices.find(fd);
    return found == descriptors_->devices.end() ? 0u : found->second.bound_channel;
}

std::optional<std::uint32_t> DriverSession::device_mapping_kind(
    std::uint32_t fd, std::uint64_t offset) const {
    const std::lock_guard lock(descriptors_->mutex);
    const auto device = descriptors_->devices.find(fd);
    if (device == descriptors_->devices.end()) return std::nullopt;
    const auto& mappings = device->second.mappings;
    const auto found = std::find_if(mappings.begin(), mappings.end(),
        [offset](const AddressMapping& mapping) { return mapping.offset == offset; });
    return found == mappings.end() ? std::nullopt
                                  : std::optional{found->kind};
}

std::optional<bool> DriverSession::is_fence_expired(
    std::uint32_t id, std::uint32_t threshold) const {
    const std::lock_guard lock(descriptors_->mutex);
    if (id >= descriptors_->syncpoints.size() || !descriptors_->timeline->failure.empty() ||
        !descriptors_->timeline->channel_failures[id].empty()) return std::nullopt;
    return syncpoint_expired(*descriptors_->timeline, id, threshold);
}

std::string DriverSession::device_path(std::uint32_t fd) const {
    const std::lock_guard lock(descriptors_->mutex);
    const auto found = descriptors_->devices.find(fd);
    return found == descriptors_->devices.end() ? std::string{} : found->second.path;
}

}  // namespace armigo::services::nvdrv
