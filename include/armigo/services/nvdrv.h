// ============================================================================
// Armigo Services - Initial Nvidia Driver Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/horizon/event.h>
#include <armigo/graphics/engines.h>
#include <armigo/host1x/host1x.h>
#include <armigo/services/graphics_buffer_backing.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace armigo::runtime {
class GuestMemory;
struct GraphicsBackingAccess;
}

namespace armigo::services::nvdrv {

struct GpuChannelState;
class GpuWorker;

// One GPU virtual-address mapping of a memory handle.
struct AddressMapping {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::uint32_t memory_handle = 0;
    std::uint32_t page_size = 0;
    std::uint64_t buffer_offset = 0;
    // MapBufferEx retains the requested 32-bit kind; hardware uses its low byte.
    std::uint32_t kind = 0;
    std::uint16_t flags = 0;
    // GPU mappings retain their allocation independently of the client handle.
    std::uint64_t guest_address = 0;
    std::uint64_t backing_size = 0;
    std::shared_ptr<void> retained_mapping;
    bool automatically_allocated = false;
    bool sparse_remap = false;
};

// A driver memory handle. The console backs each handle with real memory that the
// GPU and the CPU both address, so the service allocates the requested size and
// keeps it until both the client handle and retained mappings are released.
struct NvMapHandle {
    std::uint32_t size = 0;
    // The alignment the driver requested for this handle, reported back from the
    // parameter query.
    std::uint32_t alignment = 0;
    // Free reports bit zero from the first successful Alloc request.
    std::uint32_t allocation_flags = 0;
    // The guest address the driver reads and writes this memory through, or zero
    // while the runtime has not provided backing for it.
    std::uint64_t guest_address = 0;
    // GetId marks this client-owned object as exported only after its reply is
    // copied to guest memory. Imported presentation backing has a separate lease.
    // The ImportFromId ioctl remains unsupported.
    bool exported = false;
    std::shared_ptr<void> retained_mapping;
    std::vector<std::byte> memory;
};

// One window of GPU virtual addresses. The allocator serves page runs from the
// window and reuses a released run before it extends the cursor, so an address
// handed to the guest stays stable until the guest frees it.
class AddressWindow {
public:
    void configure(std::uint64_t start, std::uint64_t size, std::uint64_t page_size) noexcept;
    // Serves a run of `pages` pages at or above `alignment`, or nothing when the
    // window cannot hold it.
    std::optional<std::uint64_t> allocate(std::uint64_t pages, std::uint64_t alignment) noexcept;
    // Binds a run at an address the guest selected.
    bool allocate_at(std::uint64_t offset, std::uint64_t pages) noexcept;
    bool release(std::uint64_t offset, std::uint64_t pages) noexcept;
    bool bound() const noexcept { return page_size_ != 0; }
    std::uint64_t start() const noexcept { return start_; }
    std::uint64_t end() const noexcept { return start_ + size_; }
    std::uint64_t page_size() const noexcept { return page_size_; }

private:
    std::uint64_t next(std::uint64_t address, std::uint64_t alignment) const noexcept;
    std::uint64_t start_ = 0;
    std::uint64_t size_ = 0;
    std::uint64_t page_size_ = 0;
    std::uint64_t cursor_ = 0;
    // Occupied runs keyed by their first page.
    std::map<std::uint64_t, std::uint64_t> used_;
};

// A device the guest opened through the driver. The service records the device
// path because every ioctl meaning depends on it; the ioctl surface itself is
// implemented device by device, and an unknown one reports a diagnostic instead
// of pretending success.
struct Device {
    std::string path;
    std::uint32_t fd = 0;
    // Address-space state: Initialize binds the range the guest selected, and the
    // region query reports it back split into a small-page and a big-page window.
    bool address_space_bound = false;
    // Last channel bound to this address space, for diagnostics. Channels
    // independently retain their address-space descriptor for execution.
    std::uint32_t bound_channel = 0;
    std::uint32_t bound_address_space = 0;
    // NVMap descriptor the GPU channel allocates its objects through.
    std::uint32_t nvmap_fd = 0;
    bool nvmap_bound = false;
    // GPFIFO state: the channel's syncpoint and the allocated entry count.
    std::uint32_t gpfifo_entries = 0;
    std::uint32_t syncpoint_id = 0;
    // Multimedia channels: the submit timeout the guest set, in milliseconds.
    std::uint32_t submit_timeout = 0;
    std::shared_ptr<GpuChannelState> gpu_state;
    std::uint64_t client_data = 0;
    std::uint32_t big_page_size = 0;
    std::uint64_t va_range_start = 0;
    std::uint64_t va_range_end = 0;
    std::uint64_t va_range_split = 0;
    AddressWindow small_pages;
    AddressWindow big_pages;
    // GPU virtual-address mappings of the bound address space.
    std::vector<AddressMapping> mappings;
    // Generation shared by every submission view of the current mappings;
    // zero after any mapping change allocates a fresh one.
    std::uint64_t view_generation = 0;
    // Sparse AllocSpace reservations remain distinct from their mapped pages.
    std::map<std::uint64_t, std::uint64_t> sparse_reservations;
};

struct SyncpointState {
    std::uint32_t completed = 0;
    std::uint32_t reserved = 0;
};

struct SyncpointEvent {
    std::shared_ptr<horizon::Event> event;
    std::uint32_t syncpoint = 0;
    std::uint32_t threshold = 0;
    bool pending = false;
};

struct RetainedExportState {
    std::uint32_t alignment = 0;
    std::uint64_t guest_address = 0;
    std::uint64_t size = 0;
    // The import lease owns backing but expires independently of renderer pins.
    std::weak_ptr<void> owner;
};

// Publications retain only this timeline, never the descriptor/worker owner.
struct SyncpointTimeline : std::enable_shared_from_this<SyncpointTimeline> {
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::array<SyncpointState, 192> syncpoints{};
    // Event slots by /dev/nvhost-ctrl descriptor: every open of the device has
    // its own 64 slots, which EventRegister, the event waits and QueryEvent
    // name together with that descriptor.
    using EventSlots = std::array<SyncpointEvent, 64>;
    std::map<std::uint32_t, EventSlots> events;
    bool stopped = false;
    std::string failure;
    std::array<std::string, 192> channel_failures{};
    // Set, under the mutex, after a failure or channel stop is recorded for
    // this client; until then a failure poll has nothing to find and need not
    // take the descriptor lock.
    std::atomic<bool> failed{false};
};

// Descriptor state for one driver client. Session clones continue the same
// client, so they must see the same descriptors: the guest numbers them per
// client and dispatches every request against that one table.
struct DescriptorTable {
    DescriptorTable();
    ~DescriptorTable();
    // Request serialization remains held while worker barriers release mutex.
    mutable std::mutex requests;
    std::unique_ptr<GpuWorker> gpu;
    // Clones dispatch against one client, including allocations and locking.
    std::shared_ptr<SyncpointTimeline> timeline = std::make_shared<SyncpointTimeline>();
    std::mutex& mutex = timeline->mutex;
    std::condition_variable& syncpoint_changed = timeline->changed;
    graphics::RenderCallbacks rendering{};
    std::map<std::uint32_t, NvMapHandle> memory_handles;
    // Weak identities let a retained export remain verifiable after Free
    // without making the descriptor table own the mapping indefinitely.
    std::map<std::uint32_t, RetainedExportState> retained_exports;
    std::uint32_t next_memory_handle = 0;
    std::uint32_t next_gpu_syncpoint = 24;
    std::array<SyncpointState, 192>& syncpoints = timeline->syncpoints;
    std::map<std::uint32_t, SyncpointTimeline::EventSlots>& events = timeline->events;
    std::uint32_t next_fd = 1;
    std::map<std::uint32_t, Device> devices;
    // Multimedia engines (NVDEC, VIC) address memory through 32-bit device
    // addresses: MapBuffer pins an NVMap handle at one. The address stays
    // mapped after the last UnmapBuffer, because submitted work still reads
    // through it; freeing the handle releases it.
    struct Host1xPin {
        std::uint32_t address = 0;
        std::uint32_t references = 0;
        std::uint32_t size = 0;
        std::uint64_t guest_address = 0;
    };
    // Engines read pins from their own threads, so pins have their own lock,
    // taken after the descriptor lock and never while waiting.
    std::mutex host1x_pins_mutex;
    std::map<std::uint32_t, Host1xPin> host1x_pins;
    std::uint32_t next_host1x_address = 0x00100000;
    // Device address ranges released by freed handles (address -> bytes),
    // merged with their neighbours; MapBuffer takes the first that fits.
    std::map<std::uint32_t, std::uint32_t> host1x_free;
    // The multimedia engines of the process, created with the first channel.
    // Declared last: destroyed first, stopping engine threads before the state
    // they use.
    std::unique_ptr<host1x::Hub> host1x;
};

// Initial native nvdrv session: session control, device open and close, and an
// explicit refusal for ioctls that are not implemented yet.
class DriverSession final : public horizon::ClientSession {
public:
    DriverSession();
    // A clone continues the parent client: it shares the descriptor table.
    explicit DriverSession(const DriverSession& parent);
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                          std::uint64_t sender_pid,
                                          const horizon::GuestBuffers& buffers) override;
    // The runtime's guest-memory capability, supplied before dispatch.
    void set_memory_mapper(horizon::GuestMemoryMapper mapper) noexcept override;
    // Opening a GPU device (a path ending in "-gpu") stops with `reason`. The
    // Switch 2 driver session uses this where the title's own NVN driver
    // reaches the GPU, which Armigo's NVN driver replaces.
    void refuse_gpu_devices(const char* reason) noexcept;
    // Shared by clones, including an optional retained callback context owner.
    void set_render_backend(graphics::RenderCallbacks rendering);
    // Enable pipelined submission with an independently retained memory capability.
    // The owner must retain the callback context, without owning this driver.
    // Clients without a durable capability keep the synchronous adapter.
    void enable_async_gpu(horizon::GuestBuffers memory, std::shared_ptr<void> owner);
    void stop_gpu() noexcept;
    void drain_gpu();
    std::optional<std::string> gpu_failure() const;
    std::optional<services::GraphicsBufferBacking>
    retain_exported_memory(std::uint32_t export_id);
    // Binds presentation backing and producer-fence access to this exact
    // driver's shared client namespace and one process memory authority.
    // The process retains GuestMemory through pin calls; completed pins and
    // exports retain their backing independently without a handle-table cycle.
    runtime::GraphicsBackingAccess graphics_backing_access(
        std::shared_ptr<runtime::GuestMemory> memory) const;
    std::size_t domain_object_count() const;
    const char* diagnostic() const;
    // The device path recorded for a file descriptor, for host diagnostics.
    std::string device_path(std::uint32_t fd) const;
    // The GPU channel bound to an address-space descriptor, for host diagnostics.
    std::uint32_t device_bound_channel(std::uint32_t fd) const;
    // The kind recorded at an exact GPU mapping base, for host diagnostics.
    std::optional<std::uint32_t> device_mapping_kind(
        std::uint32_t fd, std::uint64_t offset) const;
    // Read-only client fence query for host consumers. Invalid ids return no
    // value; expiration keeps the driver's reservation and wrap semantics.
    std::optional<bool> is_fence_expired(std::uint32_t id,
                                         std::uint32_t threshold) const;

private:
    horizon::IpcReply unsupported(const char* detail);
    const char* refused_gpu_devices_ = nullptr;
    horizon::IpcReply open_device(const horizon::IpcMessage& message, std::size_t cmif,
                                 const horizon::GuestBuffers& buffers);
    horizon::IpcReply query_event(const horizon::IpcMessage& message, std::size_t cmif);
    horizon::IpcReply ioctl3(const horizon::IpcMessage& message, std::size_t cmif,
                             const horizon::GuestBuffers& buffers);
    horizon::IpcReply ioctl(const horizon::IpcMessage& message, std::size_t cmif,
                            const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_characteristics(const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_zcull_get_ctx_size(const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_zcull_get_info(const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_channel_set_nvmap(Device& device,
                                            const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_channel_alloc_gpfifo(Device& device,
                                               const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_channel_submit_gpfifo(Device& device, const horizon::GuestBuffers& buffers);
    horizon::IpcReply syncpoint_ioctl(std::uint32_t fd, std::uint32_t request,
                                       const horizon::GuestBuffers& buffers);
    // Channel ioctls of the multimedia engines (/dev/nvhost-nvdec, /dev/nvhost-vic).
    horizon::IpcReply multimedia_ioctl(Device& device, std::uint32_t request,
                                       const horizon::GuestBuffers& buffers);
    // Submit on a multimedia channel: command buffers to the channel's engine,
    // syncpoint increments returned as fence thresholds.
    horizon::IpcReply multimedia_submit(Device& device, const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_channel_set_client_data(Device& device,
                                                  const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_channel_get_client_data(Device& device,
                                                  const horizon::GuestBuffers& buffers);
    horizon::IpcReply address_space_bind_channel(Device& device,
                                                 const horizon::GuestBuffers& buffers);
    horizon::IpcReply gpu_tpc_masks(const horizon::GuestBuffers& buffers);
    horizon::IpcReply address_space_initialize(Device& device,
                                              const horizon::GuestBuffers& buffers);
    horizon::IpcReply address_space_va_regions(Device& device,
                                              const horizon::GuestBuffers& buffers);
    horizon::IpcReply address_space_allocate(Device& device,
                                             const horizon::GuestBuffers& buffers);
    horizon::IpcReply address_space_remap(Device& device, std::uint32_t request,
                                          const horizon::GuestBuffers& buffers);
    horizon::IpcReply address_space_map(Device& device,
                                        const horizon::GuestBuffers& buffers);
    horizon::IpcReply address_space_unmap(Device& device,
                                         const horizon::GuestBuffers& buffers);
    horizon::IpcReply memory_create(const horizon::GuestBuffers& buffers);
    horizon::IpcReply memory_get_id(const horizon::GuestBuffers& buffers);
    horizon::IpcReply memory_allocate(const horizon::GuestBuffers& buffers);
    horizon::IpcReply memory_free(const horizon::GuestBuffers& buffers);
    horizon::IpcReply memory_param(const horizon::GuestBuffers& buffers);
    horizon::IpcReply close_device(const horizon::IpcMessage& message, std::size_t cmif,
                                   const horizon::GuestBuffers& buffers);
    bool notify_memory_changed(
        std::span<const graphics::Mapping> removed,
        std::span<const std::shared_ptr<void>> released,
        const horizon::GuestBuffers& buffers) noexcept;
    horizon::IpcReply domain_request(const horizon::IpcMessage& message,
                                     const horizon::GuestBuffers& buffers);
    bool converted_ = false;
    std::shared_ptr<DescriptorTable> descriptors_ = std::make_shared<DescriptorTable>();
    // Diagnostic text naming the refused ioctl; empty when there is none.
    std::string ioctl_diagnostic_;
    // Device and request of each unimplemented ioctl already reported.
    std::set<std::string> reported_ioctls_;
    // The runtime's guest-memory capability, supplied before dispatch.
    horizon::GuestMemoryMapper mapper_{};
    // Handle of the transfer memory the client supplied to Initialize.
    std::uint32_t work_memory_ = 0;
    // Tag of the last tagged clone, recorded for host diagnostics.
    std::uint32_t clone_tag_ = 0;
    // Session handle and inline word of the last binding command.
    std::uint32_t bound_session_ = 0;
    std::uint32_t bound_value_ = 0;
    // Mode word of the last driver mode command.
    std::uint32_t driver_mode_ = 0;
    const char* diagnostic_ = nullptr;
};

}  // namespace armigo::services::nvdrv
