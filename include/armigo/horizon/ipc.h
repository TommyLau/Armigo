// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/startup_queries.h>

#include <array>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace armigo::horizon {

inline constexpr std::size_t ipc_message_capacity = 256;
inline constexpr std::uint32_t session_closed_result = 0xf601;
using IpcMessage = std::array<std::byte, ipc_message_capacity>;

// Pending, unsupported and cancelled are host dispatcher states, never guest
// Results. Cancellation is request-local host-stop provenance and must not
// resume the guest or suppress a different request's transport failure.
enum class IpcDisposition { handled, pending, unsupported, invalid_backend, cancelled };

class ClientSession;

// A reply may transfer new caller handles through the HIPC handle arrays: one
// moved session (APM OpenSession) or up to two copied objects (a network
// request's two readable events).
inline constexpr std::size_t max_transferred_handles = 2;

struct IpcReply {
    IpcDisposition disposition = IpcDisposition::unsupported;
    std::uint32_t transport_result = 0;
    IpcMessage bytes{};
    std::size_t size = 0;
    // Session objects the reply transfers, in array order. The dispatcher owns
    // handle allocation: it inserts each object into the caller's handle table
    // and writes the resulting handle into the move-handle array. A service
    // never invents a caller handle. bytes must carry the CMIF header at byte
    //16; the dispatcher relocates it past the array and fills the header.
    // Any kernel object the reply hands to the caller, not only sessions: a display,
    // an event or a transfer-memory record all travel the same way.
    std::array<std::shared_ptr<KernelObject>, max_transferred_handles> transferred{};
    std::size_t transferred_count = 0;
    // A reply that hands out a plain object the caller keeps (a shared memory, an event)
    // answers an OutCopyHandle field, so the dispatcher reports it in the copy-handle array.
    // Interface transfers, such as APM's session, are moved instead.
    bool transferred_as_copy = false;
};

// One request's guest buffers. The dispatcher validates the HIPC descriptor
// area and offers each buffer as an address plus a size; the service decides
// whether to read or write it. A reader or writer returns false when the guest
// memory is not readable or writable, so a service can never dereference an
// unchecked pointer.
struct GuestBuffer {
    std::uint64_t address = 0;
    std::uint32_t size = 0;
    bool input = false;
    bool output = false;
};

inline constexpr std::size_t max_guest_buffers = 4;

class GuestBuffers {
public:
    using Reader = bool (*)(void*, std::uint64_t, std::span<std::byte>) noexcept;
    using Writer = bool (*)(void*, std::uint64_t, std::span<const std::byte>) noexcept;

    GuestBuffers() = default;
    GuestBuffers(Reader reader, Writer writer, void* context)
        : reader_(reader), writer_(writer), context_(context) {}

    std::size_t size() const noexcept { return count_; }
    const GuestBuffer& operator[](std::size_t index) const noexcept { return buffers_[index]; }
    bool empty() const noexcept { return count_ == 0; }

    // Copies one validated buffer; false when the range or the guest mapping is
    // not usable.
    bool full() const noexcept { return count_ == buffers_.size(); }
    bool read(std::size_t index, std::span<std::byte> destination) const noexcept;
    bool write(std::size_t index, std::span<const std::byte> source) const noexcept;
    // Writes one buffer starting at `offset` bytes into it.
    bool write_at(std::size_t index, std::size_t offset,
                  std::span<const std::byte> source) const noexcept;
    // Accesses an embedded guest address through the same checked runtime
    // capability. The service owns the protocol bounds for these indirect ranges.
    bool read_memory(std::uint64_t address, std::span<std::byte> out) const noexcept {
        return reader_ && out.size() <= std::numeric_limits<std::uint64_t>::max() - address &&
               reader_(context_, address, out);
    }
    bool write_memory(std::uint64_t address, std::span<const std::byte> in) const noexcept {
        return writer_ && in.size() <= std::numeric_limits<std::uint64_t>::max() - address &&
               writer_(context_, address, in);
    }
    // Reads a NUL-terminated string from one buffer, bounded by its size.
    std::optional<std::string> read_string(std::size_t index) const;

    void add(const GuestBuffer& buffer) noexcept {
        if (count_ < buffers_.size()) buffers_[count_++] = buffer;
    }

private:
    std::array<GuestBuffer, max_guest_buffers> buffers_{};
    std::size_t count_ = 0;
    Reader reader_ = nullptr;
    Writer writer_ = nullptr;
    void* context_ = nullptr;
};

// The runtime's ability to map guest memory on a service's behalf. A service that
// must hand the guest an address it can read and write uses this instead of
// inventing one; both calls are optional and their absence stays explicit.
struct GuestMemoryMapper {
    bool (*map)(void*, std::uint64_t address, std::uint64_t size) noexcept = nullptr;
    void (*unmap)(void*, std::uint64_t address, std::uint64_t size) noexcept = nullptr;
    void* context = nullptr;
    // Required when a service retains a mapping beyond one request. The owner
    // pins both callback context and backing until the retained mapping ends.
    // Mapping callbacks must not re-enter the service that invokes them; unmap
    // may run while that service serializes descriptor destruction.
    std::shared_ptr<void> lifetime{};
};

class ClientSession : public KernelObject {
public:
    virtual ~ClientSession() = default;
    // Retain the existing close acknowledgement for legacy contracts. Opted-in
    // sessions release server state and reject subsequent requests; the client
    // handle remains owned by its caller until CloseHandle.
    virtual IpcReply close_server() { return {.disposition = IpcDisposition::handled}; }
    // Called before dispatch with the runtime's current mapping capability. The
    // default keeps it absent, so a service that needs it reports the gap.
    virtual void set_memory_mapper(GuestMemoryMapper) noexcept {}
    // sender_pid comes from the invoking process, not a PID placeholder in
    // message. Implementations validate their supported wire protocol. Handles
    // in a reply must come from IpcReply::transferred, never from the service.
    // No server-process handle namespace is emulated by this host boundary.
    virtual IpcReply request(const IpcMessage& message, std::uint64_t sender_pid) = 0;

    // Requests with the dispatcher's guest-memory capability, even when no
    // buffer descriptors are present. The default forwards to request() so a
    // service that only understands inline data keeps its existing validation.
    virtual IpcReply request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                          const GuestBuffers& buffers) {
        (void)buffers;
        return request(message, sender_pid);
    }

    // Opt in to typed copied-object delivery. Existing services may resolve
    // their own raw handles and pseudo-handles using their process context.
    virtual bool receives_copied_objects() const noexcept { return false; }

    // For opted-in services, copied handles are resolved in the caller's table
    // and pinned for this request. A retained copy keeps the shared object, never
    // the caller's numeric handle. Memory callbacks remain request-local.
    // A null entry denotes a current-process/current-thread pseudo-handle that
    // needs the service's caller context; typed-object consumers must reject it.
    // Existing services retain their own wire handling through the default.
    virtual IpcReply request_with_handles(
        const IpcMessage& message, std::uint64_t sender_pid, const GuestBuffers& buffers,
        std::span<const std::shared_ptr<KernelObject>> copied_objects) {
        (void)copied_objects;
        // Descriptor-free operations can still access indirect guest backing.
        return request_with_buffers(message, sender_pid, buffers);
    }
};

// Base for diagnostic decorators. The wrapped session must be non-null.
class ForwardingSession : public ClientSession {
public:
    explicit ForwardingSession(std::shared_ptr<ClientSession> session)
        : inner(std::move(session)) {}

    IpcReply close_server() override { return inner->close_server(); }

    void set_memory_mapper(GuestMemoryMapper mapper) noexcept override {
        inner->set_memory_mapper(mapper);
    }
    IpcReply request(const IpcMessage& message, std::uint64_t sender_pid) override {
        return inner->request(message, sender_pid);
    }
    IpcReply request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                  const GuestBuffers& buffers) override {
        return inner->request_with_buffers(message, sender_pid, buffers);
    }
    bool receives_copied_objects() const noexcept override {
        return inner->receives_copied_objects();
    }
    IpcReply request_with_handles(
        const IpcMessage& message, std::uint64_t sender_pid, const GuestBuffers& buffers,
        std::span<const std::shared_ptr<KernelObject>> copied_objects) override {
        // Preserve decorators' buffered hooks and request-local memory access.
        // Opted-in services also need the resolved copied objects.
        if (!inner->receives_copied_objects())
            return ClientSession::request_with_handles(message, sender_pid, buffers, copied_objects);
        return inner->request_with_handles(message, sender_pid, buffers, copied_objects);
    }

protected:
    const std::shared_ptr<ClientSession> inner;
};

struct SessionConnection {
    IpcDisposition disposition = IpcDisposition::unsupported;
    std::uint32_t result = 0;
    std::shared_ptr<ClientSession> session;
};

// A missing registered port returns the firmware's port-not-found Result.
// An unsupported host implementation must instead stay explicitly unsupported.
using NamedPortConnector = SessionConnection (*)(void*, std::string_view);
using GuestByteReader = bool (*)(void*, std::uint64_t, std::byte&) noexcept;

struct IpcCallContext {
    std::uint64_t process_id = 0;
    // Validated, lifetime-pinned backing for the invoking thread's IPC area.
    // The runtime owns guest address translation and mapping synchronization.
    std::span<std::byte> message;
    GuestByteReader read_byte = nullptr;
    void* memory_context = nullptr;
    // Copy helpers for buffer-carrying requests. Both are optional: without
    // them such requests stay explicitly unsupported.
    bool (*read_memory)(void*, std::uint64_t, std::span<std::byte>) noexcept = nullptr;
    bool (*write_memory)(void*, std::uint64_t, std::span<const std::byte>) noexcept = nullptr;
    // Requests carry compact buffer descriptors: the low 32 bits of the guest
    // address plus a size. The runtime supplies the region base whose low 32
    // bits are implied, so a service never guesses an address.
    std::uint64_t buffer_region_base = 0;
    // Mapping capability for services that must give the guest an address.
    GuestMemoryMapper memory_mapper{};
};

// Initial synchronous application-side IPC boundary only: SVC 0x1f and 0x21.
// No generic descriptor transfer, receive loop or scheduler is implied. The
// service implementation owns per-session request serialization. All borrowed
// callbacks and handle storage must outlive dispatch calls.
// Locates the outer CMIF header of a request. A domain header (object count and
// header size) precedes the CMIF header when present; a stale SFCI sequence in
// a payload or descriptor area must not be mistaken for it.
std::optional<std::size_t> locate_cmif(const IpcMessage& message) noexcept;

class IpcDispatcher {
public:
    IpcDispatcher(HandleTable& handles, NamedPortConnector connector, void* context = nullptr)
        : handles_(&handles), connector_(connector), connector_context_(context) {}

    IpcDisposition dispatch(std::uint16_t svc, QueryRegisters& registers,
                            const IpcCallContext& caller) const;
    const HandleTable& handles() const noexcept { return *handles_; }

private:
    IpcDisposition connect(QueryRegisters&, const IpcCallContext&) const;
    IpcDisposition send(QueryRegisters&, const IpcCallContext&) const;
    HandleTable* handles_;
    NamedPortConnector connector_;
    void* connector_context_;
};

} // namespace armigo::horizon
