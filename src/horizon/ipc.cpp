// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/ipc.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <ostream>
#include <string>
#include <utility>

namespace armigo::horizon {
namespace {
constexpr std::uint32_t invalid_handle = 0xe401;
constexpr std::uint32_t invalid_copy = 0xe601;
constexpr std::uint32_t name_too_long = 0xee01;
constexpr std::uint32_t out_of_handles = 0xd201;

void finish(QueryRegisters& registers, std::uint32_t result, std::uint64_t output = 0) {
    registers.fill(0);
    registers[0] = result;
    registers[1] = output;
}

std::uint32_t read_word(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i != 4; ++i)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + i])} << (8 * i);
    return value;
}

void write_word(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i)
        message[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

// A reply that transfers handles carries the HIPC special header in the second
// header word (bit31) and its descriptor word at byte8: bit0 marks the special
// header, bits1..4 count copied handles and bits5..8 count moved handles. The
// moved-handle words start at byte16, and the CMIF header that services author
// at byte16 is relocated to the next 16-byte boundary. That is exactly the
// layout the original SDK's reply parser derives from the descriptor counts
// (sdk sub_9BA5C and sub_9BB10): the counts place both the handle array and the
// CMIF header. Copy-handle replies already use the same descriptor word without
// the special header.
constexpr std::size_t handle_area = 20;
constexpr std::size_t handle_entry = 4;
constexpr std::uint32_t special_header_present = 1u;
constexpr unsigned copy_handle_shift = 1;
constexpr unsigned moved_handle_shift = 5;

// A request that carries guest buffers keeps two-word descriptors directly
// after the two header words: {u32 size, u32 address low bits}. The runtime
// supplies the region base whose low 32 bits are implied. A request whose domain
// header starts at byte32 (the PID variant) leaves bytes8..31 for descriptors;
// the list ends at the first entry with a zero size or a zero address. This is
// the shape the original SDK builds for file-system storage transfers.
constexpr std::size_t request_descriptor_area = 8;
constexpr std::size_t request_descriptor_limit = 32;

// The HIPC request header counts buffer descriptors instead of describing them
// inline. The first word holds the pointer, send, receive and exchange counts
// and the second word the raw data word count, the receive list type and the
// special header flag. Pointer descriptors take two words and map-alias
// descriptors take three, so a request whose CMIF header follows them keeps no
// descriptor pair where the older shape puts one.
constexpr unsigned hipc_pointer_shift = 16;
constexpr unsigned hipc_send_shift = 20;
constexpr unsigned hipc_receive_shift = 24;
constexpr unsigned hipc_exchange_shift = 28;
constexpr unsigned hipc_count_mask = 0xf;
constexpr std::size_t hipc_pointer_words = 2;
constexpr std::size_t hipc_map_alias_words = 3;
constexpr std::size_t hipc_header_words = 2;
constexpr unsigned hipc_special_pid = 1;
constexpr unsigned hipc_special_copy_shift = 1;
constexpr unsigned hipc_special_move_shift = 5;
constexpr unsigned hipc_pointer_address_high_shift = 6;
constexpr unsigned hipc_pointer_address_mid_shift = 12;
constexpr unsigned hipc_pointer_size_shift = 16;
constexpr unsigned hipc_map_alias_address_high_shift = 2;
constexpr unsigned hipc_map_alias_address_mid_shift = 28;
constexpr unsigned hipc_address_mid_bits = 4;
constexpr unsigned hipc_address_high_mask = 7;
constexpr unsigned hipc_address_mid_mask = 0xf;

bool domain_header(const IpcMessage& message, std::size_t offset) {
    if (offset + 4 > message.size()) return false;
    const auto word = read_word(message, offset);
    const auto objects = word & 0xffffu;
    const auto size = word >> 16;
    return objects >= 1 && objects <= 8 && size >= 0x10;
}

std::uint64_t hipc_address(std::uint32_t low, unsigned high, unsigned mid) {
    const auto upper = (static_cast<std::uint64_t>(high) << hipc_address_mid_bits) | mid;
    return (upper << 32) | low;
}

// Adds the buffers a counted HIPC request declares. A message that cannot be
// walked leaves the request without buffers instead of inventing any.
void add_hipc_buffers(const IpcMessage& message, GuestBuffers& buffers,
                      std::uint64_t region_base) {
    const auto counts = read_word(message, 0);
    const auto second = read_word(message, 4);
    const unsigned pointers = (counts >> hipc_pointer_shift) & hipc_count_mask;
    const unsigned sends = (counts >> hipc_send_shift) & hipc_count_mask;
    const unsigned receives = (counts >> hipc_receive_shift) & hipc_count_mask;
    const unsigned exchanges = (counts >> hipc_exchange_shift) & hipc_count_mask;
    std::size_t cursor = hipc_header_words * 4;
    if (second >> 31) {
        const auto special = read_word(message, cursor);
        cursor += 4;
        if (special & hipc_special_pid) cursor += 8;
        const auto handles = ((special >> hipc_special_copy_shift) & hipc_count_mask) +
                             ((special >> hipc_special_move_shift) & hipc_count_mask);
        cursor += handles * 4;
    }
    for (unsigned index = 0; index != pointers; ++index) {
        if (cursor + hipc_pointer_words * 4 > message.size()) return;
        const auto packed = read_word(message, cursor);
        const auto low = read_word(message, cursor + 4);
        cursor += hipc_pointer_words * 4;
        const auto size = packed >> hipc_pointer_size_shift;
        const auto address = hipc_address(
            low, (packed >> hipc_pointer_address_high_shift) & hipc_address_high_mask,
            (packed >> hipc_pointer_address_mid_shift) & hipc_address_mid_mask);
        if (size == 0 || address == 0 || buffers.full()) continue;
        buffers.add({.address = address, .size = size, .input = true, .output = true});
    }
    for (unsigned index = 0; index != sends + receives + exchanges; ++index) {
        if (cursor + hipc_map_alias_words * 4 > message.size()) return;
        const auto size_low = read_word(message, cursor);
        const auto low = read_word(message, cursor + 4);
        const auto packed = read_word(message, cursor + 8);
        cursor += hipc_map_alias_words * 4;
        const auto high = (packed >> hipc_map_alias_address_high_shift) & hipc_address_high_mask;
        const auto mid = (packed >> hipc_map_alias_address_mid_shift) & hipc_address_mid_mask;
        // A map-alias descriptor normally carries the complete address in its
        // packed word. Older clients leave the upper bits zero and rely on the
        // region base the runtime supplies, so only that zero-packed shape falls
        // back to the base.
        const auto address = high == 0 && mid == 0 && region_base != 0
                                 ? region_base | low
                                 : hipc_address(low, high, mid);
        if (size_low == 0 || address == 0 || buffers.full()) continue;
        buffers.add({.address = address, .size = size_low, .input = true, .output = true});
    }
    // Receive-list (C) entries follow the raw data unless the header gives
    // their word offset. Type 2 is one entry, a larger type is type - 2
    // entries; each holds the low address word, then the address bits above
    // 32 in its low half and the size in its high half.
    const auto receive_type = (second >> 10) & 0xfu;
    if (receive_type < 2) return;
    const auto raw_words = second & 0x3ffu;
    const auto list_offset = (second >> 20) & 0x7ffu;
    std::size_t entry = list_offset ? std::size_t{list_offset} * 4 : cursor + std::size_t{raw_words} * 4;
    const unsigned entries = receive_type == 2 ? 1u : receive_type - 2u;
    for (unsigned index = 0; index != entries; ++index, entry += 8) {
        if (entry + 8 > message.size()) return;
        const auto low = read_word(message, entry);
        const auto packed = read_word(message, entry + 4);
        const auto address = (std::uint64_t{packed & 0xffffu} << 32) | low;
        const auto size = packed >> 16;
        if (size == 0 || address == 0 || buffers.full()) continue;
        buffers.add({.address = address, .size = size, .input = false, .output = true});
    }
}
} // namespace

std::optional<std::size_t> locate_cmif(const IpcMessage& message) noexcept {
    constexpr std::uint32_t sfci = 0x49434653u;
    // A domain header at 16 or 32 puts the CMIF header sixteen bytes later.
    if (domain_header(message, 16) && read_word(message, 32) == sfci) return 32;
    if (domain_header(message, 32) && read_word(message, 48) == sfci) return 48;
    // A plain CMIF request keeps its header at the first of these offsets.
    if (read_word(message, 16) == sfci) return 16;
    if (read_word(message, 32) == sfci) return 32;
    if (read_word(message, 48) == sfci) return 48;
    return std::nullopt;
}

bool GuestBuffers::read(std::size_t index, std::span<std::byte> destination) const noexcept {
    if (index >= count_ || !reader_ || destination.size() > buffers_[index].size) return false;
    return reader_(context_, buffers_[index].address, destination);
}

bool GuestBuffers::write(std::size_t index, std::span<const std::byte> source) const noexcept {
    if (index >= count_ || !writer_ || source.size() > buffers_[index].size) return false;
    return writer_(context_, buffers_[index].address, source);
}

bool GuestBuffers::write_at(std::size_t index, std::size_t offset,
                            std::span<const std::byte> source) const noexcept {
    if (index >= count_ || !writer_) return false;
    const auto& buffer = buffers_[index];
    if (offset > buffer.size || source.size() > buffer.size - offset) return false;
    return writer_(context_, buffer.address + offset, source);
}

std::optional<std::string> GuestBuffers::read_string(std::size_t index) const {
    if (index >= count_ || !reader_) return std::nullopt;
    std::string text;
    text.reserve(buffers_[index].size);
    for (std::uint32_t offset = 0; offset < buffers_[index].size; ++offset) {
        std::byte value{};
        if (!reader_(context_, buffers_[index].address + offset, std::span<std::byte>(&value, 1)))
            return std::nullopt;
        if (value == std::byte{}) return text;
        text.push_back(static_cast<char>(std::to_integer<unsigned char>(value)));
    }
    return std::nullopt;
}

IpcDisposition IpcDispatcher::dispatch(std::uint16_t svc, QueryRegisters& registers,
                                      const IpcCallContext& caller) const {
    switch (svc) {
    case 0x1f: return connect(registers, caller);
    case 0x21: return send(registers, caller);
    default: return IpcDisposition::unsupported;
    }
}

IpcDisposition IpcDispatcher::connect(QueryRegisters& registers,
                                     const IpcCallContext& caller) const {
    if (!caller.read_byte || !connector_) return IpcDisposition::unsupported;
    const auto address = registers[1];
    std::array<char, 12> name{};
    std::size_t length = 0;
    for (; length != name.size(); ++length) {
        std::byte value;
        if (length > std::numeric_limits<std::uint64_t>::max() - address ||
            !caller.read_byte(caller.memory_context, address + length, value)) {
            finish(registers, invalid_copy, address);
            return IpcDisposition::handled;
        }
        name[length] = static_cast<char>(std::to_integer<unsigned char>(value));
        if (value == std::byte{}) break;
    }
    if (length == name.size()) {
        finish(registers, name_too_long, address);
        return IpcDisposition::handled;
    }
    auto connection = connector_(connector_context_, {name.data(), length});
    if (connection.disposition != IpcDisposition::handled) return connection.disposition;
    if (connection.result) {
        finish(registers, connection.result, address);
        return IpcDisposition::handled;
    }
    if (!connection.session) return IpcDisposition::invalid_backend;
    const auto handle = handles_->insert(std::move(connection.session));
    if (!handle) {
        finish(registers, out_of_handles, address);
        return IpcDisposition::handled;
    }
    finish(registers, 0, *handle);
    return IpcDisposition::handled;
}

IpcDisposition IpcDispatcher::send(QueryRegisters& registers,
                                  const IpcCallContext& caller) const {
    if (caller.message.size() < ipc_message_capacity) return IpcDisposition::invalid_backend;
    IpcMessage request;
    std::copy_n(caller.message.begin(), request.size(), request.begin());
    // HIPC Close (type 2) tears down the server side of the session. The client still owns
    // its handle and closes it with svcCloseHandle itself, so this dispatcher must not drop
    // the entry here: a second close would then fail with an invalid-handle result the guest
    // never asked for. No service sees the message, but the kernel validates the handle first.
    constexpr std::uint16_t close_request = 2;
    if ((read_word(request, 0) & 0xffffu) == close_request) {
        if (!handles_->lookup_as<ClientSession>(static_cast<Handle>(registers[0]))) {
            finish(registers, invalid_handle);
            return IpcDisposition::handled;
        }
        finish(registers, 0);
        return IpcDisposition::handled;
    }
    const auto session = handles_->lookup_as<ClientSession>(static_cast<Handle>(registers[0]));
    if (!session) {
        finish(registers, invalid_handle);
        return IpcDisposition::handled;
    }
    session->set_memory_mapper(caller.memory_mapper);
    // Buffer descriptors are not decoded yet: the request header layout for
    // HIPC buffers is not established, so every request reaches the session
    // without buffers and services keep their explicit unsupported paths.
    GuestBuffers buffers(caller.read_memory, caller.write_memory, caller.memory_context);
    // The PID variant keeps its domain header at byte32 and the CMIF header at
    // byte48, leaving bytes8..31 for buffer descriptors.
    const bool descriptor_variant = read_word(request, 48) == 0x49434653 &&
                                    (read_word(request, 32) & 0xFFFFu) == 1;
    // A request whose CMIF header follows the descriptor area directly keeps at most one
    // descriptor there; the PID variant has room for three.
    const bool standard_variant = read_word(request, 32) == 0x49434653;
    // A real counted HIPC packet may legitimately decode to no buffers (for
    // example a PID-only request or an empty X buffer). Its special header and
    // descriptor words must never be reinterpreted as legacy size/address pairs.
    const bool counted_transport = (read_word(request, 0) & 0xffff0000u) != 0 ||
                                   (read_word(request, 4) & 0x80000000u) != 0;
    if (caller.read_memory != nullptr) {
        // Counted descriptors take precedence for both layouts. The map-alias
        // descriptor is three words and its packed word carries the address bits
        // above 32; a zero packed word falls back to the region base for clients
        // that leave those bits implied.
        add_hipc_buffers(request, buffers, caller.buffer_region_base);
        // A PID-variant request that declares exactly one receive but writes only
        // the two-word {size, low} pair is the older file-system shape; the
        // counted walk already consumed those words, so this fallback only runs
        // when the walk produced no buffer.
        if (buffers.empty() && !counted_transport && descriptor_variant) {
            for (std::size_t offset = request_descriptor_area;
                 offset + 8 <= request_descriptor_limit && !buffers.full(); offset += 8) {
                const auto size = read_word(request, offset);
                const auto low = read_word(request, offset + 4);
                if (size == 0 || low == 0) break;
                buffers.add({.address = caller.buffer_region_base | low,
                             .size = size,
                             .input = true,
                             .output = true});
            }
        }
        if (buffers.empty() && !counted_transport && !descriptor_variant && standard_variant) {
            for (std::size_t offset = request_descriptor_area;
                 offset + 8 <= 16 && !buffers.full(); offset += 8) {
                const auto size = read_word(request, offset);
                const auto low = read_word(request, offset + 4);
                if (size == 0 || low == 0) break;
                buffers.add({.address = caller.buffer_region_base | low,
                             .size = size,
                             .input = true,
                             .output = true});
            }
        }
        // A domain request can also append its two-word descriptor after the CMIF header:
        // {u32 size, u32 address low bits} at cmif+16 or cmif+24. The account service's
        // user-list commands use that shape. The region base supplies the implied high bits.
        // Only a request whose HIPC header declares an inline descriptor (the buffer-C flag
        // in bits 10..13) is scanned; otherwise the trailing words are ordinary command data.
        const auto inline_descriptor = (read_word(request, 4) >> 10) & 0xfu;
        if (buffers.empty() && inline_descriptor != 0) {
            std::size_t cmif = 0;
            if (read_word(request, 16) == 0x49434653) cmif = 16;
            else if (read_word(request, 32) == 0x49434653) cmif = 32;
            else if (read_word(request, 48) == 0x49434653) cmif = 48;
            for (const auto offset : {cmif + 16, cmif + 24}) {
                if (cmif == 0 || offset + 8 > request.size() || buffers.full()) break;
                const auto size = read_word(request, offset);
                const auto low = read_word(request, offset + 4);
                if (size == 0 || low == 0) continue;
                // The first pair can be stale message content (for example a
                // previous SFCI header) when the real descriptor follows at the
                // next slot. A plausible guest buffer is far below this bound;
                // a magic-looking size is not a descriptor.
                if (size > 0x10000000u) continue;
                buffers.add({.address = caller.buffer_region_base | low,
                             .size = size,
                             .input = true,
                             .output = true});
                break;
            }
        }
    }
    std::array<std::shared_ptr<KernelObject>, 15> copied_objects{};
    std::size_t copied_count = 0;
    if (session->receives_copied_objects() && (read_word(request, 4) & 0x80000000u) != 0) {
        const auto special = read_word(request, 8);
        copied_count = (special >> hipc_special_copy_shift) & hipc_count_mask;
        const auto offset = 12u + ((special & hipc_special_pid) ? 8u : 0u);
        for (std::size_t index = 0; index < copied_count; ++index) {
            const auto copied_handle = read_word(request, offset + index * 4);
            // Pseudo-handles need process/thread context. Existing services such
            // as applet already resolve these from the raw request themselves.
            // Leave an explicit null entry instead of inventing a table object.
            if (copied_handle == current_process_handle || copied_handle == current_thread_handle)
                continue;
            copied_objects[index] = handles_->lookup(copied_handle);
            if (!copied_objects[index]) {
                finish(registers, invalid_handle);
                return IpcDisposition::handled;
            }
        }
    }
    auto reply = session->request_with_handles(request, caller.process_id, buffers,
        std::span{copied_objects}.first(copied_count));
    if (reply.disposition != IpcDisposition::handled) return reply.disposition;
    if (reply.size > reply.bytes.size() || (reply.transport_result && reply.size))
        return IpcDisposition::invalid_backend;
    if (reply.transferred_count > max_transferred_handles) return IpcDisposition::invalid_backend;
    if (reply.transferred_count == 0) {
        std::copy_n(reply.bytes.begin(), reply.size, caller.message.begin());
        finish(registers, reply.transport_result);
        return IpcDisposition::handled;
    }
    if (reply.size < 16) return IpcDisposition::invalid_backend;
    const auto transferred = reply.transferred_count;
    const std::size_t cmif = (handle_area + handle_entry * transferred + 15) & ~std::size_t{15};
    if (reply.size + (cmif - 16) > reply.bytes.size()) return IpcDisposition::invalid_backend;
    std::array<Handle, max_transferred_handles> handles{};
    std::size_t inserted = 0;
    for (; inserted != transferred; ++inserted) {
        const auto handle = handles_->insert(std::move(reply.transferred[inserted]));
        if (!handle) break;
        handles[inserted] = *handle;
    }
    if (inserted != transferred) {
        // Handle exhaustion is a host resource failure, not a guest Result:
        // report the dispatcher failure and leave the caller's message intact.
        for (std::size_t index = 0; index != inserted; ++index) handles_->close(handles[index]);
        return IpcDisposition::invalid_backend;
    }
    std::copy_backward(reply.bytes.begin() + 16, reply.bytes.begin() + reply.size,
                       reply.bytes.begin() + reply.size + (cmif - 16));
    std::fill(reply.bytes.begin() + 8, reply.bytes.begin() + cmif, std::byte{});
    reply.size += cmif - 16;
    for (std::size_t index = 0; index != transferred; ++index)
        write_word(reply.bytes, handle_area + handle_entry * index, handles[index]);
    write_word(reply.bytes, 0, 0);
    write_word(reply.bytes, 4,
               0x80000000u | static_cast<std::uint32_t>((reply.size - cmif) / 4));
    const auto count = static_cast<std::uint32_t>(transferred) & 0xfu;
    write_word(reply.bytes, 8,
               special_header_present |
                   (reply.transferred_as_copy ? count << copy_handle_shift
                                              : count << moved_handle_shift));
    write_word(reply.bytes, 12, 0);
    std::copy_n(reply.bytes.begin(), reply.size, caller.message.begin());
    finish(registers, reply.transport_result);
    return IpcDisposition::handled;
}

} // namespace armigo::horizon
