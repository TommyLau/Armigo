// ============================================================================
// Armigo Tests - Video Interface Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/video_interface.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <utility>
#include <string>
#include <string_view>

namespace {
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::horizon::IpcReply;
using armigo::services::vi::DisplaySession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

std::uint32_t get(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}
void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}

struct BinderBuffers {
    std::array<std::byte, 0x200> input{};
    std::array<std::byte, 0x200> output{};
    bool fail_read = false;
    std::size_t read_size = 0;
    unsigned writes = 0;
    std::size_t write_size = 0;
    bool fail_write = false;

    armigo::horizon::GuestBuffers buffers(std::uint32_t declared_size) {
        armigo::horizon::GuestBuffers result(
            [](void* raw, std::uint64_t address, std::span<std::byte> bytes) noexcept {
                auto& self = *static_cast<BinderBuffers*>(raw);
                self.read_size = bytes.size();
                if (self.fail_read || address != 0x1000 || bytes.size() > self.input.size())
                    return false;
                std::memcpy(bytes.data(), self.input.data(), bytes.size());
                return true;
            },
            [](void* raw, std::uint64_t address, std::span<const std::byte> bytes) noexcept {
                auto& self = *static_cast<BinderBuffers*>(raw);
                if (self.fail_write || address != 0x2000 || bytes.size() > self.output.size()) return false;
                std::memcpy(self.output.data(), bytes.data(), bytes.size());
                self.write_size = bytes.size();
                ++self.writes;
                return true;
            }, this);
        result.add({.address = 0x1000, .size = declared_size, .input = true});
        result.add({.address = 0x2000, .size = static_cast<std::uint32_t>(output.size()),
                    .output = true});
        return result;
    }

    void word(std::size_t offset, std::uint32_t value) {
        std::memcpy(input.data() + offset, &value, 4);
    }
    std::uint32_t answer() const {
        std::uint32_t value = 0;
        std::memcpy(&value, output.data() + 16, 4);
        return value;
    }
    std::size_t parcel(std::span<const std::uint32_t> arguments) {
        constexpr std::string_view token = "android.gui.IGraphicBufferProducer";
        input.fill(std::byte{});
        word(4, 16);
        word(20, static_cast<std::uint32_t>(token.size()));
        for (std::size_t index = 0; index < token.size(); ++index)
            input[24 + index * 2] = static_cast<std::byte>(token[index]);
        const std::size_t at = (24 + (token.size() + 1) * 2 + 3) & ~std::size_t{3};
        for (std::size_t index = 0; index < arguments.size(); ++index)
            word(at + index * 4, arguments[index]);
        word(0, static_cast<std::uint32_t>(at + arguments.size() * 4 - 16));
        return at;
    }
};

void binder_query_bounds(DisplaySession& relay, std::uint32_t width, std::uint32_t height) {
    IpcMessage query{};
    put(query, 0, 6);
    put(query, 16, 0x49434653);
    put(query, 20, 1);
    put(query, 24, 3);
    put(query, 32, 1);
    put(query, 36, 9);
    BinderBuffers data;
    const auto argument = data.parcel(std::array<std::uint32_t, 1>{0});
    for (const auto [key, expected] : {
             std::pair{0u, width}, {1u, height}, {2u, 1u}, {3u, 2u},
             {8u, 0u}, {10u, 0x800u}, {12u, 0u}}) {
        data.word(argument, key);
        auto buffers = data.buffers(static_cast<std::uint32_t>(argument + 4));
        const auto reply = relay.request_with_buffers(query, 5, buffers);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(data.read_size == argument + 4);
        CHECK(data.answer() == expected);
    }
    put(query, 36, 10);
    data.parcel(std::array<std::uint32_t, 0>{});
    auto connect_buffers = data.buffers(static_cast<std::uint32_t>(argument));
    CHECK(relay.request_with_buffers(query, 5, connect_buffers).disposition == IpcDisposition::handled);
    CHECK(data.answer() == width);
    std::uint32_t connect_height = 0;
    std::memcpy(&connect_height, data.output.data() + 20, 4);
    CHECK(connect_height == height);
    put(query, 36, 9);
    data.parcel(std::array<std::uint32_t, 1>{0});
    const auto completed_writes = data.writes;
    auto buffers = data.buffers(static_cast<std::uint32_t>(argument + 3));
    CHECK(relay.request_with_buffers(query, 5, buffers).disposition == IpcDisposition::unsupported);
    buffers = data.buffers(23);
    CHECK(relay.request_with_buffers(query, 5, buffers).disposition == IpcDisposition::unsupported);
    data.word(20, 0xffffffffu);
    buffers = data.buffers(static_cast<std::uint32_t>(argument + 4));
    CHECK(relay.request_with_buffers(query, 5, buffers).disposition == IpcDisposition::unsupported);
    data.parcel(std::array<std::uint32_t, 1>{99});
    CHECK(relay.request_with_buffers(query, 5, buffers).disposition == IpcDisposition::unsupported);
    data.word(argument, 0);
    data.fail_read = true;
    CHECK(relay.request_with_buffers(query, 5, buffers).disposition == IpcDisposition::unsupported);
    buffers = data.buffers(0);
    CHECK(relay.request_with_buffers(query, 5, buffers).disposition == IpcDisposition::unsupported);
    CHECK(data.writes == completed_writes);
}

void binder_preallocated_buffers(DisplaySession& relay) {
    struct Provider {
        armigo::services::GraphicsBufferBacking backing;
    };
    auto provider = std::make_shared<Provider>();
    auto owner = std::make_shared<int>(7);
    std::weak_ptr<void> lifetime = owner;
    // NvMap records allocation alignment independently of a supplied CPU VA.
    provider->backing = {77, 0x20000, 0x801000, 0x1000, owner};
    relay.set_buffer_backing_lookup([provider](std::uint32_t id) {
        return id == provider->backing.export_id
            ? std::optional{provider->backing}
            : std::optional<armigo::services::GraphicsBufferBacking>{};
    });
    IpcMessage message{};
    put(message, 0, 6); put(message, 16, 0x49434653); put(message, 20, 1);
    put(message, 24, 3); put(message, 32, 1);
    BinderBuffers data;
    std::array<std::byte, 0x16c> flat{};
    const auto flat_word = [&](std::size_t offset, std::uint32_t value) {
        std::memcpy(flat.data() + offset, &value, 4);
    };
    flat_word(0, 0x47424652); flat_word(4, 8); flat_word(8, 8);
    flat_word(0x0c, 8); flat_word(0x10, 1); flat_word(0x14, 0x800);
    flat_word(0x18, 0x11223344); flat_word(0x1c, 0x55667788);
    flat_word(0x20, 0); flat_word(0x24, 81); flat_word(0x2c, 77);
    flat_word(0x34, 0xdaffcaff); flat_word(0x40, 0x800); flat_word(0x44, 1);
    flat_word(0x48, 1); flat_word(0x4c, 8); flat_word(0x50, 0x200);
    flat_word(0x54, 1); flat_word(0x5c, 8); flat_word(0x60, 8);
    flat_word(0x64, 0x00532120); flat_word(0x68, 1); flat_word(0x6c, 3);
    flat_word(0x70, 64); flat_word(0x74, 0x91); flat_word(0x78, 0);
    flat_word(0x80, 0); flat_word(0x94, 0x200);
    flat_word(0x164, 0xdeadbeef); flat_word(0x168, 0xcafebabe);

    auto set_slot = [&](std::uint32_t slot, bool present) {
        put(message, 36, 14);
        const auto at = present
            ? data.parcel(std::array<std::uint32_t, 4>{slot, 1u, 0x16cu, 0u})
            : data.parcel(std::array<std::uint32_t, 2>{slot, 0u});
        std::size_t size = at + 8;
        if (present) {
            std::copy(flat.begin(), flat.end(), data.input.begin() + at + 16);
            size = at + 16 + flat.size();
            data.word(0, static_cast<std::uint32_t>(size - 16));
        }
        auto buffers = data.buffers(static_cast<std::uint32_t>(size));
        return relay.request_with_buffers(message, 5, buffers);
    };
    flat_word(0x2c, 78);
    CHECK(set_slot(0, true).disposition == IpcDisposition::unsupported);
    flat_word(0x2c, 77);
    provider->backing.size = 0x100;
    CHECK(set_slot(0, true).disposition == IpcDisposition::unsupported);
    provider->backing.size = 0x1000;
    flat_word(0x98, 1);
    CHECK(set_slot(0, true).disposition == IpcDisposition::unsupported);
    flat_word(0x98, 0);
    flat_word(4, 9);
    CHECK(set_slot(0, true).disposition == IpcDisposition::unsupported);
    flat_word(4, 8);
    flat_word(0x94, 0x100);
    CHECK(set_slot(0, true).disposition == IpcDisposition::unsupported);
    flat_word(0x94, 0x200);
    CHECK(set_slot(0, true).disposition == IpcDisposition::handled);
    CHECK(data.write_size == 0x14);
    flat_word(0x50, 0x2000); flat_word(0x94, 0x2000);
    CHECK(set_slot(0, true).disposition == IpcDisposition::unsupported);
    flat_word(0x50, 0x200); flat_word(0x94, 0x200);
    data.fail_write = true;
    CHECK(set_slot(0, true).disposition == IpcDisposition::unsupported);
    data.fail_write = false;

    // Declaring the parcel one byte short must reject the otherwise complete
    // registration and preserve the installed owner/state.
    put(message, 36, 14);
    auto truncated_at = data.parcel(std::array<std::uint32_t, 4>{0, 1, 0x16c, 0});
    std::copy(flat.begin(), flat.end(), data.input.begin() + truncated_at + 16);
    const auto truncated_size = truncated_at + 16 + flat.size();
    data.word(0, static_cast<std::uint32_t>(truncated_size - 17));
    auto truncated_buffers = data.buffers(static_cast<std::uint32_t>(truncated_size));
    CHECK(relay.request_with_buffers(message, 5, truncated_buffers).disposition ==
          IpcDisposition::unsupported);

    IpcMessage clone_message{};
    put(clone_message, 0, 7); put(clone_message, 16, 0x49434653); put(clone_message, 24, 2);
    auto clone_reply = relay.request(clone_message, 5);
    auto clone = std::dynamic_pointer_cast<DisplaySession>(clone_reply.transferred[0]);
    CHECK(clone != nullptr);

    put(message, 36, 10);
    const auto connect_at = data.parcel(std::array<std::uint32_t, 0>{});
    auto connect_buffers = data.buffers(static_cast<std::uint32_t>(connect_at));
    CHECK(clone && clone->request_with_buffers(message, 5, connect_buffers).disposition ==
          IpcDisposition::handled);
    CHECK(data.answer() == 8);
    std::uint32_t connect_height = 0;
    std::memcpy(&connect_height, data.output.data() + 20, 4);
    CHECK(connect_height == 8);

    put(message, 36, 9);
    for (const auto [selector, expected] :
         {std::pair{0u, 8u}, {1u, 8u}, {2u, 1u}}) {
        auto query_at = data.parcel(std::array<std::uint32_t, 1>{selector});
        auto query_buffers = data.buffers(static_cast<std::uint32_t>(query_at + 4));
        CHECK(clone && clone->request_with_buffers(message, 5, query_buffers).disposition ==
              IpcDisposition::handled);
        CHECK(data.answer() == expected);
    }

    // Readiness includes free holes below the active maximum, even though
    // dequeue of such a hole reports that server allocation is unavailable.
    CHECK(set_slot(1, true).disposition == IpcDisposition::handled);
    auto native_message = message;
    put(native_message, 24, 2);
    const auto native_reply = relay.request(native_message, 5);
    const auto free_event = std::dynamic_pointer_cast<armigo::horizon::Event>(native_reply.transferred[0]);
    CHECK(free_event && free_event->is_signaled());
    for (unsigned slot = 0; slot < 2; ++slot) {
        put(message, 36, 3);
        const auto dequeue_at = data.parcel(std::array<std::uint32_t, 5>{0, 0, 0, 0, 0});
        const auto dequeue_buffers = data.buffers(static_cast<std::uint32_t>(dequeue_at + 20));
        CHECK(relay.request_with_buffers(message, 5, dequeue_buffers).disposition == IpcDisposition::handled);
    }
    CHECK(free_event && !free_event->is_signaled());
    CHECK(set_slot(0, false).disposition == IpcDisposition::handled);
    CHECK(free_event && free_event->is_signaled());
    CHECK(set_slot(0, true).disposition == IpcDisposition::handled);
    CHECK(set_slot(1, false).disposition == IpcDisposition::handled);

    put(message, 36, 3);
    auto at = data.parcel(std::array<std::uint32_t, 5>{0, 0, 0, 0, 0});
    auto buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
    CHECK(clone && clone->request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);
    CHECK(data.write_size == 0x48 && data.answer() == 0);
    std::uint32_t dequeue_status = 1;
    std::memcpy(&dequeue_status, data.output.data() + 0x44, 4);
    CHECK(dequeue_status == 0);

    put(message, 36, 1);
    at = data.parcel(std::array<std::uint32_t, 1>{0});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 4));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);
    CHECK(data.write_size == 0x18c);
    CHECK(std::equal(flat.begin(), flat.begin() + 0x164, data.output.begin() + 0x1c));
    CHECK(std::all_of(data.output.begin() + 0x180, data.output.begin() + 0x188,
                      [](std::byte value) { return value == std::byte{}; }));

    put(message, 36, 7);
    std::array<std::uint32_t, 24> queued{};
    queued[0] = 0; queued[1] = 84; queued[2] = 0;
    queued[5] = 1;
    queued[12] = 8;
    queued[14] = 2;
    queued[15] = 1;
    queued[16] = 24; queued[17] = 3;
    queued[18] = queued[20] = queued[22] = 0xffffffffu;
    at = data.parcel(queued);
    buffers = data.buffers(static_cast<std::uint32_t>(at + queued.size() * 4));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::unsupported);
    bool presented = false;
    relay.set_presenter([&](const armigo::services::GraphicsPresentation& presentation) {
        presented = true;
        CHECK(presentation.slot == 0 && presentation.frame_number == 1);
        CHECK(presentation.width == 8 && presentation.height == 8 && presentation.pitch == 64);
        CHECK(presentation.surface_offset == 0 && presentation.surface_size == 0x200);
        CHECK(presentation.auto_timestamp && presentation.swap_interval == 2);
        CHECK(presentation.fence_count == 1);
        CHECK(presentation.producer_fences[0].syncpoint == 24 &&
              presentation.producer_fences[0].threshold == 3);
        return armigo::services::GraphicsPresentationResult{true, {}};
    });
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);
    CHECK(presented && data.write_size == 0x2c);
    std::uint32_t hint = 0, pending = 0, frame = 0;
    std::memcpy(&hint, data.output.data() + 0x18, 4);
    std::memcpy(&pending, data.output.data() + 0x1c, 4);
    std::memcpy(&frame, data.output.data() + 0x20, 4);
    CHECK(hint == 0x20 && pending == 1 && frame == 1);

    put(message, 36, 3);
    at = data.parcel(std::array<std::uint32_t, 5>{0, 8, 8, 1, 0});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);
    put(message, 36, 1);
    at = data.parcel(std::array<std::uint32_t, 1>{0});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 4));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);

    put(message, 36, 8);
    std::array<std::uint32_t, 12> cancel{};
    cancel[0] = 0; cancel[1] = 0x24;
    at = data.parcel(cancel);
    buffers = data.buffers(static_cast<std::uint32_t>(at + cancel.size() * 4));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);
    CHECK(data.write_size == 0x10);

    put(message, 36, 1);
    at = data.parcel(std::array<std::uint32_t, 1>{0});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 4));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::unsupported);

    put(message, 36, 3);
    at = data.parcel(std::array<std::uint32_t, 5>{0, 8, 8, 1, 0});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);

    // A high slot raises the scan bound. With slot zero still dequeued, the
    // lower empty hole is selected and reports allocation-disabled instead of
    // being skipped in favor of slot three.
    flat_word(0x78, 0x200);
    CHECK(set_slot(3, true).disposition == IpcDisposition::handled);
    flat_word(0x78, 0);
    put(message, 36, 3);
    at = data.parcel(std::array<std::uint32_t, 5>{0, 8, 8, 1, 0});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);
    std::memcpy(&dequeue_status, data.output.data() + 0x44, 4);
    CHECK(static_cast<std::int32_t>(dequeue_status) == -12);

    CHECK(set_slot(3, false).disposition == IpcDisposition::handled);
    auto second_owner = std::make_shared<int>(9);
    std::weak_ptr<void> second_lifetime = second_owner;
    provider->backing.owner = second_owner;
    CHECK(set_slot(1, true).disposition == IpcDisposition::handled);
    put(message, 36, 3);
    at = data.parcel(std::array<std::uint32_t, 5>{0, 0, 0, 0, 0});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);
    CHECK(data.answer() == 1);
    put(message, 36, 1);
    at = data.parcel(std::array<std::uint32_t, 1>{1});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 4));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::handled);
    auto second_clone_reply = relay.request(clone_message, 5);
    auto second_clone = std::dynamic_pointer_cast<DisplaySession>(second_clone_reply.transferred[0]);
    CHECK(second_clone != nullptr);

    std::uint64_t failed_frame = 0;
    std::uint32_t failed_slot = 99, failed_width = 0, failed_height = 0;
    std::uint32_t failed_scaling = 0, failed_transform = 0, failed_sticky = 0;
    std::int32_t failed_swap = 0;
    const std::string long_detail = "synthetic failure " + std::string(600, 'x');
    relay.set_presenter([&](const armigo::services::GraphicsPresentation& presentation) {
        failed_frame = presentation.frame_number;
        failed_slot = presentation.slot;
        failed_width = presentation.width;
        failed_height = presentation.height;
        failed_scaling = presentation.scaling_mode;
        failed_transform = presentation.transform;
        failed_sticky = presentation.sticky_transform;
        failed_swap = presentation.swap_interval;
        return armigo::services::GraphicsPresentationResult{false, long_detail};
    });
    auto failed = queued;
    failed[6] = 1; failed[7] = 2; failed[8] = 5; failed[9] = 6;
    failed[10] = 3; failed[11] = 6; failed[14] = 2;
    put(message, 36, 7);
    at = data.parcel(failed);
    buffers = data.buffers(static_cast<std::uint32_t>(at + failed.size() * 4));
    CHECK(clone && clone->request_with_buffers(message, 5, buffers).disposition ==
                       IpcDisposition::unsupported);
    const char* failure_diagnostic = clone ? clone->diagnostic() : nullptr;
    CHECK(failure_diagnostic != nullptr);
    std::string first_diagnostic;
    if (failure_diagnostic != nullptr) {
        const std::string_view text{failure_diagnostic};
        first_diagnostic = text;
        CHECK(text.size() == 512);
        CHECK(failure_diagnostic[text.size()] == '\0');
        CHECK(text.find("frame_number=" + std::to_string(failed_frame)) != std::string_view::npos);
        CHECK(text.find("slot=" + std::to_string(failed_slot)) != std::string_view::npos);
        CHECK(text.find("width=" + std::to_string(failed_width)) != std::string_view::npos);
        CHECK(text.find("height=" + std::to_string(failed_height)) != std::string_view::npos);
        CHECK(text.find("crop=(1,2,5,6)") != std::string_view::npos);
        CHECK(text.find("scaling_mode=" + std::to_string(failed_scaling)) != std::string_view::npos);
        CHECK(text.find("transform=" + std::to_string(failed_transform)) != std::string_view::npos);
        CHECK(text.find("sticky_transform=" + std::to_string(failed_sticky)) != std::string_view::npos);
        CHECK(text.find("swap_interval=" + std::to_string(failed_swap)) != std::string_view::npos);
        CHECK(text.find("synthetic failure") != std::string_view::npos);
        CHECK(text.back() == 'x');
    }
    // A throwing presenter reports the same request metadata in its session
    // diagnostic, and its frame remains ACQUIRED and pinned.
    std::uint64_t thrown_frame = 0;
    std::uint32_t thrown_slot = 99;
    std::uint32_t thrown_scaling = 0, thrown_transform = 0, thrown_sticky = 0;
    std::int32_t thrown_swap = 0;
    relay.set_presenter([&](const armigo::services::GraphicsPresentation& presentation)
                            -> armigo::services::GraphicsPresentationResult {
        thrown_frame = presentation.frame_number;
        thrown_slot = presentation.slot;
        thrown_scaling = presentation.scaling_mode;
        thrown_transform = presentation.transform;
        thrown_sticky = presentation.sticky_transform;
        thrown_swap = presentation.swap_interval;
        throw 42;
    });
    auto thrown = queued;
    thrown[0] = 1;
    thrown[6] = 7; thrown[7] = 4; thrown[8] = 8; thrown[9] = 8;
    thrown[10] = 4; thrown[11] = 1; thrown[12] = 0; thrown[14] = 3;
    put(message, 36, 7);
    at = data.parcel(thrown);
    buffers = data.buffers(static_cast<std::uint32_t>(at + thrown.size() * 4));
    // Observe clone A while clone B formats a different failure diagnostic.
    std::atomic<bool> reader_started = false;
    std::atomic<bool> keep_reading = true;
    std::atomic<bool> first_diagnostic_stable = true;
    std::thread diagnostic_reader([&] {
        reader_started.store(true, std::memory_order_release);
        while (keep_reading.load(std::memory_order_acquire)) {
            const char* current = clone ? clone->diagnostic() : nullptr;
            if (current == nullptr || std::string_view{current} != first_diagnostic)
                first_diagnostic_stable.store(false, std::memory_order_relaxed);
        }
    });
    while (!reader_started.load(std::memory_order_acquire)) std::this_thread::yield();
    CHECK(second_clone && second_clone->request_with_buffers(message, 5, buffers).disposition ==
                              IpcDisposition::unsupported);
    keep_reading.store(false, std::memory_order_release);
    diagnostic_reader.join();
    CHECK(first_diagnostic_stable.load(std::memory_order_relaxed));
    CHECK(clone && clone->diagnostic() != nullptr);
    if (clone && clone->diagnostic() != nullptr)
        CHECK(std::string_view{clone->diagnostic()} == first_diagnostic);
    const char* exception_diagnostic = second_clone ? second_clone->diagnostic() : nullptr;
    CHECK(exception_diagnostic != failure_diagnostic);
    CHECK(exception_diagnostic != nullptr);
    if (exception_diagnostic != nullptr) {
        const std::string_view text{exception_diagnostic};
        CHECK(text.size() < 512);
        CHECK(text.find("frame_number=" + std::to_string(thrown_frame)) != std::string_view::npos);
        CHECK(text.find("slot=" + std::to_string(thrown_slot)) != std::string_view::npos);
        CHECK(text.find("crop=(7,4,8,8)") != std::string_view::npos);
        CHECK(text.find("scaling_mode=" + std::to_string(thrown_scaling)) != std::string_view::npos);
        CHECK(text.find("transform=" + std::to_string(thrown_transform)) != std::string_view::npos);
        CHECK(text.find("sticky_transform=" + std::to_string(thrown_sticky)) != std::string_view::npos);
        CHECK(text.find("swap_interval=" + std::to_string(thrown_swap)) != std::string_view::npos);
        CHECK(text.find("Synchronous presenter threw an exception") != std::string_view::npos);
    }
    // No free slot remains: the throwing frame was not recycled.
    put(message, 36, 3);
    at = data.parcel(std::array<std::uint32_t, 5>{0, 0, 0, 0, 0});
    buffers = data.buffers(static_cast<std::uint32_t>(at + 20));
    CHECK(relay.request_with_buffers(message, 5, buffers).disposition == IpcDisposition::unsupported);
    provider->backing.owner.reset();
    owner.reset();
    second_owner.reset();
    CHECK(!lifetime.expired());
    CHECK(!second_lifetime.expired());
    CHECK(set_slot(0, false).disposition == IpcDisposition::unsupported);
    CHECK(!lifetime.expired());
    CHECK(set_slot(1, false).disposition == IpcDisposition::unsupported);
    CHECK(!second_lifetime.expired());
    CHECK(set_slot(1, true).disposition == IpcDisposition::unsupported);
}
}  // namespace

int test_mode(armigo::services::OperationMode mode) {
    armigo::horizon::HandleTable handles{4};
    DisplaySession session(handles, mode);
    IpcMessage message{};
    message.fill(std::byte{0});
    put(message, 0, 7);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 24, 3);
    auto reply = session.request(message, 5);
    CHECK(reply.disposition == IpcDisposition::handled);
    put(message, 24, 0);
    reply = session.request(message, 5);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(session.domain_object_count() == 1);

    // The root session hands out the display service (command 1), which answers the
    // display and binder commands.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 9);
    put(message, 12, 0xb8015);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, 1);
    reply = session.request(message, 5);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.transferred_count == 1);
    auto display_service = std::dynamic_pointer_cast<DisplaySession>(reply.transferred[0]);
    CHECK(display_service != nullptr);
    CHECK(get(reply.bytes, 16) == 0x4f434653);
    if (display_service == nullptr) return 1;

    // The application service reports compatibility geometry even in console
    // mode. The firmware does not validate or select by the supplied display ID.
    for (const auto type : {4u, 6u}) {
        for (const auto display_id : {0u, 1u, 0xffffffffu}) {
            IpcMessage resolution{};
            put(resolution, 0, type);
            put(resolution, 4, 10);
            put(resolution, 16, 0x49434653);
            put(resolution, 24, 1102);
            put(resolution, 32, display_id);
            put(resolution, 36, 0xffffffffu);
            const auto resolution_reply = display_service->request(resolution, 5);
            CHECK(resolution_reply.disposition == IpcDisposition::handled);
            IpcMessage expected{};
            put(expected, 4, 10);
            put(expected, 16, 0x4f434653);
            put(expected, 28, 0xadaf6cad);
            put(expected, 32, 1280);
            put(expected, 40, 720);
            CHECK(resolution_reply.size == 48 && resolution_reply.bytes == expected);
            CHECK(resolution_reply.transferred_count == 0);
            CHECK(session.request(resolution, 5).disposition == IpcDisposition::unsupported);
            IpcMessage clone_request{};
            put(clone_request, 0, 7);
            put(clone_request, 16, 0x49434653);
            put(clone_request, 24, 2);
            const auto clone_reply = display_service->request(clone_request, 5);
            auto clone = std::dynamic_pointer_cast<DisplaySession>(clone_reply.transferred[0]);
            CHECK(clone && clone->request(resolution, 5).bytes == expected);
            for (const auto extent : {7u, 0x8000000au, 0xffffffffu}) {
                put(resolution, 4, extent);
                CHECK(display_service->request(resolution, 5).disposition == IpcDisposition::unsupported);
            }
        }
    }

    // GetDisplayVsyncEvent (command 5202) copies the display's readable event. A short
    // refresh interval keeps the test fast; the event is signaled by the source thread.
    {
        auto fast = std::make_shared<armigo::services::vi::VsyncSource>(
            std::chrono::milliseconds(2));
        display_service->set_vsync_source(fast);
        message.fill(std::byte{0});
        put(message, 0, 6);
        put(message, 4, 9);
        put(message, 16, 0x49434653);
        put(message, 20, 1);
        put(message, 24, 5202);
        put(message, 32, 0);
        put(message, 36, 0);
        reply = display_service->request(message, 5);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(reply.transferred_count == 1);
        CHECK(reply.transferred_as_copy);
        CHECK(get(reply.bytes, 16) == 0x4f434653);
        const auto event = std::dynamic_pointer_cast<armigo::horizon::Event>(reply.transferred[0]);
        CHECK(event != nullptr);
        if (event != nullptr) {
            for (unsigned spin = 0; spin != 200 && !event->is_signaled(); ++spin)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            CHECK(event->is_signaled());
        }
        // A second request returns the same per-display event.
        reply = display_service->request(message, 5);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(reply.transferred[0] == event);
        display_service->set_vsync_source(nullptr);
    }

    // GetRelayService (command 100) hands the client the binder driver as a session of its
    // own; that session answers the binder commands.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 9);
    put(message, 12, 0xb8015);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, 100);
    reply = display_service->request(message, 5);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.transferred_count == 1);
    auto relay = std::dynamic_pointer_cast<DisplaySession>(reply.transferred[0]);
    CHECK(relay != nullptr);
    if (relay == nullptr) return 1;
    binder_query_bounds(*relay, mode == armigo::services::OperationMode::Console ? 1920 : 1280,
                        mode == armigo::services::OperationMode::Console ? 1080 : 720);
    binder_preallocated_buffers(*relay);
    // AdjustRefcount is acknowledged and GetNativeHandle transfers a readable event.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, 1);
    reply = relay->request(message, 5);
    CHECK(reply.disposition == IpcDisposition::handled);
    put(message, 24, 2);
    reply = relay->request(message, 5);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.transferred_count == 1);
    // A producer transaction without the caller's buffers stays explicit.
    put(message, 24, 3);
    put(message, 32, 1);   // binder id
    put(message, 36, 10);  // connect
    reply = relay->request(message, 5);
    CHECK(reply.disposition == IpcDisposition::unsupported);

    // OpenDisplay over the application display service (command 1010) transfers the
    // display session and answers with a display id: the client reads a zero result
    // and the id from the words after the interface word.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, 1010);
    std::memcpy(message.data() + 32, "Default", 8);
    reply = session.request(message, 5);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.transferred_count == 1);
    CHECK(get(reply.bytes, 16) == 0x4f434653);
    CHECK(get(reply.bytes, 20) == 1);
    CHECK(get(reply.bytes, 24) == 0);
    CHECK(get(reply.bytes, 28) == 0);
    const auto display_id = get(reply.bytes, 32);
    CHECK(display_id != 0);
    CHECK(get(reply.bytes, 36) == 0);
    // A second display gets its own id.
    reply = session.request(message, 5);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 32) == display_id + 1);

    // OpenLayer (command 2020) fills the caller's buffer with a parcel that names
    // the layer's producer and answers with the written length. The client's own
    // parser reads a 40-byte binder object from the parcel's data region and a
    // four-byte object table entry pointing at it; the reported length covers
    // exactly header, data and table.
    {
        std::array<std::byte, 0x100> layer_buffer{};
        armigo::horizon::GuestBuffers buffers(
            [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
                auto* bytes = static_cast<std::array<std::byte, 0x100>*>(raw);
                const auto offset = address - reinterpret_cast<std::uint64_t>(bytes->data());
                if (offset > bytes->size() || out.size() > bytes->size() - offset) return false;
                std::memcpy(out.data(), bytes->data() + offset, out.size());
                return true;
            },
            [](void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
                auto* bytes = static_cast<std::array<std::byte, 0x100>*>(raw);
                const auto offset = address - reinterpret_cast<std::uint64_t>(bytes->data());
                if (offset > bytes->size() || in.size() > bytes->size() - offset) return false;
                std::memcpy(bytes->data() + offset, in.data(), in.size());
                return true;
            },
            &layer_buffer);
        buffers.add({.address = reinterpret_cast<std::uint64_t>(layer_buffer.data()),
                     .size = static_cast<std::uint32_t>(layer_buffer.size()),
                     .input = true,
                     .output = true});
        message.fill(std::byte{0});
        put(message, 0, 6);
        put(message, 4, 12);
        put(message, 16, 0x49434653);
        put(message, 20, 1);
        put(message, 24, 2020);
        std::memcpy(message.data() + 32, "Default", 8);
        const std::uint64_t layer_id = 1;
        std::memcpy(message.data() + 32 + 0x40, &layer_id, 8);
        reply = session.request_with_buffers(message, 5, buffers);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(get(reply.bytes, 16) == 0x4f434653);
        CHECK(get(reply.bytes, 24) == 0);
        CHECK(get(reply.bytes, 32) == 0x3c);
        const auto* words = reinterpret_cast<const std::uint32_t*>(layer_buffer.data());
        CHECK(words[0] == 0x28);    // data bytes
        CHECK(words[1] == 0x10);    // data offset
        CHECK(words[2] == 0x04);    // object table bytes
        CHECK(words[3] == 0x38);    // object table offset
        CHECK(words[4] == 2);       // handle object
        CHECK(words[5] == 0x7f);    // proxy binder flags
        CHECK(std::memcmp(layer_buffer.data() + 0x28, "dispdrv", 8) == 0);
        CHECK(words[6] == 1);       // producer handle
        CHECK(words[14] == 0);      // object 0 sits at data offset zero
        CHECK(words[15] == 0);      // nothing written past the 0x3c-byte parcel
        // Without the caller's buffer the command stays explicit.
        reply = session.request(message, 5);
        CHECK(reply.disposition == IpcDisposition::unsupported);
    }

    // A command without a name is refused instead of inventing a display.
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, 3);
    reply = session.request(message, 5);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    // An unknown display command stays explicit.
    put(message, 40, 99);
    std::memcpy(message.data() + 48, "Default", 8);
    reply = session.request(message, 5);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    CHECK(session.diagnostic() != nullptr);

    // A failed refresh source cannot become a successful never-signaled event.
    session.set_vsync_source(std::make_shared<armigo::services::vi::VsyncSource>(
        std::chrono::nanoseconds{0}));
    message.fill(std::byte{});
    put(message, 0, 6);
    put(message, 16, 0x49434653);
    put(message, 24, 5202);
    put(message, 32, 1);
    reply = session.request(message, 5);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    CHECK(reply.transferred_count == 0);
    CHECK(session.diagnostic() != nullptr);

    if (failures == 0) std::fprintf(stderr, "Video interface service failures: 0\n");
    return failures == 0 ? 0 : 1;
}

int main() {
    test_mode(armigo::services::OperationMode::Console);
    test_mode(armigo::services::OperationMode::Handheld);
    return failures == 0 ? 0 : 1;
}
