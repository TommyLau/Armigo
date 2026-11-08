// ============================================================================
// Armigo Tests - Audio Renderer Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>
#include <armigo/services/audio_renderer.h>

#include <array>

#include <cstdio>
#include <cstring>
#include <memory>
#include <span>

namespace {
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::services::audren::AudioDeviceSession;
using armigo::services::audren::ManagerSession;
using armigo::services::audren::RendererSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}

void put64(IpcMessage& message, std::size_t offset, std::uint64_t value) {
    put(message, offset, static_cast<std::uint32_t>(value));
    put(message, offset + 4, static_cast<std::uint32_t>(value >> 32));
}

std::uint32_t get(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}

IpcMessage request(std::uint32_t command, std::size_t cmif = 16) {
    IpcMessage message{};
    message.fill(std::byte{0});
    put(message, 0, 6);
    put(message, 4, 0x15);
    put(message, cmif, 0x49434653);
    put(message, cmif + 4, 1);
    put(message, cmif + 8, command);
    return message;
}

void put_parameter(IpcMessage& message, std::size_t offset, std::uint32_t sample_rate,
                   std::uint32_t sample_count, std::uint32_t mixes, std::uint32_t revision) {
    put(message, offset + 0x00, sample_rate);
    put(message, offset + 0x04, sample_count);
    put(message, offset + 0x08, mixes);
    put(message, offset + 0x10, 1);  // voices
    put(message, offset + 0x14, 1);  // sinks
    put(message, offset + 0x30, revision);
}

// "REV" followed by the revision digit, as the audren client sends it.
constexpr std::uint32_t rev15 = 0x3f564552;

// Pins any range; records the calls so the test can check their pairing.
struct RecordingMapper {
    unsigned maps = 0;
    unsigned unmaps = 0;
    static bool map(void* raw, std::uint64_t, std::uint64_t) noexcept {
        ++static_cast<RecordingMapper*>(raw)->maps;
        return true;
    }
    static void unmap(void* raw, std::uint64_t, std::uint64_t) noexcept {
        ++static_cast<RecordingMapper*>(raw)->unmaps;
    }
};

}  // namespace

int main() {
    ManagerSession manager(17);
    manager.set_render_threads(false);
    // GetWorkBufferSize returns the renderer's work area size as a u64 in the
    // CMIF output at byte 32: nonzero and page aligned.
    auto message = request(1);
    put_parameter(message, 32, 48000, 240, 3, rev15);
    auto reply = manager.request(message, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.size == 40);
    CHECK(get(reply.bytes, 16) == 0x4f434653);
    CHECK(get(reply.bytes, 24) == 0);
    const std::uint64_t work_buffer_size =
        get(reply.bytes, 32) | (std::uint64_t{get(reply.bytes, 36)} << 32);
    CHECK(work_buffer_size != 0 && work_buffer_size % 0x1000 == 0);

    // A revision newer than the renderer knows is refused with the audio
    // module's invalid-revision result.
    message = request(1);
    put_parameter(message, 32, 48000, 240, 2, 0x40564552);
    reply = manager.request(message, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 24) == (153u | (1537u << 9)));

    // OpenAudioRenderer with too small a transfer memory fails without an object.
    message = request(0);
    put_parameter(message, 32, 48000, 240, 3, rev15);
    put64(message, 32 + 0x38, work_buffer_size - 0x1000);
    reply = manager.request(message, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.transferred_count == 0);
    CHECK(get(reply.bytes, 24) == (153u | (4u << 9)));

    // OpenAudioRenderer returns a renderer object as a transferred handle.
    message = request(0);
    put_parameter(message, 32, 48000, 240, 3, rev15);
    put64(message, 32 + 0x38, work_buffer_size);
    reply = manager.request(message, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.transferred_count == 1);
    auto renderer = std::dynamic_pointer_cast<RendererSession>(reply.transferred[0]);
    CHECK(renderer != nullptr);
    if (renderer == nullptr) return 1;

    // The renderer reports its parameter and start/stop state.
    auto renderer_request = request(0);
    reply = renderer->request(renderer_request, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 32) == 48000);
    renderer_request = request(1);
    reply = renderer->request(renderer_request, 17);
    CHECK(get(reply.bytes, 32) == 240);
    renderer_request = request(2);
    reply = renderer->request(renderer_request, 17);
    CHECK(get(reply.bytes, 32) == 3);
    renderer_request = request(3);
    reply = renderer->request(renderer_request, 17);
    CHECK(get(reply.bytes, 32) == 1);
    renderer_request = request(5);
    reply = renderer->request(renderer_request, 17);
    CHECK(reply.disposition == IpcDisposition::handled && renderer->started());
    reply = renderer->request(renderer_request = request(3), 17);
    CHECK(get(reply.bytes, 32) == 0);

    // The rendering time limit and voice drop parameter round-trip.
    renderer_request = request(8);
    put(renderer_request, 32, 75);
    CHECK(renderer->request(renderer_request, 17).disposition == IpcDisposition::handled);
    reply = renderer->request(request(9), 17);
    CHECK(get(reply.bytes, 32) == 75);
    renderer_request = request(12);
    const float drop = 0.25f;
    std::memcpy(renderer_request.data() + 32, &drop, sizeof drop);
    CHECK(renderer->request(renderer_request, 17).disposition == IpcDisposition::handled);
    reply = renderer->request(request(13), 17);
    float read_drop = 0.0f;
    std::memcpy(&read_drop, reply.bytes.data() + 32, sizeof read_drop);
    CHECK(read_drop == drop);

    // The renderer event is transferred as a copy handle so the session keeps
    // ownership; each rendered frame signals it.
    renderer_request = request(7);
    reply = renderer->request(renderer_request, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.transferred_count == 1 && reply.transferred_as_copy);
    auto event = std::dynamic_pointer_cast<armigo::horizon::Event>(reply.transferred[0]);
    CHECK(event != nullptr);
    if (event == nullptr) return 1;
    CHECK(!event->is_signaled());
    renderer->render_frame();
    CHECK(event->is_signaled());
    CHECK(renderer->rendered_frames() == 1);

    // RequestUpdate without the two update buffers stays explicit.
    auto update = request(4, 48);
    armigo::horizon::GuestBuffers no_buffers;
    reply = renderer->request_with_buffers(update, 17, no_buffers);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    CHECK(renderer->diagnostic() != nullptr);

    // RequestUpdate through the session: a synthetic revision 15 update with
    // one memory pool attachment, one unused voice and sink, and an empty
    // dirty-mix list. The reply carries the result; the status buffer gets the
    // update header and the per-section statuses in the guest's order.
    {
        struct UpdateContext {
            std::array<std::byte, 0x1000> input{};
            std::array<std::byte, 0x1000> output{};
            static bool read(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
                auto& self = *static_cast<UpdateContext*>(raw);
                if (address + out.size() > self.input.size()) return false;
                std::memcpy(out.data(), self.input.data() + address, out.size());
                return true;
            }
            static bool write(void* raw, std::uint64_t address,
                              std::span<const std::byte> in) noexcept {
                auto& self = *static_cast<UpdateContext*>(raw);
                if (address < 0x10000 || address - 0x10000 + in.size() > self.output.size())
                    return false;
                std::memcpy(self.output.data() + (address - 0x10000), in.data(), in.size());
                return true;
            }
        } context;
        const auto input_word = [&](std::size_t offset, std::uint32_t value) {
            std::memcpy(context.input.data() + offset, &value, 4);
        };
        constexpr std::uint32_t pools = 4 * 0x20, resources = 0x70, voices = 0x188;
        constexpr std::uint32_t mixes = 0x20, sinks = 0x140, performance = 0x10;
        constexpr std::uint32_t total = 0x40 + 0x10 + pools + resources + voices + mixes + sinks + performance;
        input_word(0x00, rev15);
        input_word(0x04, 0x10);
        input_word(0x08, pools);
        input_word(0x0C, voices);
        input_word(0x10, resources);
        input_word(0x18, mixes);
        input_word(0x1C, sinks);
        input_word(0x20, performance);
        input_word(0x3C, total);
        input_word(0x40, rev15);
        // Memory pool 0: RequestAttach of two pages at a page-aligned address.
        input_word(0x50, 0x200000);
        input_word(0x58, 0x2000);
        input_word(0x60, 4);
        RecordingMapper mapper;
        renderer->set_memory_mapper({RecordingMapper::map, RecordingMapper::unmap, &mapper});
        armigo::horizon::GuestBuffers buffers(UpdateContext::read, UpdateContext::write, &context);
        buffers.add({.address = 0, .size = total, .input = true, .output = false});
        constexpr std::uint32_t status_size = 0x40 + 4 * 0x10 + 0x10 + 0x20 + 0x10 + 0xB0 + 0x10;
        buffers.add({.address = 0x10000, .size = status_size, .input = false, .output = true});
        reply = renderer->request_with_buffers(request(4, 48), 17, buffers);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(get(reply.bytes, 48 + 8) == 0);
        // A successful update clears the renderer event.
        CHECK(!event->is_signaled());
        const auto status = [&](std::size_t offset) {
            std::uint32_t value = 0;
            std::memcpy(&value, context.output.data() + offset, 4);
            return value;
        };
        CHECK(status(0x00) == rev15);
        CHECK(status(0x04) == 0xB0);
        CHECK(status(0x08) == 4 * 0x10);
        CHECK(status(0x0C) == 0x10);
        CHECK(status(0x14) == 0);
        CHECK(status(0x1C) == 0x20);
        CHECK(status(0x20) == 0x10);
        CHECK(status(0x28) == 0x10);
        CHECK(status(0x3C) == status_size);
        // The pool is pinned through the runtime mapper and reported attached.
        CHECK(status(0x40) == 5);
        CHECK(mapper.maps == 1);
        // Frames elapsed: one frame was rendered before the update.
        CHECK(status(status_size - 0x10) == 1);

        // A header that declares the wrong total size is refused with the
        // invalid-update-info result.
        input_word(0x60, 0);
        input_word(0x3C, total + 4);
        buffers = armigo::horizon::GuestBuffers(UpdateContext::read, UpdateContext::write, &context);
        buffers.add({.address = 0, .size = total + 4, .input = true, .output = false});
        buffers.add({.address = 0x10000, .size = status_size, .input = false, .output = true});
        reply = renderer->request_with_buffers(request(10, 48), 17, buffers);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(get(reply.bytes, 48 + 8) == (153u | (41u << 9)));

        // Stopping and closing the renderer releases the pool's pin.
        renderer_request = request(6);
        reply = renderer->request(renderer_request, 17);
        CHECK(reply.disposition == IpcDisposition::handled && !renderer->started());
        renderer.reset();
        CHECK(mapper.unmaps == 1);
    }

    // The audio device service: the manager hands one out for both the plain and
    // the revision-info requests.
    auto device_request = request(4);
    put(device_request, 32, 0x3f564552);  // revision
    put64(device_request, 36, 17);        // applet resource user id
    reply = manager.request(device_request, 17);
    CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1);
    auto device = std::dynamic_pointer_cast<AudioDeviceSession>(reply.transferred[0]);
    CHECK(device != nullptr);
    if (device == nullptr) return 1;
    device_request = request(2);
    put64(device_request, 32, 17);
    reply = manager.request(device_request, 17);
    CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1);
    CHECK(std::dynamic_pointer_cast<AudioDeviceSession>(reply.transferred[0]) != nullptr);

    // Device-set notifications: a copied event that never fires, and a release.
    for (const std::uint32_t acquire : {15u, 17u}) {
        auto notification = request(acquire);
        put64(notification, 32, 0);
        reply = device->request(notification, 17);
        CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1 &&
              reply.transferred_as_copy);
        auto event = std::dynamic_pointer_cast<armigo::horizon::Event>(reply.transferred[0]);
        CHECK(event && !event->is_signaled());
        CHECK(device->request(request(acquire + 1), 17).disposition == IpcDisposition::handled);
    }

    // The device exposes one built-in output, a stereo channel count and a
    // settable volume.
    auto device_command = request(5);
    reply = device->request(device_command, 17);
    CHECK(reply.disposition == IpcDisposition::handled && get(reply.bytes, 32) == 2);
    struct NameBuffer {
        std::array<std::byte, 0x100> bytes{};
        static bool read(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
            auto& self = *static_cast<NameBuffer*>(raw);
            if (address > self.bytes.size() || out.size() > self.bytes.size() - address)
                return false;
            std::memcpy(out.data(), self.bytes.data() + address, out.size());
            return true;
        }
        static bool write(void* raw, std::uint64_t address,
                          std::span<const std::byte> in) noexcept {
            auto& self = *static_cast<NameBuffer*>(raw);
            if (address > self.bytes.size() || in.size() > self.bytes.size() - address)
                return false;
            std::memcpy(self.bytes.data() + address, in.data(), in.size());
            return true;
        }
    } name_buffer;
    armigo::horizon::GuestBuffers device_buffers(NameBuffer::read, NameBuffer::write,
                                                 &name_buffer);
    device_buffers.add({.address = 0, .size = 0x100, .input = false, .output = true});
    device_command = request(0);
    reply = device->request_with_buffers(device_command, 17, device_buffers);
    CHECK(reply.disposition == IpcDisposition::handled && get(reply.bytes, 32) == 1);
    CHECK(std::memcmp(name_buffer.bytes.data(), "AudioBuiltInSpeakerOutput", 25) == 0);
    device_command = request(1);
    const float volume = 0.5f;
    std::memcpy(device_command.data() + 32, &volume, sizeof volume);
    reply = device->request_with_buffers(device_command, 17, device_buffers);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(device->volume() == volume);
    device_command = request(2);
    reply = device->request_with_buffers(device_command, 17, device_buffers);
    CHECK(reply.disposition == IpcDisposition::handled);
    float read_volume = 0.0f;
    std::memcpy(&read_volume, reply.bytes.data() + 32, sizeof read_volume);
    CHECK(read_volume == volume);
    device_command = request(4);
    reply = device->request(device_command, 17);
    CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1);
    CHECK(std::dynamic_pointer_cast<armigo::horizon::Event>(reply.transferred[0]) != nullptr);

    // The manager refuses a different process and an unimplemented command.
    message = request(1);
    CHECK(manager.request(message, 18).disposition == IpcDisposition::unsupported);
    message = request(99);
    reply = manager.request(message, 17);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    CHECK(manager.diagnostic() != nullptr);
    // The protocol locates the CMIF header in the descriptor variants too.
    message = request(1, 32);
    put_parameter(message, 48, 48000, 240, 3, rev15);
    reply = manager.request(message, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK((get(reply.bytes, 48) | (std::uint64_t{get(reply.bytes, 52)} << 32)) == work_buffer_size);

    std::printf("audio renderer failures: %u\n", failures);
    return failures ? 1 : 0;
}
