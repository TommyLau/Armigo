// ============================================================================
// Armigo Tests - Switch 2 Sound Mixer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// audsmx:u streams: creation checks, opening against the ring layout, the
// mixer consuming an engaged stream's frames into the host output in real
// time with its gain, data and state events, and release at shutdown.

#include <armigo/services/sound_mixer.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {
using armigo::horizon::GuestBuffers;
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::horizon::IpcReply;
using namespace armigo::services::audsmx;
using namespace std::chrono_literals;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) { std::memcpy(message.data() + offset, &value, 4); }
void put64(IpcMessage& message, std::size_t offset, std::uint64_t value) { std::memcpy(message.data() + offset, &value, 8); }
std::uint32_t word(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}

// A request whose CMIF header is at 16.
IpcMessage request(std::uint32_t command) {
    IpcMessage message{};
    message.fill(std::byte{0});
    put(message, 0, 4);
    put(message, 4, 0x10);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command);
    return message;
}

// The host output: records what the mixer writes.
class RecordingOutput final : public armigo::audio::PcmOutput {
public:
    bool set_active(bool active) override {
        const std::lock_guard lock(mutex);
        this->active = active;
        return true;
    }
    std::size_t write(const std::int16_t* samples, std::size_t frames, unsigned channels) override {
        const std::lock_guard lock(mutex);
        if (!active || channels != 2) return 0;
        written.insert(written.end(), samples, samples + frames * channels);
        return frames;
    }
    bool set_volume(float) override { return true; }
    armigo::audio::PcmCounters counters() const override { return {}; }
    std::vector<std::int16_t> samples() const {
        const std::lock_guard lock(mutex);
        return written;
    }
    mutable std::mutex mutex;
    bool active = false;
    std::vector<std::int16_t> written;
};

// The title's memory: one ring at 0x10000, a stream-id list at 0x20000.
struct Title {
    std::vector<std::byte> ring = std::vector<std::byte>(0x4000);
    std::vector<std::byte> ids = std::vector<std::byte>(16);
    bool released = false;
    static bool read(void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<Title*>(context);
        if (address < 0x20000 || address + out.size() > 0x20000 + self.ids.size()) return false;
        std::memcpy(out.data(), self.ids.data() + (address - 0x20000), out.size());
        return true;
    }
    static bool write(void*, std::uint64_t, std::span<const std::byte>) noexcept { return false; }
    std::uint32_t slot(std::uint64_t offset) const {
        std::uint32_t value;
        std::memcpy(&value, ring.data() + offset, 4);
        return value;
    }
    void set_slot(std::uint64_t offset, std::uint32_t value) { std::memcpy(ring.data() + offset, &value, 4); }
};

IpcReply engage(UserSession& user, Title& title, std::uint32_t id, bool engaged, std::uint64_t pid) {
    std::memcpy(title.ids.data(), &id, 4);
    GuestBuffers buffers(Title::read, Title::write, &title);
    buffers.add({0x20000, 4, true, true});
    return user.request_with_buffers(request(engaged ? 2 : 3), pid, buffers);
}

template <typename Predicate>
bool eventually(Predicate predicate) {
    for (int i = 0; i < 200 && !predicate(); ++i) std::this_thread::sleep_for(5ms);
    return predicate();
}
} // namespace

int main() {
    constexpr std::uint64_t pid = 0x81;
    Title title;
    auto output = std::make_shared<RecordingOutput>();
    auto pin_holder = std::make_shared<int>(0);
    auto mixer = std::make_shared<Mixer>(
        [&](std::uint64_t address, std::uint64_t size) -> std::optional<StreamMemory> {
            if (address != 0x10000 || size != title.ring.size()) return std::nullopt;
            return StreamMemory{title.ring, pin_holder};
        },
        [output] { return output; });

    SoundMixerSession service(pid, mixer);
    Title other;
    auto mixer2 = std::make_shared<Mixer>(
        [&](std::uint64_t address, std::uint64_t size) -> std::optional<StreamMemory> {
            if (address != 0x10000 || size != other.ring.size()) return std::nullopt;
            return StreamMemory{other.ring, nullptr};
        },
        [] { return armigo::audio::make_muted_pcm_output(); });
    // QueryPointerBufferSize (control type 7, command 3).
    auto query = request(3);
    put(query, 0, 7);
    const auto size = service.request(query, pid);
    CHECK(size.disposition == IpcDisposition::handled && size.size == 36 && word(size.bytes, 16) == 0x4f434653 &&
          word(size.bytes, 32) == 0x8000);
    put(query, 24, 1);
    CHECK(service.request(query, pid).disposition == IpcDisposition::unsupported && service.diagnostic());
    auto opened = service.request(request(0), pid);
    CHECK(opened.disposition == IpcDisposition::handled && opened.transferred_count == 1);
    const auto user = std::dynamic_pointer_cast<UserSession>(opened.transferred[0]);
    CHECK(user != nullptr);
    CHECK(service.request(request(0), pid + 1).disposition == IpcDisposition::unsupported);
    if (!user) return 1;

    const auto create = [&](std::uint32_t device, std::uint32_t flags, std::uint64_t buffer, std::uint64_t size) {
        auto message = request(1);
        put(message, 32, device);
        put(message, 36, flags);
        put64(message, 40, 0x1234);
        put64(message, 48, buffer);
        put64(message, 56, size);
        return user->request(message, pid);
    };
    CHECK(word(create(2, 0x24, 0x10000, 0x4000).bytes, 24) == result_invalid_argument);
    CHECK(word(create(1, 0x20, 0x10000, 0x4000).bytes, 24) == result_not_supported);
    CHECK(word(create(1, 0x24, 0x10800, 0x4000).bytes, 24) == result_invalid_address);
    CHECK(word(create(1, 0x24, 0x30000, 0x4000).bytes, 24) == result_invalid_address);
    auto created = create(1, 0x24, 0x10000, 0x4000);
    CHECK(created.disposition == IpcDisposition::handled && created.transferred_count == 1 &&
          !created.transferred_as_copy && word(created.bytes, 24) == 0 && word(created.bytes, 32) == 1);
    auto stream = std::dynamic_pointer_cast<StreamSession>(created.transferred[0]);
    CHECK(stream != nullptr);
    if (!stream) return 1;

    // OpenStream: 16-bit stereo 48 kHz, 480 frames (481 slots); prefill and
    // flags must be zero, and the format must be one the device takes.
    const auto open = [&](std::uint32_t type, std::uint32_t channels, std::uint32_t frames, std::uint32_t prefill) {
        auto message = request(0);
        put(message, 32, type);
        put(message, 36, channels);
        put(message, 40, 48000);
        put(message, 44, frames);
        put(message, 48, prefill);
        put(message, 52, 0);
        return word(stream->request(message, pid).bytes, 24);
    };
    CHECK(open(2, 6, 480, 0) == result_not_supported);
    CHECK(open(2, 2, 480, 1) == result_not_supported);
    CHECK(open(2, 2, 0x10000, 0) == result_insufficient_buffer);
    CHECK(open(2, 2, 480, 0) == 0);
    CHECK(open(2, 2, 480, 0) == result_operation_failed);
    CHECK(word(stream->request(request(2), pid).bytes, 32) == 1);

    // The title writes 300 frames of 0x4000 and asks for the data event once
    // 400 slots are free.
    for (std::uint32_t frame = 0; frame < 300; ++frame) {
        const std::int16_t sample = 0x4000;
        std::memcpy(title.ring.data() + Stream::payload_offset + frame * 4, &sample, 2);
        std::memcpy(title.ring.data() + Stream::payload_offset + frame * 4 + 2, &sample, 2);
    }
    title.set_slot(Stream::write_offset, 300);
    auto data = request(7);
    put(data, 32, 2);
    put(data, 36, 400);
    const auto data_event = stream->request(data, pid);
    CHECK(data_event.disposition == IpcDisposition::handled && data_event.transferred_as_copy &&
          data_event.transferred_count == 1);
    const auto event = std::dynamic_pointer_cast<armigo::horizon::Event>(data_event.transferred[0]);
    CHECK(event && !event->is_signaled());
    const auto state = stream->request(request(5), pid);
    CHECK(state.transferred_count == 1 && state.transferred_as_copy);

    // Nothing moves until the stream is engaged; then 240 frames every 5 ms.
    std::this_thread::sleep_for(20ms);
    CHECK(title.slot(Stream::read_offset) == 0 && output->samples().empty());
    CHECK(word(engage(*user, title, 7, true, pid).bytes, 24) == result_not_found);
    CHECK(word(engage(*user, title, 1, true, pid).bytes, 24) == 0);
    CHECK(word(stream->request(request(2), pid).bytes, 32) == 2);
    CHECK(eventually([&] { return title.slot(Stream::read_offset) == 300; }));
    const auto played = output->samples();
    CHECK(played.size() == 600 && played.front() == 0x4000 && played.back() == 0x4000);
    CHECK(event && event->is_signaled());

    // Gain scales the samples; a ramp is not supported.
    auto gain = request(4);
    const float half = 0.5f;
    std::memcpy(gain.data() + 32, &half, 4);
    CHECK(word(stream->request(gain, pid).bytes, 24) == 0);
    put64(gain, 40, 1000);
    CHECK(word(stream->request(gain, pid).bytes, 24) == result_not_supported);
    for (std::uint32_t frame = 300; frame < 310; ++frame) {
        const std::int16_t sample = 0x4000;
        std::memcpy(title.ring.data() + Stream::payload_offset + frame * 4, &sample, 2);
        std::memcpy(title.ring.data() + Stream::payload_offset + frame * 4 + 2, &sample, 2);
    }
    title.set_slot(Stream::write_offset, 310);
    CHECK(eventually([&] { return title.slot(Stream::read_offset) == 310; }));
    CHECK(output->samples().size() == 620 && output->samples().back() == 0x2000);

    // Clock, prime and purge are not supported; other commands are not implemented.
    CHECK(word(stream->request(request(9), pid).bytes, 24) == result_not_supported);
    CHECK(word(stream->request(request(11), pid).bytes, 24) == result_not_supported);
    CHECK(stream->request(request(13), pid).disposition == IpcDisposition::unsupported && stream->diagnostic());

    // Disengaged, the ring stays where it is.
    CHECK(word(engage(*user, title, 1, false, pid).bytes, 24) == 0);
    title.set_slot(Stream::write_offset, 320);
    std::this_thread::sleep_for(20ms);
    CHECK(title.slot(Stream::read_offset) == 310);

    // Converted to a domain, the service is object 1; the user session and
    // its streams are new objects, released by a close message.
    {
        SoundMixerSession domain(pid, mixer2);
        auto convert = request(0);
        put(convert, 0, 7);
        const auto converted = domain.request(convert, pid);
        CHECK(converted.disposition == IpcDisposition::handled && word(converted.bytes, 32) == 1);
        const auto object = [](std::uint32_t id, std::uint32_t command) {
            IpcMessage message{};
            message.fill(std::byte{0});
            put(message, 0, 4);
            put(message, 4, 0x14);
            put(message, 16, 1 | 0x20u << 16);
            put(message, 20, id);
            put(message, 32, 0x49434653);
            put(message, 36, 1);
            put(message, 40, command);
            return message;
        };
        const auto user_object = domain.request(object(1, 0), pid);
        CHECK(user_object.disposition == IpcDisposition::handled && user_object.transferred_count == 0 &&
              word(user_object.bytes, 16) == 1 && word(user_object.bytes, 32) == 0x4f434653 &&
              word(user_object.bytes, 40) == 0 && word(user_object.bytes, 48) == 2);
        auto create_in_domain = object(2, 1);
        put(create_in_domain, 48, 1);
        put(create_in_domain, 52, 0x24);
        put64(create_in_domain, 64, 0x10000);
        put64(create_in_domain, 72, 0x4000);
        const auto stream_object = domain.request(create_in_domain, pid);
        CHECK(stream_object.disposition == IpcDisposition::handled && word(stream_object.bytes, 16) == 1 &&
              word(stream_object.bytes, 40) == 0 && word(stream_object.bytes, 48) == 1 &&
              word(stream_object.bytes, 52) == 3);
        CHECK(domain.domain_objects() == 2);
        const auto state_in_domain = domain.request(object(3, 2), pid);
        CHECK(state_in_domain.disposition == IpcDisposition::handled && word(state_in_domain.bytes, 48) == 0);
        IpcMessage close{};
        close.fill(std::byte{0});
        put(close, 0, 4);
        put(close, 4, 0x8);
        put(close, 16, 2);
        put(close, 20, 3);
        CHECK(domain.request(close, pid).disposition == IpcDisposition::handled && domain.domain_objects() == 1);
        CHECK(domain.request(object(3, 2), pid).disposition == IpcDisposition::unsupported);
        CHECK(domain.request(object(1, 5), pid).disposition == IpcDisposition::unsupported && domain.diagnostic());
        // The title releases the service object last.
        put(close, 20, 1);
        CHECK(domain.request(close, pid).disposition == IpcDisposition::handled);
        CHECK(domain.request(close, pid).disposition == IpcDisposition::unsupported);
        CHECK(domain.request(object(1, 0), pid).disposition == IpcDisposition::unsupported);
    }

    // Stopping the mixer releases the stream's view of the title's memory.
    CHECK(pin_holder.use_count() == 2);
    mixer->stop();
    CHECK(pin_holder.use_count() == 1);
    CHECK(word(stream->request(request(2), pid).bytes, 32) == 0);
    // Stops may race (the runtime's stop request and a session that reaches
    // the mixer after it): every one returns, and the worker is joined once.
    for (int round = 0; round != 50; ++round) {
        auto racing = std::make_shared<Mixer>(
            [](std::uint64_t, std::uint64_t) -> std::optional<StreamMemory> { return std::nullopt; },
            [] { return armigo::audio::make_muted_pcm_output(); });
        std::vector<std::thread> stoppers;
        for (int i = 0; i != 8; ++i) stoppers.emplace_back([racing] { racing->stop(); });
        for (auto& stopper : stoppers) stopper.join();
    }
    std::printf("Sound mixer failures: %u\n", failures);
    return failures ? 1 : 0;
}
