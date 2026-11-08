// ============================================================================
// Armigo Services - Switch 2 Sound Mixer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/sound_mixer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

namespace armigo::services::audsmx {
namespace {
using namespace horizon;

constexpr std::uint32_t sfco = 0x4f434653;
// Sample types of a stream: 16-bit integer and 32-bit float PCM.
constexpr std::uint32_t sample_pcm_int16 = 2;
constexpr std::uint32_t sample_pcm_float = 5;
// The only creation flags the title's streams use, and the virtual device.
constexpr std::uint32_t stream_creation_flags = 0x24;
constexpr std::uint32_t virtual_device = 1;
constexpr auto pump_interval = std::chrono::milliseconds(5);

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) { return (value + alignment - 1) & ~(alignment - 1); }

std::uint32_t read(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(message[offset + i])) << (8 * i);
    return value;
}

std::uint64_t read64(const IpcMessage& message, std::size_t offset) {
    return read(message, offset) | std::uint64_t(read(message, offset + 4)) << 32;
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) message[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

constexpr std::uint32_t sfci = 0x49434653;
// The pointer buffer size these sessions answer, the process default the
// practical reference answers (as for hwopus).
constexpr std::uint16_t pointer_buffer_size = 0x8000;

// A reply whose CMIF header is at `out` (16, or 32 after a domain header)
// with `data` bytes after it.
IpcReply reply_with(std::size_t data, std::size_t out = 16) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = out + 16 + data;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, out, sfco);
    return reply;
}

IpcReply result_reply(std::uint32_t code, std::size_t out) {
    auto reply = reply_with(0, out);
    write(reply.bytes, out + 8, code);
    return reply;
}

IpcReply value_reply(std::uint32_t value, std::size_t out) {
    auto reply = reply_with(4, out);
    write(reply.bytes, out + 16, value);
    return reply;
}

// A reply that copies an event's handle to the caller.
IpcReply copy_reply(std::shared_ptr<Event> event, std::size_t out) {
    auto reply = reply_with(0, out);
    reply.transferred[0] = std::move(event);
    reply.transferred_count = 1;
    reply.transferred_as_copy = true;
    return reply;
}

// A CMIF control request (type 5 or 7) and its reply.
bool control(const IpcMessage& message) {
    const auto type = read(message, 0) & 0xffffu;
    return type == 5 || type == 7;
}
IpcReply control_value(std::uint32_t value, unsigned width) {
    auto reply = reply_with(4);
    write(reply.bytes, 32, width == 2 ? value & 0xffffu : value);
    return reply;
}

// The ring's slot words are shared with the title's producer, which reads the
// read slot and writes the write slot with native atomics.
std::uint32_t load(std::span<std::byte> bytes, std::uint64_t offset) {
    return std::atomic_ref(*reinterpret_cast<std::uint32_t*>(bytes.data() + offset)).load(std::memory_order_acquire);
}
void store(std::span<std::byte> bytes, std::uint64_t offset, std::uint32_t value) {
    std::atomic_ref(*reinterpret_cast<std::uint32_t*>(bytes.data() + offset)).store(value, std::memory_order_release);
}
} // namespace

Stream::Stream(std::uint32_t id, StreamMemory memory, std::uint32_t creation_flags,
               std::shared_ptr<audio::PcmOutput> output)
    : id_(id), creation_flags_(creation_flags), output_(std::move(output)), memory_(std::move(memory)) {}

std::uint32_t Stream::validate_region(std::uint64_t buffer, std::uint64_t size, std::uint32_t flags) noexcept {
    if (flags != stream_creation_flags) return result_not_supported;
    if (!buffer || (buffer & 0xfff) || (size & 0xfff) || size < 0x2000 ||
        size > std::numeric_limits<std::uint64_t>::max() - buffer)
        return result_invalid_address;
    return 0;
}

bool Stream::format_supported(std::uint32_t sample_type, std::uint32_t channels, std::uint32_t rate) noexcept {
    return (sample_type == sample_pcm_int16 || sample_type == sample_pcm_float) && channels == 2 && rate == 48000;
}

bool Stream::occupancy(std::uint32_t& read, std::uint32_t& write, std::uint32_t& filled) const {
    if (memory_.bytes.size() < payload_offset) return false;
    read = load(memory_.bytes, read_offset);
    write = load(memory_.bytes, write_offset);
    if (read != consumer_ || read >= slots_ || write >= slots_) return false;
    filled = static_cast<std::uint32_t>((std::uint64_t{write} + slots_ - read) % slots_);
    return true;
}

void Stream::notify(std::uint32_t filled) {
    if (data_mode_ == 2 && slots_ - 1 - filled >= static_cast<std::uint32_t>(threshold_)) data_event_->signal();
}

std::uint32_t Stream::fault(std::uint32_t result) {
    faulted_ = true;
    engaged_ = false;
    ++counters_.faults;
    if (output_) output_->set_active(false);
    state_event_->signal();
    return result;
}

std::uint32_t Stream::open(std::uint32_t sample_type, std::uint32_t channels, std::uint32_t rate,
                           std::uint32_t frames, std::uint32_t prefill, std::uint32_t flags) {
    const std::lock_guard lock(mutex_);
    if (open_) return result_operation_failed;
    if (!format_supported(sample_type, channels, rate) || prefill || flags) return result_not_supported;
    if (!frames || frames >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        return result_invalid_argument;
    if (memory_.bytes.empty()) return result_invalid_address;
    const std::uint64_t slots = std::uint64_t{frames} + 1;
    const std::uint32_t frame_bytes = channels * (sample_type == sample_pcm_int16 ? 2u : 4u);
    const auto minimum = align_up(0x1000 + align_up(0x40 + std::min<std::uint64_t>(16, slots) * 8, 64) +
                                      align_up(slots * frame_bytes, 64), 0x1000);
    if (memory_.bytes.size() < minimum || payload_offset + slots * frame_bytes > memory_.bytes.size())
        return result_insufficient_buffer;
    const auto read = load(memory_.bytes, read_offset), write = load(memory_.bytes, write_offset);
    if (read >= slots || write >= slots) return result_invalid_address;
    // The host output starts accepting frames when the stream is engaged.
    if (!output_) return result_operation_failed;
    slots_ = static_cast<std::uint32_t>(slots);
    float_samples_ = sample_type == sample_pcm_float;
    frame_bytes_ = frame_bytes;
    consumer_ = read;
    faulted_ = false;
    open_ = true;
    engaged_ = false;
    gain_ = 1.0f;
    counters_ = {};
    state_event_->signal();
    notify(static_cast<std::uint32_t>((std::uint64_t{write} + slots - read) % slots));
    return 0;
}

void Stream::close() {
    const std::lock_guard lock(mutex_);
    open_ = false;
    engaged_ = false;
    data_mode_ = 0;
    if (output_) output_->set_active(false);
    state_event_->signal();
}

void Stream::release() {
    const std::lock_guard lock(mutex_);
    open_ = false;
    engaged_ = false;
    data_mode_ = 0;
    if (output_) output_->set_active(false);
    memory_ = {};
}

std::uint32_t Stream::set_engaged(bool engaged) {
    const std::lock_guard lock(mutex_);
    if (!open_ || faulted_ || !output_) return result_operation_failed;
    if (engaged_ == engaged) return 0;
    if (!output_->set_active(engaged)) return engaged ? result_operation_failed : fault(result_operation_failed);
    engaged_ = engaged;
    state_event_->signal();
    return 0;
}

std::uint32_t Stream::set_data_notification(std::uint32_t mode, std::int32_t threshold) {
    const std::lock_guard lock(mutex_);
    if (mode != 0 && mode != 2) return result_not_supported;
    if (!open_ || faulted_ || threshold < 0 || static_cast<std::uint32_t>(threshold) >= slots_)
        return result_invalid_argument;
    std::uint32_t read = 0, write = 0, filled = 0;
    if (!occupancy(read, write, filled)) return fault();
    data_mode_ = mode;
    threshold_ = threshold;
    notify(filled);
    return 0;
}

std::uint32_t Stream::set_gain(float gain, std::int64_t ramp_nanoseconds) {
    if (ramp_nanoseconds) return result_not_supported;
    if (!std::isfinite(gain) || gain < 0.0f || gain > 128.0f) return result_invalid_argument;
    const std::lock_guard lock(mutex_);
    gain_ = gain;
    return 0;
}

float Stream::gain() const {
    const std::lock_guard lock(mutex_);
    return gain_;
}

std::uint32_t Stream::state() const {
    const std::lock_guard lock(mutex_);
    return open_ && !faulted_ ? (engaged_ ? 2 : 1) : 0;
}

StreamCounters Stream::counters() const {
    const std::lock_guard lock(mutex_);
    return counters_;
}

void Stream::pump() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock || !open_ || faulted_) return;
    ++counters_.ticks;
    std::uint32_t read = 0, write = 0, filled = 0;
    if (!occupancy(read, write, filled)) {
        fault();
        return;
    }
    if (!engaged_) {
        notify(filled);
        return;
    }
    const auto taken = std::min(filled, block_frames);
    const auto first = std::min(taken, slots_ - read);
    std::array<std::int16_t, block_frames * 2> samples{};
    for (std::uint32_t frame = 0; frame < taken; ++frame) {
        const auto slot = frame < first ? read + frame : frame - first;
        const auto* source = memory_.bytes.data() + payload_offset + std::uint64_t{slot} * frame_bytes_;
        for (unsigned channel = 0; channel < 2; ++channel) {
            float value;
            if (float_samples_) {
                std::memcpy(&value, source + 4 * channel, sizeof value);
            } else {
                std::int16_t sample;
                std::memcpy(&sample, source + 2 * channel, sizeof sample);
                value = static_cast<float>(sample) / 32768.0f;
            }
            if (!std::isfinite(value)) continue;
            const auto scaled = static_cast<std::int32_t>(std::clamp(value * gain_, -1.0f, 1.0f) * 32768.0f);
            samples[frame * 2 + channel] = static_cast<std::int16_t>(std::clamp(scaled, -32768, 32767));
        }
    }
    // The host output never blocks; frames it cannot take are dropped and
    // counted there and here, and the title's ring still advances in real time.
    if (taken) {
        const auto accepted = output_->write(samples.data(), taken, 2);
        counters_.accepted_frames += accepted;
        counters_.dropped_frames += taken - accepted;
    }
    const auto next = (read + taken) % slots_;
    if (load(memory_.bytes, read_offset) != read) {
        fault();
        return;
    }
    store(memory_.bytes, read_offset, next);
    consumer_ = next;
    if (!occupancy(read, write, filled)) {
        fault();
        return;
    }
    notify(filled);
}

Mixer::Mixer(MemoryPinner pinner, OutputFactory output_factory)
    : pinner_(std::move(pinner)), output_factory_(std::move(output_factory)),
      worker_([this](std::stop_token stop) { run(stop); }) {}

Mixer::~Mixer() { stop(); }

void Mixer::stop() noexcept {
    {
        // Stops may race (the runtime's stop request and a session that
        // reaches the mixer after it): one joins the worker, the others wait
        // for that join instead of joining it again.
        const std::lock_guard stop_lock(stop_mutex_);
        worker_.request_stop();
        if (worker_.joinable()) worker_.join();
    }
    std::map<std::uint32_t, std::shared_ptr<Stream>> streams;
    {
        const std::lock_guard lock(mutex_);
        stopped_ = true;
        streams.swap(streams_);
    }
    for (auto& [id, stream] : streams) stream->release();
}

void Mixer::run(std::stop_token stop) {
    // An absolute schedule keeps 240 frames every 5 ms however late a tick runs.
    auto next = std::chrono::steady_clock::now() + pump_interval;
    std::mutex sleep_mutex;
    std::condition_variable_any sleep;
    while (!stop.stop_requested()) {
        {
            std::unique_lock lock(sleep_mutex);
            sleep.wait_until(lock, stop, next, [] { return false; });
        }
        if (stop.stop_requested()) break;
        const auto now = std::chrono::steady_clock::now();
        const auto ticks = std::max<std::int64_t>(1, (now - next) / pump_interval + 1);
        next += pump_interval * ticks;
        std::vector<std::shared_ptr<Stream>> streams;
        {
            const std::lock_guard lock(mutex_);
            for (auto& [id, stream] : streams_) streams.push_back(stream);
        }
        for (auto& stream : streams) stream->pump();
    }
}

std::uint32_t Mixer::create(std::uint64_t buffer, std::uint64_t size, std::uint32_t flags, std::uint32_t& id,
                            std::shared_ptr<Stream>& stream) {
    if (const auto invalid = Stream::validate_region(buffer, size, flags)) return invalid;
    auto memory = pinner_ ? pinner_(buffer, size) : std::nullopt;
    if (!memory || memory->bytes.size() != size) return result_invalid_address;
    auto output = output_factory_ ? output_factory_() : nullptr;
    if (!output) output = audio::make_muted_pcm_output();
    const std::lock_guard lock(mutex_);
    if (stopped_) return result_operation_failed;
    id = next_id_++;
    stream = std::make_shared<Stream>(id, std::move(*memory), flags, std::move(output));
    streams_.emplace(id, stream);
    return 0;
}

void Mixer::remove(std::uint32_t id) {
    std::shared_ptr<Stream> stream;
    {
        const std::lock_guard lock(mutex_);
        const auto found = streams_.find(id);
        if (found == streams_.end()) return;
        stream = std::move(found->second);
        streams_.erase(found);
    }
    stream->release();
}

std::uint32_t Mixer::set_engaged(std::uint32_t id, bool engaged) {
    std::shared_ptr<Stream> stream;
    {
        const std::lock_guard lock(mutex_);
        const auto found = streams_.find(id);
        if (found == streams_.end()) return result_not_found;
        stream = found->second;
    }
    return stream->set_engaged(engaged);
}

IpcReply Interface::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

const char* Interface::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

IpcReply Interface::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Sound mixer session belongs to another process");
    return serve(message, nullptr);
}

IpcReply Interface::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                         const GuestBuffers& buffers) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Sound mixer session belongs to another process");
    return serve(message, &buffers);
}

// A session of its own: an interface it opens is moved to the caller.
IpcReply Interface::serve(const IpcMessage& message, const GuestBuffers* buffers) {
    if (control(message)) {
        if (read(message, 16) == sfci && read(message, 24) == 3) return control_value(pointer_buffer_size, 2);
        return unsupported("Sound mixer control request is unsupported");
    }
    const auto cmif = locate_cmif(message);
    if (!cmif || read(message, *cmif) != sfci) return unsupported("Sound mixer request shape is unsupported");
    std::shared_ptr<Interface> opened;
    const char* detail = nullptr;
    auto reply = command(message, *cmif, 16, buffers, opened, detail);
    if (!reply) return unsupported(detail);
    if (opened) {
        reply->transferred[0] = std::move(opened);
        reply->transferred_count = 1;
        reply->transferred_as_copy = false;
    }
    return std::move(*reply);
}

StreamSession::StreamSession(std::uint64_t process_id, std::shared_ptr<Mixer> mixer, std::shared_ptr<Stream> stream,
                             std::uint32_t id)
    : Interface(process_id), mixer_(std::move(mixer)), stream_(std::move(stream)), id_(id) {}

StreamSession::~StreamSession() { mixer_->remove(id_); }

std::optional<IpcReply> StreamSession::command(const IpcMessage& message, std::size_t cmif, std::size_t out,
                                               const GuestBuffers*, std::shared_ptr<Interface>&,
                                               const char*& detail) {
    switch (read(message, cmif + 8)) {
    case 0:  // OpenStream(sample type, channels, rate, frames, prefill, flags)
        return result_reply(stream_->open(read(message, cmif + 16), read(message, cmif + 20),
                                          read(message, cmif + 24), read(message, cmif + 28),
                                          read(message, cmif + 32), read(message, cmif + 36)), out);
    case 1:  // CloseStream
        stream_->close();
        return result_reply(0, out);
    case 2:  // GetStreamState
        return value_reply(stream_->state(), out);
    case 3: {  // GetStreamGain
        const auto gain = stream_->gain();
        std::uint32_t bits;
        std::memcpy(&bits, &gain, sizeof bits);
        return value_reply(bits, out);
    }
    case 4: {  // SetStreamGain(gain, ramp in ns)
        float gain;
        const auto bits = read(message, cmif + 16);
        std::memcpy(&gain, &bits, sizeof gain);
        const auto ramp = static_cast<std::int64_t>(read64(message, cmif + 24));
        return result_reply(stream_->set_gain(gain, ramp), out);
    }
    case 5:  // AcquireStreamStateChangeEvent
        return copy_reply(stream_->state_event(), out);
    case 6:   // ReleaseStreamStateChangeEvent
    case 12:  // RenameStream
        return result_reply(0, out);
    case 7: {  // AcquireStreamDataEvent(mode, threshold)
        const auto result = stream_->set_data_notification(read(message, cmif + 16),
                                                           static_cast<std::int32_t>(read(message, cmif + 20)));
        if (result) return result_reply(result, out);
        return copy_reply(stream_->data_event(), out);
    }
    case 8:  // ReleaseStreamDataEvent
        return result_reply(stream_->set_data_notification(0, 0), out);
    case 9:   // ClockStream
    case 10:  // PrimeStream
    case 11:  // PurgeStream
        return result_reply(result_not_supported, out);
    default:
        detail = "Sound mixer stream command is not implemented";
        return std::nullopt;
    }
}

UserSession::UserSession(std::uint64_t process_id, std::shared_ptr<Mixer> mixer)
    : Interface(process_id), mixer_(std::move(mixer)) {}

std::optional<IpcReply> UserSession::command(const IpcMessage& message, std::size_t cmif, std::size_t out,
                                             const GuestBuffers* buffers, std::shared_ptr<Interface>& opened,
                                             const char*& detail) {
    const auto command = read(message, cmif + 8);
    switch (command) {
    case 1: {  // CreateStream(device, flags, aruid, buffer, size; name) -> id, stream
        const auto device = read(message, cmif + 16), flags = read(message, cmif + 20);
        const auto buffer = read64(message, cmif + 32), size = read64(message, cmif + 40);
        if (device != virtual_device) return result_reply(result_invalid_argument, out);
        std::uint32_t id = 0;
        std::shared_ptr<Stream> stream;
        if (const auto result = mixer_->create(buffer, size, flags, id, stream)) return result_reply(result, out);
        opened = std::make_shared<StreamSession>(process_id_, mixer_, std::move(stream), id);
        return value_reply(id, out);
    }
    case 2:    // EngageStreams(ids)
    case 3: {  // DisengageStreams(ids)
        if (!buffers || buffers->empty()) {
            detail = "Sound mixer stream list needs its buffer";
            return std::nullopt;
        }
        const auto& list = (*buffers)[0];
        if (list.size % 4) return result_reply(result_invalid_argument, out);
        std::vector<std::byte> ids(list.size);
        if (!buffers->read(0, ids)) {
            detail = "Sound mixer stream list is not readable";
            return std::nullopt;
        }
        std::uint32_t result = 0;
        for (std::size_t offset = 0; offset < ids.size() && !result; offset += 4) {
            std::uint32_t id;
            std::memcpy(&id, ids.data() + offset, sizeof id);
            result = mixer_->set_engaged(id, command == 2);
        }
        return result_reply(result, out);
    }
    default:
        detail = "Sound mixer session command is not implemented";
        return std::nullopt;
    }
}

SoundMixerSession::SoundMixerSession(std::uint64_t process_id, std::shared_ptr<Mixer> mixer)
    : SoundMixerSession(process_id, std::move(mixer), std::make_shared<Domain>()) {}

SoundMixerSession::SoundMixerSession(std::uint64_t process_id, std::shared_ptr<Mixer> mixer,
                                     std::shared_ptr<Domain> domain)
    : Interface(process_id), mixer_(std::move(mixer)), domain_(std::move(domain)) {}

std::size_t SoundMixerSession::domain_objects() const {
    const std::lock_guard lock(domain_->mutex);
    return domain_->objects.size();
}

std::optional<IpcReply> SoundMixerSession::command(const IpcMessage& message, std::size_t cmif, std::size_t out,
                                                   const GuestBuffers*, std::shared_ptr<Interface>& opened,
                                                   const char*& detail) {
    if (read(message, cmif + 8) != 0) {
        detail = "Sound mixer service command is not implemented";
        return std::nullopt;
    }
    // OpenUserSession takes the title's process handle; this session already
    // belongs to that process.
    opened = std::make_shared<UserSession>(process_id_, mixer_);
    return result_reply(0, out);
}

IpcReply SoundMixerSession::serve(const IpcMessage& message, const GuestBuffers* buffers) {
    const std::lock_guard lock(domain_->mutex);
    if (control(message)) {
        // QueryPointerBufferSize; ConvertCurrentObjectToDomain, after which
        // the service is object 1; CloneCurrentObject, another session of
        // the same domain.
        if (read(message, 16) != sfci) return unsupported("Sound mixer control request is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_value(pointer_buffer_size, 2);
        if (command == 0 && !domain_->converted) {
            domain_->converted = true;
            return control_value(1, 4);
        }
        if (command == 2 && domain_->converted) {
            auto reply = control_value(0, 4);
            reply.transferred[0].reset(new SoundMixerSession(process_id_, mixer_, domain_));
            reply.transferred_count = 1;
            return reply;
        }
        return unsupported("Sound mixer control request is unsupported");
    }
    if (!domain_->converted) return Interface::serve(message, buffers);
    // A domain message: {type, input objects, data size} then the object id.
    // Type 2 without data releases the object (the title releases the
    // service, object 1, last); type 1 carries a CMIF request.
    const auto header = read(message, 16);
    if ((header & 0xff) == 2 && (header >> 16) == 0) {
        const auto id = read(message, 20);
        if (id == 1 && !domain_->root_released) domain_->root_released = true;
        else if (!domain_->objects.erase(id)) return unsupported("Sound mixer domain object is not open");
        return reply_with(0);
    }
    std::optional<std::size_t> cmif;
    for (const std::size_t at : {32u, 48u, 64u})
        if (read(message, at) == sfci) {
            cmif = at;
            break;
        }
    if (!cmif || (read(message, *cmif - 16) & 0xff) != 1)
        return unsupported("Sound mixer domain request shape is unsupported");
    const auto id = read(message, *cmif - 12);
    Interface* target = this;
    if (id == 1 && domain_->root_released) return unsupported("Sound mixer domain object is not open");
    if (id != 1) {
        const auto found = domain_->objects.find(id);
        if (found == domain_->objects.end()) return unsupported("Sound mixer domain object is not open");
        target = found->second.get();
    }
    std::shared_ptr<Interface> opened;
    const char* detail = nullptr;
    auto reply = target->command(message, *cmif, 32, buffers, opened, detail);
    if (!reply) return unsupported(detail);
    if (opened) {
        // The opened interface becomes an object of the domain: the reply's
        // domain header counts one output object, whose id follows the
        // command's output data.
        const auto object = domain_->next_object++;
        domain_->objects.emplace(object, std::move(opened));
        write(reply->bytes, 16, 1);
        const auto at = reply->size;
        reply->size += 4;
        write(reply->bytes, 4, static_cast<std::uint32_t>((reply->size - 8) / 4));
        write(reply->bytes, at, object);
    }
    return std::move(*reply);
}

} // namespace armigo::services::audsmx
