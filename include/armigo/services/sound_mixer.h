// ============================================================================
// Armigo Services - Switch 2 Sound Mixer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The Switch 2 "audsmx:u" sound mixer, following the practical reference. A
// title opens a user session (0), creates output streams on the virtual
// output device (1) and engages or disengages them (2, 3). A stream plays a
// ring the title writes in its own memory: the consumer's read slot at +0x44,
// the producer's write slot at +0x1004 and the frames from +0x1040, one slot
// more than the frames the stream was opened with. The mixer takes up to 240
// frames (5 ms at 48 kHz) from each engaged stream every 5 ms, writes them to
// the host output as PCM16 with the stream's gain, advances the read slot and
// signals the stream's data event once the free space reaches its threshold.
// Streams play stereo 48 kHz 16-bit or float PCM. Clocking, priming and
// purging a stream and any other request are unsupported, the first three
// with the audio module's not-supported result as in the reference. The
// sessions answer QueryPointerBufferSize with no pointer buffer.

#pragma once

#include <armigo/audio/pcm_output.h>
#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>

namespace armigo::services::audsmx {

// Results of the audio module (153).
constexpr std::uint32_t result(std::uint32_t description) noexcept { return 153u | description << 9; }
constexpr std::uint32_t result_not_found = result(1);
constexpr std::uint32_t result_operation_failed = result(2);
constexpr std::uint32_t result_insufficient_buffer = result(4);
constexpr std::uint32_t result_invalid_address = result(42);
constexpr std::uint32_t result_not_supported = result(513);
constexpr std::uint32_t result_invalid_argument = result(900);
constexpr std::uint32_t result_invalid_handle = result(1536);

// A stream's ring: a live view of the title's buffer, kept mapped until the
// view is released.
struct StreamMemory {
    std::span<std::byte> bytes;
    std::shared_ptr<void> keep;
};
// Maps a title's buffer for a stream, or nothing when the range is not the
// title's readable and writable memory.
using MemoryPinner = std::function<std::optional<StreamMemory>(std::uint64_t address, std::uint64_t size)>;
using OutputFactory = std::function<std::shared_ptr<audio::PcmOutput>()>;

struct StreamCounters {
    std::uint64_t ticks = 0;
    std::uint64_t accepted_frames = 0;
    std::uint64_t dropped_frames = 0;
    std::uint64_t faults = 0;
};

class Stream {
public:
    static constexpr std::uint64_t read_offset = 0x44;
    static constexpr std::uint64_t write_offset = 0x1004;
    static constexpr std::uint64_t payload_offset = 0x1040;
    static constexpr std::uint32_t block_frames = 240;

    Stream(std::uint32_t id, StreamMemory memory, std::uint32_t creation_flags,
           std::shared_ptr<audio::PcmOutput> output);
    // Creation flags, alignment and size of a stream buffer.
    static std::uint32_t validate_region(std::uint64_t buffer, std::uint64_t size, std::uint32_t flags) noexcept;
    static bool format_supported(std::uint32_t sample_type, std::uint32_t channels, std::uint32_t rate) noexcept;

    std::uint32_t open(std::uint32_t sample_type, std::uint32_t channels, std::uint32_t rate, std::uint32_t frames,
                       std::uint32_t prefill, std::uint32_t flags);
    void close();
    std::uint32_t set_engaged(bool engaged);
    std::uint32_t set_data_notification(std::uint32_t mode, std::int32_t threshold);
    std::uint32_t set_gain(float gain, std::int64_t ramp_nanoseconds);
    float gain() const;
    // 0 closed or faulted, 1 open, 2 engaged.
    std::uint32_t state() const;
    // Moves one block to the host output; the mixer calls it every 5 ms.
    void pump();
    // Drops the view of the title's buffer; the stream stays closed.
    void release();
    const std::shared_ptr<horizon::Event>& state_event() const noexcept { return state_event_; }
    const std::shared_ptr<horizon::Event>& data_event() const noexcept { return data_event_; }
    StreamCounters counters() const;

private:
    bool occupancy(std::uint32_t& read, std::uint32_t& write, std::uint32_t& filled) const;
    void notify(std::uint32_t filled);
    std::uint32_t fault(std::uint32_t result = result_invalid_address);

    const std::uint32_t id_;
    const std::uint32_t creation_flags_;
    const std::shared_ptr<audio::PcmOutput> output_;
    const std::shared_ptr<horizon::Event> state_event_ = std::make_shared<horizon::Event>(false);
    const std::shared_ptr<horizon::Event> data_event_ = std::make_shared<horizon::Event>(false);
    mutable std::mutex mutex_;
    StreamMemory memory_;
    bool float_samples_ = false;
    std::uint32_t frame_bytes_ = 0;
    std::uint32_t slots_ = 0;
    std::uint32_t consumer_ = 0;
    std::uint32_t data_mode_ = 0;
    std::int32_t threshold_ = 0;
    float gain_ = 1.0f;
    bool open_ = false;
    bool engaged_ = false;
    bool faulted_ = false;
    StreamCounters counters_;
};

// The process's streams and the 5 ms worker that pumps them.
class Mixer {
public:
    Mixer(MemoryPinner pinner, OutputFactory output_factory);
    ~Mixer();
    // Stops the worker and closes every stream, releasing their buffers; for
    // process shutdown, before the title's memory goes away.
    void stop() noexcept;
    // A new stream on the title's buffer and its id, or a result.
    std::uint32_t create(std::uint64_t buffer, std::uint64_t size, std::uint32_t flags, std::uint32_t& id,
                         std::shared_ptr<Stream>& stream);
    void remove(std::uint32_t id);
    std::uint32_t set_engaged(std::uint32_t id, bool engaged);

private:
    void run(std::stop_token stop);
    const MemoryPinner pinner_;
    const OutputFactory output_factory_;
    std::mutex mutex_;
    std::map<std::uint32_t, std::shared_ptr<Stream>> streams_;
    std::uint32_t next_id_ = 1;
    bool stopped_ = false;
    // Serializes stops around the worker's join.
    std::mutex stop_mutex_;
    std::jthread worker_;
};

// A sound mixer interface: served as a session of its own, or as an object of
// the domain the service session was converted to.
class Interface : public horizon::ClientSession {
public:
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;
    // One command whose CMIF header is at `cmif`; the reply's CMIF header is
    // at `out` (16, or 32 after a domain header). A command that opens an
    // interface sets `opened`. Null with `detail` set when unsupported.
    virtual std::optional<horizon::IpcReply> command(const horizon::IpcMessage&, std::size_t cmif, std::size_t out,
                                                     const horizon::GuestBuffers*,
                                                     std::shared_ptr<Interface>& opened, const char*& detail) = 0;

protected:
    explicit Interface(std::uint64_t process_id) : process_id_(process_id) {}
    virtual horizon::IpcReply serve(const horizon::IpcMessage&, const horizon::GuestBuffers*);
    horizon::IpcReply unsupported(const char* detail);
    const std::uint64_t process_id_;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

class StreamSession final : public Interface {
public:
    StreamSession(std::uint64_t process_id, std::shared_ptr<Mixer> mixer, std::shared_ptr<Stream> stream,
                  std::uint32_t id);
    ~StreamSession() override;
    std::optional<horizon::IpcReply> command(const horizon::IpcMessage&, std::size_t cmif, std::size_t out,
                                             const horizon::GuestBuffers*, std::shared_ptr<Interface>& opened,
                                             const char*& detail) override;

private:
    const std::shared_ptr<Mixer> mixer_;
    const std::shared_ptr<Stream> stream_;
    const std::uint32_t id_;
};

class UserSession final : public Interface {
public:
    UserSession(std::uint64_t process_id, std::shared_ptr<Mixer> mixer);
    std::optional<horizon::IpcReply> command(const horizon::IpcMessage&, std::size_t cmif, std::size_t out,
                                             const horizon::GuestBuffers*, std::shared_ptr<Interface>& opened,
                                             const char*& detail) override;

private:
    const std::shared_ptr<Mixer> mixer_;
};

// The service session. The SDK converts it to a domain (it becomes object 1)
// and reaches the user session and its streams as objects of that domain.
class SoundMixerSession final : public Interface {
public:
    SoundMixerSession(std::uint64_t process_id, std::shared_ptr<Mixer> mixer);
    std::optional<horizon::IpcReply> command(const horizon::IpcMessage&, std::size_t cmif, std::size_t out,
                                             const horizon::GuestBuffers*, std::shared_ptr<Interface>& opened,
                                             const char*& detail) override;
    // Open objects of the domain besides the service itself.
    std::size_t domain_objects() const;

protected:
    horizon::IpcReply serve(const horizon::IpcMessage&, const horizon::GuestBuffers*) override;

private:
    struct Domain {
        std::mutex mutex;
        bool converted = false;
        bool root_released = false;
        std::uint32_t next_object = 2;
        std::map<std::uint32_t, std::shared_ptr<Interface>> objects;
    };
    SoundMixerSession(std::uint64_t process_id, std::shared_ptr<Mixer> mixer, std::shared_ptr<Domain> domain);
    const std::shared_ptr<Mixer> mixer_;
    const std::shared_ptr<Domain> domain_;
};

} // namespace armigo::services::audsmx
