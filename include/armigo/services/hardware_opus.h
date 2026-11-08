// ============================================================================
// Armigo Services - Hardware Opus Decoder
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The "hwopus" decoder manager and its decoders. Decoding runs on the host
// through libopus; the decoder state lives in host memory, and the guest's
// transfer memory only declares the work buffer size it lent. Work buffer
// sizes are those of the console's decoder (a fixed decoder object size plus
// frame and packet buffers), not of the host library. The Switch 2 console
// serves single-stream decoders. The Switch 1 console (the audio module)
// also serves multi-stream decoders. Its ExEx size queries (commands 8 and 9)
// take the decoder state's size from the audio DSP, which the module does not
// hold; Armigo answers them with the module's own decode object sizes, an
// unverified assumption that affects only how much memory the guest lends.
// The hardware audio codec client session (command 100) and any other request
// are unsupported with an explicit diagnostic.

#pragma once

#include <armigo/horizon/ipc.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

namespace armigo::services::hwopus {

// Results of the hwopus module (111).
constexpr std::uint32_t result(std::uint32_t description) noexcept { return 111u | description << 9; }
constexpr std::uint32_t result_bad_argument = result(2);
constexpr std::uint32_t result_buffer_too_small = result(3);
constexpr std::uint32_t result_internal_error = result(4);
constexpr std::uint32_t result_unimplemented = result(5);
constexpr std::uint32_t result_invalid_state = result(6);
constexpr std::uint32_t result_allocation_failed = result(7);
constexpr std::uint32_t result_input_too_small = result(8);
constexpr std::uint32_t result_invalid_packet = result(17);
constexpr std::uint32_t result_invalid_sample_rate = result(1001);
constexpr std::uint32_t result_invalid_channel_count = result(1002);

// The console whose decoder sizes and commands a manager serves.
enum class Console { switch1, switch2 };

struct DecoderParameters {
    std::uint32_t sample_rate = 0;
    std::uint32_t channel_count = 0;
    bool large_frames = false;
    // A multi-stream decoder: stream counts and the channel-to-stream mapping.
    bool multistream = false;
    std::uint32_t total_streams = 0;
    std::uint32_t stereo_streams = 0;
    std::array<std::uint8_t, 256> mapping{};
};

// The sizing a size query selects: the legacy decode object (commands 0-7)
// or the ExEx decoder (commands 8 and 9).
enum class SizeBackend { legacy, exex };

// The work buffer a decoder with these parameters needs, or a result.
[[nodiscard]] std::uint32_t work_buffer_size(const DecoderParameters& parameters, std::uint32_t& size,
                                             Console console = Console::switch2,
                                             SizeBackend backend = SizeBackend::exex) noexcept;

// Whether this build decodes (libopus linked); without it decoders fail to open.
[[nodiscard]] bool decoding_available() noexcept;

class DecoderSession final : public horizon::ClientSession {
public:
    // Opens libopus for the parameters; result() is nonzero on failure.
    DecoderSession(std::uint64_t process_id, const DecoderParameters& parameters);
    ~DecoderSession() override;
    std::uint32_t result() const noexcept { return result_; }
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    const char* diagnostic() const;

    // Decodes one packet (an 8-byte big-endian size and final-range header,
    // then the packet) into interleaved 16-bit PCM. consumed counts the header.
    struct Decoded {
        std::uint32_t result = 0;
        std::uint32_t consumed = 0;
        std::uint32_t samples = 0;
        std::uint64_t microseconds = 0;
    };
    Decoded decode(std::span<const std::byte> input, std::span<std::byte> output, bool reset);

    // One decoder command whose CMIF header is at `cmif`; the reply's CMIF
    // header goes at `out` (16, or 32 after a domain header). Null with
    // `detail` set when the command is unsupported.
    std::optional<horizon::IpcReply> command(const horizon::IpcMessage&, std::size_t cmif, std::size_t out,
                                             const horizon::GuestBuffers*, const char*& detail);

private:
    horizon::IpcReply method(const horizon::IpcMessage&, const horizon::GuestBuffers*);
    horizon::IpcReply unsupported(const char* detail);
    struct Decoder;
    const std::uint64_t process_id_;
    const DecoderParameters parameters_;
    std::unique_ptr<Decoder> decoder_;
    std::uint32_t result_ = 0;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

// The manager. The SDK converts its session to a domain and clones it before
// opening a decoder; decoders opened through a domain are objects of that
// domain, and their commands arrive with their object id on any session of the
// domain (clones share its objects).
class ManagerSession final : public horizon::ClientSession {
public:
    explicit ManagerSession(std::uint64_t process_id, Console console = Console::switch2);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    const char* diagnostic() const;
    // Live decoder objects of the domain.
    std::size_t domain_objects() const;

private:
    horizon::IpcReply dispatch(const horizon::IpcMessage&, const horizon::GuestBuffers*);
    std::optional<horizon::IpcReply> manager_command(const horizon::IpcMessage&, std::size_t cmif, std::size_t out,
                                                     const horizon::GuestBuffers*);
    std::optional<horizon::IpcReply> open(std::shared_ptr<DecoderSession> decoder, std::size_t out);
    horizon::IpcReply unsupported(const char* detail);
    struct Domain {
        std::mutex mutex;
        bool converted = false;
        std::uint32_t next_object = 2;
        std::map<std::uint32_t, std::shared_ptr<DecoderSession>> objects;
        const char* diagnostic = nullptr;
    };
    ManagerSession(std::uint64_t process_id, Console console, std::shared_ptr<Domain> domain);
    const std::uint64_t process_id_;
    const Console console_;
    const std::shared_ptr<Domain> domain_;
};

} // namespace armigo::services::hwopus
