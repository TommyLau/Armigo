// ============================================================================
// Armigo Services - Hardware Opus Decoder
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/hardware_opus.h>

#include <chrono>
#include <cstring>
#include <vector>

#ifdef ARMIGO_HAS_OPUS
#include <opus/opus.h>
#include <opus/opus_multistream.h>
#endif

namespace armigo::services::hwopus {
namespace {
using namespace horizon;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;

// A single-stream decode object: its header, then the libopus decoder state
// of the console's build, which games size their audio memory around.
constexpr std::uint32_t decode_object_header_size = 24;
constexpr std::uint32_t mono_decoder_size = 18468;
constexpr std::uint32_t stereo_decoder_size = 27236;
// Each decoder stages one packet of up to 0x600 bytes.
constexpr std::uint32_t packet_buffer_size = 0x600;
constexpr std::uint32_t frame_samples = 1920;
constexpr std::uint32_t large_frame_samples = 5760;
// The 23.0.1 audio modules' legacy decode objects (size commands 1, 3, 5
// and 7). The single-stream object has the same size table on both
// platforms (one and two channels), followed by its packet staging: 0x600
// bytes on Switch 1, eight fewer on Switch 2 (S1 0x95310/0x93e40, S2
// 0x612640/0x19f790). The multi-stream object holds a fixed part plus one
// decoder state per mono or stereo stream and stages bytes per stream (S1
// 0x95340/0x94380, S2 0x19f8d0).
constexpr std::array<std::uint32_t, 2> legacy_decoder_size{44408, 53128};
struct MultistreamLayout {
    std::uint32_t base, mono_stream, stereo_stream, stream_staging;
};
constexpr MultistreamLayout switch1_multistream{47392, 17764, 26484, 1500};
constexpr MultistreamLayout switch2_multistream{47384, 17756, 26476, 1508};
constexpr std::uint32_t switch2_legacy_packet_size = packet_buffer_size - 8;
constexpr std::uint32_t max_streams = 255;
// Multi-stream parameters arrive in a pointer buffer: rate, channels, total
// and stereo stream counts, then the 256-byte mapping (at 0x10), or after the
// large-frame flag at 0x10 and padding (at 0x18) in the extended form.
constexpr std::size_t multistream_parameters_size = 0x110;
constexpr std::size_t multistream_parameters_ex_size = 0x118;
// The packet header is a big-endian size and final range.
constexpr std::size_t packet_header_size = 8;
// The SDK's generated client asks for the pointer buffer before its first
// command; the practical reference answers its process default.
constexpr std::uint16_t pointer_buffer_size = 0x8000;

std::uint32_t read(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::uint32_t(std::to_integer<unsigned char>(message[offset + i])) << (8 * i);
    return value;
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value, unsigned width = 4) {
    for (unsigned i = 0; i < width; ++i) message[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

std::uint32_t big_endian(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value = value << 8 | std::to_integer<std::uint32_t>(bytes[offset + i]);
    return value;
}

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

IpcReply result_reply(std::uint32_t code, std::size_t out = 16) {
    auto reply = reply_with(0, out);
    write(reply.bytes, out + 8, code);
    return reply;
}

// A CMIF control request with `value` (`width` bytes) as its output.
IpcReply control_value(std::uint32_t value, unsigned width) {
    auto reply = reply_with(4);
    write(reply.bytes, 32, value, width);
    return reply;
}

bool control(const IpcMessage& message) {
    const auto type = read(message, 0) & 0xffffu;
    return type == 5 || type == 7;
}

// The CMIF header of a request: after the HIPC header and descriptors, and
// after a domain header when `domain` is set.
std::optional<std::size_t> request_cmif(const IpcMessage& message, bool domain) {
    if (!domain) {
        const auto located = locate_cmif(message);
        if (!located || read(message, *located) != sfci) return std::nullopt;
        return located;
    }
    for (const std::size_t at : {32u, 48u, 64u})
        if (read(message, at) == sfci) return at;
    return std::nullopt;
}

DecoderParameters parameters_from(const IpcMessage& message, std::size_t offset, bool extended) {
    DecoderParameters parameters;
    parameters.sample_rate = read(message, offset);
    parameters.channel_count = read(message, offset + 4);
    parameters.large_frames = extended && (read(message, offset + 8) & 0xffu) != 0;
    return parameters;
}

std::uint32_t align_up(std::uint32_t value, std::uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Bytes of one frame of interleaved samples at the decoder's rate, as the
// console's decoder sizes its output staging.
std::uint32_t frame_bytes(const DecoderParameters& parameters) {
    const auto samples = parameters.large_frames ? large_frame_samples : frame_samples;
    return samples * parameters.channel_count / (48'000 / parameters.sample_rate);
}

// Channel count, then rate, then (multi-stream) stream counts, in the order
// the console checks them; an invalid stream count reports the rate result.
std::uint32_t validate(const DecoderParameters& parameters) {
    if (parameters.multistream) {
        if (parameters.channel_count < 1 || parameters.channel_count > max_streams)
            return result_invalid_channel_count;
    } else if (parameters.channel_count != 1 && parameters.channel_count != 2) {
        return result_invalid_channel_count;
    }
    switch (parameters.sample_rate) {
    case 8'000: case 12'000: case 16'000: case 24'000: case 48'000: break;
    default: return result_invalid_sample_rate;
    }
    if (parameters.multistream &&
        (parameters.total_streams < 1 || parameters.total_streams > max_streams ||
         parameters.stereo_streams > parameters.total_streams ||
         parameters.total_streams + parameters.stereo_streams > parameters.channel_count))
        return result_invalid_sample_rate;
    return 0;
}

std::optional<DecoderParameters> multistream_parameters(const GuestBuffers* buffers, bool extended) {
    const std::size_t size = extended ? multistream_parameters_ex_size : multistream_parameters_size;
    if (!buffers || buffers->size() < 1 || (*buffers)[0].size < size) return std::nullopt;
    std::array<std::byte, multistream_parameters_ex_size> bytes{};
    if (!buffers->read(0, std::span(bytes).first(size))) return std::nullopt;
    auto word = [&](std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, 4);
        return value;
    };
    DecoderParameters parameters;
    parameters.multistream = true;
    parameters.sample_rate = word(0);
    parameters.channel_count = word(4);
    parameters.total_streams = word(8);
    parameters.stereo_streams = word(12);
    parameters.large_frames = extended && std::to_integer<unsigned>(bytes[0x10]) != 0;
    std::memcpy(parameters.mapping.data(), bytes.data() + (extended ? 0x18 : 0x10), parameters.mapping.size());
    return parameters;
}

#ifdef ARMIGO_HAS_OPUS
std::uint32_t result_from_opus(int error) {
    switch (error) {
    case OPUS_OK: return 0;
    case OPUS_BAD_ARG: return result_bad_argument;
    case OPUS_BUFFER_TOO_SMALL: return result_buffer_too_small;
    case OPUS_INTERNAL_ERROR: return result_internal_error;
    case OPUS_INVALID_PACKET: return result_invalid_packet;
    case OPUS_UNIMPLEMENTED: return result_unimplemented;
    case OPUS_INVALID_STATE: return result_invalid_state;
    case OPUS_ALLOC_FAIL: return result_allocation_failed;
    default: return result_internal_error;
    }
}
#endif
} // namespace

std::uint32_t work_buffer_size(const DecoderParameters& parameters, std::uint32_t& size, Console console,
                               SizeBackend backend) noexcept {
    const bool switch2 = console == Console::switch2;
    // The Switch 2 ExEx multi-stream backend (0x1a0560 -> 0x1a2db0) is not recovered.
    if (parameters.multistream && switch2 && backend == SizeBackend::exex) return result_unimplemented;
    if (const auto failure = validate(parameters)) return failure;
    if (parameters.multistream) {
        const auto& layout = switch2 && backend == SizeBackend::legacy ? switch2_multistream : switch1_multistream;
        size = layout.base +
               (parameters.total_streams - parameters.stereo_streams) * layout.mono_stream +
               parameters.stereo_streams * layout.stereo_stream +
               align_up(layout.stream_staging * parameters.total_streams, 64) +
               align_up(frame_bytes(parameters), 64);
        return 0;
    }
    if (backend == SizeBackend::legacy || !switch2) {
        // Switch 1 ExEx (0x94010 -> 0x96a20) reads a runtime-populated base
        // table that is not recovered; it keeps the legacy object here.
        size = legacy_decoder_size[parameters.channel_count - 1] + align_up(frame_bytes(parameters), 64) +
               (switch2 ? switch2_legacy_packet_size : packet_buffer_size);
        return 0;
    }
    size = decode_object_header_size + (parameters.channel_count == 1 ? mono_decoder_size : stereo_decoder_size) +
           align_up(frame_bytes(parameters), 64) + packet_buffer_size;
    return 0;
}

bool decoding_available() noexcept {
#ifdef ARMIGO_HAS_OPUS
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------- decoder

struct DecoderSession::Decoder {
#ifdef ARMIGO_HAS_OPUS
    OpusDecoder* opus = nullptr;
    OpusMSDecoder* multistream = nullptr;
    ~Decoder() {
        if (opus) opus_decoder_destroy(opus);
        if (multistream) opus_multistream_decoder_destroy(multistream);
    }
#endif
};

DecoderSession::DecoderSession(std::uint64_t process_id, const DecoderParameters& parameters)
    : process_id_(process_id), parameters_(parameters) {
    if ((result_ = validate(parameters))) return;
#ifdef ARMIGO_HAS_OPUS
    decoder_ = std::make_unique<Decoder>();
    int error = OPUS_OK;
    if (parameters.multistream) {
        decoder_->multistream = opus_multistream_decoder_create(
            static_cast<opus_int32>(parameters.sample_rate), static_cast<int>(parameters.channel_count),
            static_cast<int>(parameters.total_streams), static_cast<int>(parameters.stereo_streams),
            parameters.mapping.data(), &error);
        result_ = decoder_->multistream ? result_from_opus(error)
                                        : error != OPUS_OK ? result_from_opus(error) : result_allocation_failed;
        return;
    }
    decoder_->opus = opus_decoder_create(static_cast<opus_int32>(parameters.sample_rate),
                                         static_cast<int>(parameters.channel_count), &error);
    result_ = decoder_->opus ? result_from_opus(error) : result_allocation_failed;
#else
    result_ = result_unimplemented;
#endif
}

DecoderSession::~DecoderSession() = default;

IpcReply DecoderSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

const char* DecoderSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

DecoderSession::Decoded DecoderSession::decode(std::span<const std::byte> input, std::span<std::byte> output,
                                               bool reset) {
    Decoded decoded;
    if (input.size() <= packet_header_size) {
        decoded.result = result_input_too_small;
        return decoded;
    }
    const auto size = big_endian(input, 0);
    // A multi-stream decoder (Switch 1 only) stages 1500 bytes per stream instead of one packet buffer.
    const auto staging = parameters_.multistream ? switch1_multistream.stream_staging * parameters_.total_streams
                                                 : packet_buffer_size;
    if (size > staging || size > input.size() - packet_header_size) {
        decoded.result = result_buffer_too_small;
        return decoded;
    }
#ifdef ARMIGO_HAS_OPUS
    if (!decoder_ || (!decoder_->opus && !decoder_->multistream)) {
        decoded.result = result_invalid_state;
        return decoded;
    }
    const auto begin = std::chrono::steady_clock::now();
    if (reset) {
        const auto error = decoder_->multistream ? opus_multistream_decoder_ctl(decoder_->multistream, OPUS_RESET_STATE)
                                                 : opus_decoder_ctl(decoder_->opus, OPUS_RESET_STATE);
        if (error != OPUS_OK) {
            decoded.result = result_from_opus(error);
            return decoded;
        }
    }
    // Decode into a buffer of whole frames the guest's output can hold; a
    // packet longer than that is too large for it.
    const auto channels = parameters_.channel_count;
    const auto capacity = static_cast<int>(output.size() / (sizeof(opus_int16) * channels));
    std::vector<opus_int16> pcm(static_cast<std::size_t>(capacity) * channels);
    const auto* packet = reinterpret_cast<const unsigned char*>(input.data() + packet_header_size);
    const auto samples = decoder_->multistream
        ? opus_multistream_decode(decoder_->multistream, packet, static_cast<opus_int32>(size), pcm.data(), capacity, 0)
        : opus_decode(decoder_->opus, packet, static_cast<opus_int32>(size), pcm.data(), capacity, 0);
    if (samples < 0) {
        decoded.result = result_from_opus(samples);
        return decoded;
    }
    std::memcpy(output.data(), pcm.data(), static_cast<std::size_t>(samples) * channels * sizeof(opus_int16));
    decoded.consumed = static_cast<std::uint32_t>(packet_header_size + size);
    decoded.samples = static_cast<std::uint32_t>(samples);
    decoded.microseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count());
#else
    (void)output;
    (void)reset;
    decoded.result = result_unimplemented;
#endif
    return decoded;
}

IpcReply DecoderSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Opus decoder session belongs to another process");
    if (control(message)) {
        if (read(message, 16) == sfci && read(message, 24) == 3) return control_value(pointer_buffer_size, 2);
        return unsupported("Opus decoder control request is unsupported");
    }
    return method(message, nullptr);
}

IpcReply DecoderSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                              const GuestBuffers& buffers) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Opus decoder session belongs to another process");
    return method(message, &buffers);
}

IpcReply DecoderSession::method(const IpcMessage& message, const GuestBuffers* buffers) {
    const auto cmif = request_cmif(message, false);
    if (!cmif) return unsupported("Opus decoder request shape is unsupported");
    const char* detail = nullptr;
    auto reply = command(message, *cmif, 16, buffers, detail);
    return reply ? std::move(*reply) : unsupported(detail);
}

std::optional<IpcReply> DecoderSession::command(const IpcMessage& message, std::size_t cmif, std::size_t out,
                                                const GuestBuffers* buffers, const char*& detail) {
    const auto which = read(message, cmif + 8);
    // Commands 2, 3, 5, 7 and 9 are the multi-stream forms of 0, 1, 4, 6 and 8;
    // each decoder answers only its own kind.
    const bool multistream_command = which == 2 || which == 3 || which == 5 || which == 7 || which == 9;
    if (which < 10 && multistream_command != parameters_.multistream) {
        detail = parameters_.multistream ? "Single-stream command on a multi-stream Opus decoder"
                                         : "Multi-stream command on a single-stream Opus decoder";
        return std::nullopt;
    }
    switch (which) {
    case 1:  // SetContext
    case 3:  // SetContextForMultiStream
        // The context describes the decoder's work buffer to the decoding
        // hardware. The host decoder keeps its whole state host-side, so the
        // context has nothing to restore and later packets decode unchanged.
        return reply_with(0, out);
    case 0:  // DecodeInterleavedOld
    case 2:  // DecodeInterleavedForMultiStreamOld
    case 4:  // DecodeInterleavedWithPerfOld
    case 5:  // DecodeInterleavedForMultiStreamWithPerfOld
    case 6:  // DecodeInterleavedWithPerfAndResetOld(reset)
    case 7:  // DecodeInterleavedForMultiStreamWithPerfAndResetOld(reset)
    case 8:  // DecodeInterleaved(reset)
    case 9: {  // DecodeInterleavedForMultiStream(reset)
        // One input buffer (the packet) and one output buffer (the PCM).
        if (!buffers || buffers->size() != 2) {
            detail = "Opus decode needs its packet and PCM buffers";
            return std::nullopt;
        }
        const std::size_t in = (*buffers)[0].input ? 0 : 1, pcm_index = 1 - in;
        if (!(*buffers)[in].input || !(*buffers)[pcm_index].output) {
            detail = "Opus decode buffers have unexpected directions";
            return std::nullopt;
        }
        std::vector<std::byte> packet((*buffers)[in].size), pcm((*buffers)[pcm_index].size);
        if (!buffers->read(in, packet)) {
            detail = "Opus packet buffer is not readable";
            return std::nullopt;
        }
        const bool reset = which >= 6 && (read(message, cmif + 16) & 0xffu) != 0;
        const auto decoded = decode(packet, pcm, reset);
        if (decoded.result) return result_reply(decoded.result, out);
        if (!buffers->write_at(pcm_index, 0, std::span(pcm).first(decoded.samples * parameters_.channel_count * 2))) {
            detail = "Opus PCM buffer is not writable";
            return std::nullopt;
        }
        const bool performance = which > 3;
        auto reply = reply_with(performance ? 16 : 8, out);
        write(reply.bytes, out + 16, decoded.consumed);
        write(reply.bytes, out + 20, decoded.samples);
        if (performance) {
            write(reply.bytes, out + 24, static_cast<std::uint32_t>(decoded.microseconds));
            write(reply.bytes, out + 28, static_cast<std::uint32_t>(decoded.microseconds >> 32));
        }
        return reply;
    }
    default:
        detail = "Opus decoder command is not implemented";
        return std::nullopt;
    }
}

// ---------------------------------------------------------------- manager

ManagerSession::ManagerSession(std::uint64_t process_id, Console console)
    : ManagerSession(process_id, console, std::make_shared<Domain>()) {}

ManagerSession::ManagerSession(std::uint64_t process_id, Console console, std::shared_ptr<Domain> domain)
    : process_id_(process_id), console_(console), domain_(std::move(domain)) {}

IpcReply ManagerSession::unsupported(const char* detail) {
    domain_->diagnostic = detail ? detail : "Opus decoder manager request is unsupported";
    return {};
}

const char* ManagerSession::diagnostic() const {
    const std::lock_guard lock(domain_->mutex);
    return domain_->diagnostic;
}

std::size_t ManagerSession::domain_objects() const {
    const std::lock_guard lock(domain_->mutex);
    return domain_->objects.size();
}

IpcReply ManagerSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(domain_->mutex);
    domain_->diagnostic = nullptr;
    if (sender_pid != process_id_) return unsupported("Opus decoder manager session belongs to another process");
    return dispatch(message, nullptr);
}

IpcReply ManagerSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                              const GuestBuffers& buffers) {
    const std::lock_guard lock(domain_->mutex);
    domain_->diagnostic = nullptr;
    if (sender_pid != process_id_) return unsupported("Opus decoder manager session belongs to another process");
    return dispatch(message, &buffers);
}

IpcReply ManagerSession::dispatch(const IpcMessage& message, const GuestBuffers* buffers) {
    if (control(message)) {
        // QueryPointerBufferSize, and ConvertCurrentObjectToDomain: the manager
        // becomes object 1 of its domain.
        if (read(message, 16) != sfci) return unsupported("Opus decoder manager control request is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_value(pointer_buffer_size, 2);
        if (command == 0 && !domain_->converted) {
            domain_->converted = true;
            return control_value(1, 4);
        }
        if (command == 2) {
            // CloneCurrentObject: another session of the same domain.
            auto reply = control_value(0, 4);
            reply.transferred[0].reset(new ManagerSession(process_id_, console_, domain_));
            reply.transferred_count = 1;
            return reply;
        }
        return unsupported("Opus decoder manager control request is unsupported");
    }
    if (!domain_->converted) {
        const auto cmif = request_cmif(message, false);
        if (!cmif) return unsupported("Opus decoder manager request shape is unsupported");
        auto reply = manager_command(message, *cmif, 16, buffers);
        return reply ? std::move(*reply) : unsupported(domain_->diagnostic);
    }
    // A domain message: {type, input objects, data size} then the object id.
    // Type 2 without data releases the object; type 1 carries a CMIF request.
    const auto header = read(message, 16);
    if ((header & 0xff) == 2 && (header >> 16) == 0) {
        const auto id = read(message, 20);
        if (!domain_->objects.erase(id)) return unsupported("Opus decoder domain object is not open");
        return reply_with(0);
    }
    const auto cmif = request_cmif(message, true);
    if (!cmif || (read(message, *cmif - 16) & 0xffff) != 1)
        return unsupported("Opus decoder domain request shape is unsupported");
    const auto id = read(message, *cmif - 12);
    if (id == 1) {
        auto reply = manager_command(message, *cmif, 32, buffers);
        return reply ? std::move(*reply) : unsupported(domain_->diagnostic);
    }
    const auto found = domain_->objects.find(id);
    if (found == domain_->objects.end()) return unsupported("Opus decoder domain object is not open");
    const char* detail = nullptr;
    auto reply = found->second->command(message, *cmif, 32, buffers, detail);
    return reply ? std::move(*reply) : unsupported(detail);
}

std::optional<IpcReply> ManagerSession::open(std::shared_ptr<DecoderSession> decoder, std::size_t out) {
    if (decoder->result()) return result_reply(decoder->result(), out);
    if (out == 32) {
        // In a domain the decoder is a new object: the reply's domain
        // header counts one output object, whose id follows the CMIF header.
        const auto id = domain_->next_object++;
        domain_->objects.emplace(id, std::move(decoder));
        auto reply = reply_with(8, out);
        write(reply.bytes, 16, 1);
        write(reply.bytes, out + 16, id);
        return reply;
    }
    auto reply = reply_with(0, out);
    reply.transferred[0] = std::move(decoder);
    reply.transferred_count = 1;
    return reply;
}

std::optional<IpcReply> ManagerSession::manager_command(const IpcMessage& message, std::size_t cmif,
                                                        std::size_t out, const GuestBuffers* buffers) {
    const auto command = read(message, cmif + 8);
    switch (command) {
    case 1:    // GetWorkBufferSize(parameters)
    case 5:    // GetWorkBufferSizeEx(extended parameters)
    case 8: {  // GetWorkBufferSizeExEx(extended parameters)
        std::uint32_t size = 0;
        const auto backend = command == 8 ? SizeBackend::exex : SizeBackend::legacy;
        if (const auto failure = work_buffer_size(parameters_from(message, cmif + 16, command != 1), size,
                                                  console_, backend))
            return result_reply(failure, out);
        auto reply = reply_with(4, out);
        write(reply.bytes, out + 16, size);
        return reply;
    }
    case 3:    // GetWorkBufferSizeForMultiStream(parameters in a pointer buffer)
    case 7:    // GetWorkBufferSizeForMultiStreamEx(extended parameters in a pointer buffer)
    case 9: {  // GetWorkBufferSizeForMultiStreamExEx(extended parameters in a pointer buffer)
        const auto backend = command == 9 ? SizeBackend::exex : SizeBackend::legacy;
        if (console_ == Console::switch2 && backend == SizeBackend::exex) break;
        const auto parameters = multistream_parameters(buffers, command != 3);
        if (!parameters) {
            domain_->diagnostic = "Opus multi-stream parameters are not readable";
            return std::nullopt;
        }
        std::uint32_t size = 0;
        if (const auto failure = work_buffer_size(*parameters, size, console_, backend))
            return result_reply(failure, out);
        auto reply = reply_with(4, out);
        write(reply.bytes, out + 16, size);
        return reply;
    }
    case 0:    // OpenHardwareOpusDecoder(parameters, transfer memory size; transfer memory)
    case 4: {  // OpenHardwareOpusDecoderEx(extended parameters, transfer memory size; transfer memory)
        // The decoder keeps its state in host memory; the transfer memory only
        // declares the work buffer the guest lent, which must be large enough.
        const auto parameters = parameters_from(message, cmif + 16, command == 4);
        const auto lent = read(message, cmif + 16 + (command == 4 ? 16 : 8));
        // The open's own size check is not recovered; it admits the smallest
        // work buffer a size query reports for these parameters.
        std::uint32_t size = 0;
        if (const auto failure = work_buffer_size(parameters, size, console_))
            return result_reply(failure, out);
        if (lent < size) return result_reply(result_buffer_too_small, out);
        if (!decoding_available()) {
            domain_->diagnostic = "This build has no Opus decoder";
            return std::nullopt;
        }
        return open(std::make_shared<DecoderSession>(process_id_, parameters), out);
    }
    case 2:    // OpenHardwareOpusDecoderForMultiStream(transfer memory size; transfer memory, parameters)
    case 6: {  // OpenHardwareOpusDecoderForMultiStreamEx(transfer memory size; transfer memory, parameters)
        if (console_ != Console::switch1) break;
        const auto parameters = multistream_parameters(buffers, command == 6);
        if (!parameters) {
            domain_->diagnostic = "Opus multi-stream parameters are not readable";
            return std::nullopt;
        }
        const auto lent = read(message, cmif + 16);
        std::uint32_t size = 0;
        if (const auto failure = work_buffer_size(*parameters, size, console_))
            return result_reply(failure, out);
        if (lent < size) return result_reply(result_buffer_too_small, out);
        if (!decoding_available()) {
            domain_->diagnostic = "This build has no Opus decoder";
            return std::nullopt;
        }
        return open(std::make_shared<DecoderSession>(process_id_, *parameters), out);
    }
    default:
        break;
    }
    domain_->diagnostic = "Opus decoder manager command is not implemented";
    return std::nullopt;
}

} // namespace armigo::services::hwopus
