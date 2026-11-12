// ============================================================================
// Armigo Tests - Hardware Opus Decoder
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/hardware_opus.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#ifdef ARMIGO_HAS_OPUS
#include <opus.h>
#include <opus_multistream.h>
#endif

namespace {
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
namespace hw = armigo::services::hwopus;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}

std::uint32_t get(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}

// Guest memory for pointer buffers: one span at a fixed address.
struct Memory {
    std::uint64_t base = 0x10000;
    std::vector<std::byte> bytes = std::vector<std::byte>(0x1000);
};

bool read_memory(void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
    auto& memory = *static_cast<Memory*>(context);
    if (address < memory.base || address - memory.base + out.size() > memory.bytes.size()) return false;
    std::memcpy(out.data(), memory.bytes.data() + (address - memory.base), out.size());
    return true;
}

bool write_memory(void* context, std::uint64_t address, std::span<const std::byte> in) noexcept {
    auto& memory = *static_cast<Memory*>(context);
    if (address < memory.base || address - memory.base + in.size() > memory.bytes.size()) return false;
    std::memcpy(memory.bytes.data() + (address - memory.base), in.data(), in.size());
    return true;
}

IpcMessage request(std::uint32_t command, std::uint32_t words) {
    IpcMessage message{};
    message.fill(std::byte{0});
    put(message, 0, 4);
    put(message, 4, words);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command);
    return message;
}

void switch1_tests(std::uint64_t pid) {
    using hw::Console;
    // 23.0.1 legacy objects: single-stream 44408 and 53128 bytes, and
    // multi-stream objects of a fixed part plus each stream's decoder state.
    std::uint32_t size = 0;
    CHECK(hw::work_buffer_size({48'000, 1, false}, size, Console::switch1) == 0 && size == 44408 + 1920 + 0x600);
    CHECK(hw::work_buffer_size({48'000, 2, true}, size, Console::switch1) == 0 && size == 53128 + 11520 + 0x600);
    hw::DecoderParameters surround{48'000, 6, false, true, 4, 2, {}};
    CHECK(hw::work_buffer_size(surround, size, Console::switch1) == 0 &&
          size == 47392 + 2 * 17764 + 2 * 26484 + 6016 + 11520);
    CHECK(hw::work_buffer_size(surround, size, Console::switch2) == hw::result_unimplemented);
    // Switch 2's legacy objects stage eight fewer packet bytes and use a
    // smaller multi-stream layout with 1508 staging bytes per stream.
    using hw::SizeBackend;
    CHECK(hw::work_buffer_size({48'000, 2, false}, size, Console::switch1, SizeBackend::legacy) == 0 &&
          size == 58504);
    CHECK(hw::work_buffer_size({48'000, 2, false}, size, Console::switch2, SizeBackend::legacy) == 0 &&
          size == 58496);
    hw::DecoderParameters stereo_stream{48'000, 2, false, true, 1, 1, {}};
    CHECK(hw::work_buffer_size(stereo_stream, size, Console::switch1, SizeBackend::legacy) == 0 && size == 79252);
    CHECK(hw::work_buffer_size(stereo_stream, size, Console::switch2, SizeBackend::legacy) == 0 && size == 79236);
    // Stream counts that do not fit the channels report the rate result.
    hw::DecoderParameters crowded{48'000, 5, false, true, 4, 2, {}};
    CHECK(hw::work_buffer_size(crowded, size, Console::switch1) == hw::result_invalid_sample_rate);
    hw::DecoderParameters silent{48'000, 0, false, true, 1, 0, {}};
    CHECK(hw::work_buffer_size(silent, size, Console::switch1) == hw::result_invalid_channel_count);
    hw::DecoderParameters odd_rate{44'100, 2, false, true, 1, 1, {}};
    CHECK(hw::work_buffer_size(odd_rate, size, Console::switch1) == hw::result_invalid_sample_rate);

    Memory memory;
    armigo::horizon::GuestBuffers buffers(read_memory, write_memory, &memory);
    buffers.add({.address = memory.base, .size = 0x118, .input = true, .output = true});
    // Extended parameters: 48 kHz, 3 channels, 2 streams (1 stereo), large
    // frames off, mapping 0, 1, 2 at 0x18.
    auto put_word = [&](std::size_t offset, std::uint32_t value) { std::memcpy(memory.bytes.data() + offset, &value, 4); };
    put_word(0, 48'000);
    put_word(4, 3);
    put_word(8, 2);
    put_word(12, 1);
    for (std::uint8_t channel = 0; channel < 3; ++channel) memory.bytes[0x18 + channel] = std::byte{channel};
    const std::uint32_t expected = 47392 + 17764 + 26484 + 3008 + 5760;
    hw::ManagerSession manager(pid, Console::switch1);
    auto query = request(7, 8);
    auto sized = manager.request_with_buffers(query, pid, buffers);
    CHECK(sized.disposition == IpcDisposition::handled && get(sized.bytes, 24) == 0 && get(sized.bytes, 32) == expected);
    // The ExEx query answers the same decode object size.
    auto exex = manager.request_with_buffers(request(9, 8), pid, buffers);
    CHECK(exex.disposition == IpcDisposition::handled && get(exex.bytes, 32) == expected);
    auto single = request(8, 12);
    put(single, 32, 48'000);
    put(single, 36, 1);
    auto single_size = manager.request(single, pid);
    CHECK(single_size.disposition == IpcDisposition::handled && get(single_size.bytes, 32) == 44408 + 1920 + 0x600);
    // A Switch 2 manager answers the legacy multi-stream size queries with its
    // own layout; its ExEx query and multi-stream decoders are not served.
    hw::ManagerSession switch2(pid);
    const auto switch2_sized = switch2.request_with_buffers(query, pid, buffers);
    CHECK(switch2_sized.disposition == IpcDisposition::handled &&
          get(switch2_sized.bytes, 32) == 47384 + 17756 + 26476 + 3072 + 5760);
    CHECK(switch2.request_with_buffers(request(9, 8), pid, buffers).disposition == IpcDisposition::unsupported);
    CHECK(switch2.request_with_buffers(request(6, 9), pid, buffers).disposition == IpcDisposition::unsupported);
    auto open = request(6, 9);
    put(open, 32, expected - 1);
    CHECK(get(manager.request_with_buffers(open, pid, buffers).bytes, 24) == hw::result_buffer_too_small);
    put(open, 32, expected);
    auto opened = manager.request_with_buffers(open, pid, buffers);
#ifdef ARMIGO_HAS_OPUS
    CHECK(opened.disposition == IpcDisposition::handled && opened.transferred_count == 1);
    auto decoder = std::dynamic_pointer_cast<hw::DecoderSession>(opened.transferred[0]);
    CHECK(decoder != nullptr);
    if (!decoder) return;
    int error = 0;
    const unsigned char mapping[3] = {0, 1, 2};
    OpusMSEncoder* encoder = opus_multistream_encoder_create(48'000, 3, 2, 1, mapping, OPUS_APPLICATION_AUDIO, &error);
    CHECK(encoder && error == OPUS_OK);
    if (!encoder) return;
    std::vector<opus_int16> pcm(960 * 3);
    for (std::size_t i = 0; i < 960; ++i)
        pcm[3 * i] = pcm[3 * i + 1] = pcm[3 * i + 2] = static_cast<opus_int16>(8000 * std::sin(i * 0.05));
    std::vector<unsigned char> packet(0x600);
    const auto length =
        opus_multistream_encode(encoder, pcm.data(), 960, packet.data(), static_cast<opus_int32>(packet.size()));
    opus_multistream_encoder_destroy(encoder);
    CHECK(length > 0);
    std::vector<std::byte> input(8 + static_cast<std::size_t>(length));
    for (unsigned i = 0; i < 4; ++i) input[i] = std::byte(static_cast<std::uint32_t>(length) >> (24 - 8 * i));
    std::memcpy(input.data() + 8, packet.data(), static_cast<std::size_t>(length));
    std::vector<std::byte> output(1920 * 3 * 2);
    const auto decoded = decoder->decode(input, output, true);
    CHECK(decoded.result == 0 && decoded.samples == 960 && decoded.consumed == input.size());
    // A multi-stream decoder refuses the single-stream decode command.
    const char* detail = nullptr;
    CHECK(!decoder->command(request(8, 10), 16, 16, nullptr, detail) && detail);
    // SetContextForMultiStream succeeds and changes nothing: the next packet
    // decodes exactly as on a twin decoder that was never given a context.
    auto twin_opened = manager.request_with_buffers(open, pid, buffers);
    auto twin = twin_opened.transferred_count == 1
        ? std::dynamic_pointer_cast<hw::DecoderSession>(twin_opened.transferred[0]) : nullptr;
    CHECK(twin != nullptr);
    if (!twin) return;
    std::vector<std::byte> twin_output(output.size());
    CHECK(twin->decode(input, twin_output, true).result == 0);
    std::fill(memory.bytes.begin(), memory.bytes.begin() + 0x118, std::byte{0x5a});
    const char* context_detail = nullptr;
    const auto context = decoder->command(request(3, 8), 16, 16, &buffers, context_detail);
    CHECK(context && get(context->bytes, 24) == 0);
    CHECK(decoder->decode(input, output, false).result == 0);
    CHECK(twin->decode(input, twin_output, false).result == 0);
    CHECK(output == twin_output);
#else
    CHECK(opened.disposition == IpcDisposition::unsupported);
#endif
}
} // namespace

int main() {
    constexpr std::uint64_t pid = 0x81;
    // Work buffers: the console decoder state for the channel count, a frame of
    // output at the rate and the packet staging.
    std::uint32_t size = 0;
    CHECK(hw::work_buffer_size({48'000, 1, false}, size) == 0 && size == 24 + 18468 + 1920 + 0x600);
    CHECK(hw::work_buffer_size({48'000, 2, true}, size) == 0 && size == 24 + 27236 + 11520 + 0x600);
    CHECK(hw::work_buffer_size({24'000, 2, false}, size) == 0 && size == 24 + 27236 + 1920 + 0x600);
    CHECK(hw::work_buffer_size({44'100, 2, false}, size) == hw::result_invalid_sample_rate);
    CHECK(hw::work_buffer_size({48'000, 3, false}, size) == hw::result_invalid_channel_count);

    hw::ManagerSession manager(pid);
    // QueryPointerBufferSize (control) answers the reference's process default.
    auto control = request(3, 8);
    put(control, 0, 7);
    auto pointer = manager.request(control, pid);
    CHECK(pointer.disposition == IpcDisposition::handled && (get(pointer.bytes, 32) & 0xffff) == 0x8000);
    // GetWorkBufferSize(48 kHz stereo) sizes the legacy decode object; the
    // ExEx query (8) sizes the smaller ExEx decoder.
    auto query = request(1, 10);
    put(query, 32, 48'000);
    put(query, 36, 2);
    auto sized = manager.request(query, pid);
    CHECK(sized.disposition == IpcDisposition::handled && get(sized.bytes, 24) == 0 &&
          get(sized.bytes, 32) == 58496);
    auto exex_query = request(8, 12);
    put(exex_query, 32, 48'000);
    put(exex_query, 36, 2);
    auto exex_sized = manager.request(exex_query, pid);
    CHECK(exex_sized.disposition == IpcDisposition::handled && get(exex_sized.bytes, 24) == 0 &&
          get(exex_sized.bytes, 32) == 24 + 27236 + 3840 + 0x600);
    put(query, 36, 5);
    CHECK(get(manager.request(query, pid).bytes, 24) == hw::result_invalid_channel_count);
    // Unknown commands and other processes stay unsupported.
    CHECK(manager.request(request(100, 8), pid).disposition == IpcDisposition::unsupported && manager.diagnostic());
    CHECK(manager.request(query, pid + 1).disposition == IpcDisposition::unsupported);

    // OpenHardwareOpusDecoder: refused when the lent memory is too small.
    auto open = request(0, 12);
    put(open, 32, 48'000);
    put(open, 36, 2);
    put(open, 40, 0x100);
    CHECK(get(manager.request(open, pid).bytes, 24) == hw::result_buffer_too_small);
    put(open, 40, 0x10000);
    auto opened = manager.request(open, pid);
#ifdef ARMIGO_HAS_OPUS
    CHECK(opened.disposition == IpcDisposition::handled && opened.transferred_count == 1);
    auto decoder = std::dynamic_pointer_cast<hw::DecoderSession>(opened.transferred[0]);
    CHECK(decoder != nullptr);
    if (!decoder) return 1;
    // A 20 ms stereo sine packet decodes to 960 samples per channel.
    int error = 0;
    OpusEncoder* encoder = opus_encoder_create(48'000, 2, OPUS_APPLICATION_AUDIO, &error);
    CHECK(encoder && error == OPUS_OK);
    std::vector<opus_int16> pcm(960 * 2);
    for (std::size_t i = 0; i < 960; ++i)
        pcm[2 * i] = pcm[2 * i + 1] = static_cast<opus_int16>(8000 * std::sin(i * 0.05));
    std::vector<unsigned char> packet(0x600);
    const auto length = opus_encode(encoder, pcm.data(), 960, packet.data(), static_cast<opus_int32>(packet.size()));
    opus_encoder_destroy(encoder);
    CHECK(length > 0);
    std::vector<std::byte> input(8 + static_cast<std::size_t>(length));
    for (unsigned i = 0; i < 4; ++i) input[i] = std::byte(static_cast<std::uint32_t>(length) >> (24 - 8 * i));
    std::memcpy(input.data() + 8, packet.data(), static_cast<std::size_t>(length));
    std::vector<std::byte> output(1920 * 2 * 2);
    const auto decoded = decoder->decode(input, output, false);
    CHECK(decoded.result == 0 && decoded.samples == 960 && decoded.consumed == input.size());
    // An output too small for the packet's samples, and a packet larger than
    // the decoder's staging, are refused.
    std::vector<std::byte> small(100);
    CHECK(decoder->decode(input, small, false).result == hw::result_buffer_too_small);
    std::vector<std::byte> oversized(8 + 0x700);
    oversized[2] = std::byte{0x07};
    CHECK(decoder->decode(oversized, output, true).result == hw::result_buffer_too_small);
    std::vector<std::byte> tiny(8);
    CHECK(decoder->decode(tiny, output, false).result == hw::result_input_too_small);
    // Through a domain: the manager becomes object 1 and decoders new objects,
    // released by a close message.
    hw::ManagerSession domain(pid);
    auto convert = request(0, 8);
    put(convert, 0, 7);
    auto converted = domain.request(convert, pid);
    CHECK(converted.disposition == IpcDisposition::handled && get(converted.bytes, 32) == 1);
    IpcMessage in_domain{};
    in_domain.fill(std::byte{0});
    put(in_domain, 0, 4);
    put(in_domain, 4, 16);
    put(in_domain, 16, 1u | (0x20u << 16));
    put(in_domain, 20, 1);
    put(in_domain, 32, 0x49434653);
    put(in_domain, 36, 1);
    put(in_domain, 40, 0);
    put(in_domain, 48, 48'000);
    put(in_domain, 52, 2);
    put(in_domain, 56, 0x10000);
    auto object = domain.request(in_domain, pid);
    CHECK(object.disposition == IpcDisposition::handled && object.transferred_count == 0);
    CHECK(get(object.bytes, 16) == 1 && get(object.bytes, 32) == 0x4f434653 && get(object.bytes, 48) == 2);
    CHECK(domain.domain_objects() == 1);
    // A clone shares the domain's objects.
    auto clone = request(2, 8);
    put(clone, 0, 7);
    auto cloned = domain.request(clone, pid);
    CHECK(cloned.disposition == IpcDisposition::handled && cloned.transferred_count == 1);
    auto other = std::dynamic_pointer_cast<hw::ManagerSession>(cloned.transferred[0]);
    CHECK(other && other->domain_objects() == 1);
    IpcMessage close{};
    close.fill(std::byte{0});
    put(close, 0, 4);
    put(close, 4, 4);
    put(close, 16, 2);
    put(close, 20, 2);
    CHECK(domain.request(close, pid).disposition == IpcDisposition::handled && domain.domain_objects() == 0);
    CHECK(domain.request(close, pid).disposition == IpcDisposition::unsupported);
#else
    CHECK(opened.disposition == IpcDisposition::unsupported);
#endif
    switch1_tests(pid);
    std::printf("Hardware Opus failures: %u\n", failures);
    return failures ? 1 : 0;
}
