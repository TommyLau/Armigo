// ============================================================================
// Armigo Tests - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/host1x/host1x.h>
#include <armigo/host1x/video_decoder.h>

#include "../../src/host1x/vp9.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace {
using namespace armigo::host1x;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #c); ++failures; } } while (false)

// Engine memory, syncpoints and failures the tests observe.
struct Harness {
    std::mutex mutex;
    std::map<std::uint64_t, std::vector<std::byte>> memory;  // device address -> bytes
    std::map<std::uint32_t, std::uint32_t> increments;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> waits;
    std::vector<std::pair<std::uint32_t, std::string>> failures;

    static bool read(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<Harness*>(raw);
        const std::lock_guard lock(self.mutex);
        for (const auto& [base, bytes] : self.memory)
            if (address >= base && address - base + out.size() <= bytes.size()) {
                std::memcpy(out.data(), bytes.data() + (address - base), out.size());
                return true;
            }
        return false;
    }
    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto& self = *static_cast<Harness*>(raw);
        const std::lock_guard lock(self.mutex);
        for (auto& [base, bytes] : self.memory)
            if (address >= base && address - base + in.size() <= bytes.size()) {
                std::memcpy(bytes.data() + (address - base), in.data(), in.size());
                return true;
            }
        return false;
    }
    static void increment(void* raw, std::uint32_t id) noexcept {
        auto& self = *static_cast<Harness*>(raw);
        const std::lock_guard lock(self.mutex);
        ++self.increments[id];
    }
    static bool wait(void* raw, std::uint32_t id, std::uint32_t threshold, std::stop_token) noexcept {
        auto& self = *static_cast<Harness*>(raw);
        const std::lock_guard lock(self.mutex);
        self.waits.emplace_back(id, threshold);
        return true;
    }
    static void fail(void* raw, std::uint32_t syncpoint, std::string detail) noexcept {
        auto& self = *static_cast<Harness*>(raw);
        const std::lock_guard lock(self.mutex);
        self.failures.emplace_back(syncpoint, std::move(detail));
    }
    Hub hub{{this, read, write}, {this, increment, wait}, {this, fail}};
};

constexpr std::uint32_t header(SubmissionMode mode, std::uint32_t method, std::uint32_t data) {
    return static_cast<std::uint32_t>(mode) << 28 | (method & 0xfffu) << 16 | (data & 0xffffu);
}
constexpr std::uint32_t set_class(ClassId id) {
    return header(SubmissionMode::set_class, 0, static_cast<std::uint32_t>(id) << 6);
}

// Every submission mode reaches the engine's host-interface registers; an
// increment completes its syncpoint and the control class waits for one.
void test_command_stream() {
    Harness h;
    h.hub.start_device(3, ChannelType::video_compositor, 21);
    h.hub.push(3, {
        set_class(ClassId::video_compositor),
        header(SubmissionMode::non_incrementing, 0x00, 2), 21, 21,   // two increments of syncpoint 21
        header(SubmissionMode::immediate, 0x00, 22),                 // one increment of syncpoint 22
        set_class(ClassId::control),
        header(SubmissionMode::incrementing, 0x4e, 1), 9,            // payload 9
        header(SubmissionMode::immediate, 0x08, 21),                 // wait for syncpoint 21 to reach 9
        set_class(ClassId::video_compositor),
        header(SubmissionMode::mask, 0x00, 0x1), 23,                 // masked write: one increment of 23
    });
    h.hub.drain(3);
    {
        const std::lock_guard lock(h.mutex);
        CHECK(h.increments[21] == 2 && h.increments[22] == 1 && h.increments[23] == 1);
        CHECK(h.waits.size() == 1 && h.waits[0] == std::make_pair(21u, 9u));
        CHECK(h.failures.empty());
    }
    // An unknown submission mode stops the channel and reports its syncpoint.
    h.hub.push(3, {header(SubmissionMode::restart, 0, 0)});
    h.hub.drain(3);
    {
        const std::lock_guard lock(h.mutex);
        CHECK(h.failures.size() == 1 && h.failures[0].first == 21);
    }
    h.hub.stop_device(3);
    CHECK(!h.hub.push(3, {}));
}

// The decoder engine refuses codecs other than VP9 explicitly.
void test_decoder_codec() {
    Harness h;
    h.hub.start_device(4, ChannelType::video_decoder, 30);
    h.hub.push(4, {
        set_class(ClassId::video_decoder),
        header(SubmissionMode::incrementing, 0x10, 2), 0x80, 0x3,  // method 0: codec register, method 1: H.264
    });
    h.hub.drain(4);
    const std::lock_guard lock(h.mutex);
    CHECK(h.failures.size() == 1 && h.failures[0].first == 30 &&
          h.failures[0].second.find("codec 3") != std::string::npos);
}

// The frame queue finds the decoder channel holding a named surface. A take
// for a surface supplies that surface's latest picture and drops the older
// pictures queued before it (the player skipped them); a surface with no
// queued picture (a picture shown again from an earlier surface) takes the
// channel's oldest picture.
void test_frame_queue() {
    FrameQueue queue;
    queue.open(1);
    queue.open(2);
    auto a = std::make_shared<DecodedFrame>();
    auto b = std::make_shared<DecodedFrame>();
    auto c = std::make_shared<DecodedFrame>();
    auto d = std::make_shared<DecodedFrame>();
    queue.push(1, 0x1000, a);
    queue.push(1, 0x2000, b);
    queue.push(2, 0x9000, c);
    const auto first = queue.find_decoder(0x2000);
    CHECK(first == 1 && queue.find_decoder(0x9000) == 2 && queue.find_decoder(0x3000) == -1);
    // In step: each compose names the oldest queued surface.
    CHECK(queue.take(first, 0x1000) == a);
    CHECK(queue.take(first, 0x2000) == b);
    CHECK(queue.take(first, 0x1000) == nullptr);
    // Behind: composes skipped two pictures; naming the newest surface catches up.
    queue.push(1, 0x1000, a);
    queue.push(1, 0x2000, b);
    queue.push(1, 0x3000, d);
    CHECK(queue.take(1, 0x3000) == d);
    CHECK(queue.take(1, 0x1000) == nullptr);
    // A reused surface supplies its latest picture.
    queue.push(1, 0x1000, a);
    queue.push(1, 0x2000, b);
    queue.push(1, 0x1000, d);
    CHECK(queue.take(1, 0x1000) == d && queue.take(1, 0x2000) == nullptr);
    // A picture shown again from an earlier surface: the oldest queued picture.
    queue.push(1, 0x2000, b);
    queue.push(1, 0x3000, d);
    CHECK(queue.take(1, 0x5000) == b && queue.take(1, 0x5000) == d);
    CHECK(queue.take(2, 0x9000) == c);
    CHECK(queue.take(-1, 0x9000) == nullptr);
    queue.close(1);
    queue.push(1, 0x1000, a);
    CHECK(queue.find_decoder(0x1000) == -1 && queue.take(1, 0x1000) == nullptr);
    CHECK(!queue.is_open(1) && queue.is_open(2) && !queue.is_open(-1));
}

// VP9 boolean decoder (specification 9.2), to read back the range writer.
struct BoolReader {
    const std::vector<std::uint8_t>& data;
    std::size_t position = 0;
    std::uint64_t value = 0;
    std::uint32_t range = 255;
    std::int32_t bits = -8;
    explicit BoolReader(const std::vector<std::uint8_t>& bytes) : data(bytes) { fill(); read(128); }
    void fill() {
        while (bits < 0) {
            value = (value << 8) | (position < data.size() ? data[position] : 0);
            ++position;
            bits += 8;
        }
    }
    bool read(std::uint32_t probability) {
        const std::uint32_t split = 1 + (((range - 1) * probability) >> 8);
        const std::uint64_t big_split = std::uint64_t{split} << bits;
        bool bit;
        if (value < big_split) {
            range = split;
            bit = false;
        } else {
            range -= split;
            value -= big_split;
            bit = true;
        }
        while (range < 128) {
            range <<= 1;
            --bits;
            if (bits < 0) fill();
        }
        return bit;
    }
};

void test_range_writer() {
    Vp9RangeWriter writer;
    std::vector<std::pair<bool, std::int32_t>> written;
    std::uint32_t seed = 12345;
    for (int i = 0; i < 2000; ++i) {
        seed = seed * 1664525u + 1013904223u;
        const bool bit = (seed >> 16) & 1;
        const std::int32_t probability = 1 + static_cast<std::int32_t>((seed >> 20) % 255);
        writer.write(bit, probability);
        written.emplace_back(bit, probability);
    }
    writer.end();
    BoolReader reader(writer.bytes());
    bool same = true;
    for (const auto& [bit, probability] : written)
        same = same && reader.read(static_cast<std::uint32_t>(probability)) == bit;
    CHECK(same);
}

// A key frame: the composed picture starts with the uncompressed header
// (marker, profile 0, shown key frame, sync code, size) and ends with the
// guest's tile data. The next execution composes it again, one execution late,
// shown because the next picture reports its predecessor as shown.
void test_composer() {
    Harness h;
    constexpr std::uint64_t picture_address = 0x100000, probabilities_address = 0x200000,
                             bitstream_address = 0x300000;
    Vp9GuestPicture picture{};
    picture.bitstream_size = 4;
    picture.current_frame_size = {320, 180, 320, 320};
    picture.flags = 1 | 16;  // key frame; the previous picture was shown
    picture.base_q_index = 60;
    picture.tx_mode = 3;
    picture.log2_tile_cols = 0;
    h.memory[picture_address].resize(sizeof picture);
    std::memcpy(h.memory[picture_address].data(), &picture, sizeof picture);
    h.memory[probabilities_address].resize(sizeof(Vp9GuestProbabilities));
    h.memory[bitstream_address] = {std::byte{0xde}, std::byte{0xad}, std::byte{0xbe}, std::byte{0xef}};
    DecoderRegisters registers;
    registers.values[DecoderRegisters::picture_info] = picture_address >> 8;
    registers.values[DecoderRegisters::vp9_probabilities] = probabilities_address >> 8;
    registers.values[DecoderRegisters::bitstream] = bitstream_address >> 8;
    Vp9Composer composer(h.hub.memory());
    std::vector<std::uint8_t> composed;
    bool hidden = true;
    std::string failure;
    CHECK(composer.compose(registers, composed, hidden, failure) && !hidden);
    CHECK(composed.size() > 10);
    if (composed.size() <= 10) return;
    CHECK(composed[0] == 0x82);  // marker 2, profile 0, no existing frame, key frame, shown, not error resilient
    CHECK(composed[1] == 0x49 && composed[2] == 0x83 && composed[3] == 0x42);  // sync code
    // Colour space (3 bits) and range (1 bit) are zero; width - 1 and height - 1 follow.
    const std::uint32_t size_bits = (std::uint32_t{composed[4]} << 24) | (std::uint32_t{composed[5]} << 16) |
                                    (std::uint32_t{composed[6]} << 8) | composed[7];
    CHECK(((size_bits >> 12) & 0xffffu) == 319);
    CHECK(composed[composed.size() - 4] == 0xde && composed.back() == 0xef);
    std::vector<std::uint8_t> again;
    CHECK(composer.compose(registers, again, hidden, failure) && again == composed);
    // An unreadable picture description is an explicit failure.
    registers.values[DecoderRegisters::picture_info] = 0x900000 >> 8;
    CHECK(!composer.compose(registers, composed, hidden, failure) &&
          failure.find("picture description") != std::string::npos);
}

// The platform decoder exists for VP9 and rejects data that is not VP9.
void test_platform_decoder() {
    std::string failure;
    auto decoder = make_vp9_decoder(failure);
    CHECK(decoder != nullptr);
    if (!decoder) return;
    const std::vector<std::uint8_t> garbage(64, 0x5a);
    std::vector<std::shared_ptr<const DecodedFrame>> shown;
    CHECK(!decoder->decode(garbage, 64, 64, shown, failure) && shown.empty());
    CHECK(failure.find("could not be created") == std::string::npos);
    CHECK(!decoder->decode(garbage, 0, 64, shown, failure));
}

// Compositor setup shared by the tests below: an 8x4 picture with luma 40
// and chroma U 50, V 60 queued by decoder channel 9 for luma surface
// 0x500000, and a configuration whose slot 0 converts it through
// R = Y, G = U, B = 2V - Y + 8 (10-bit units) into the output surface.
struct CompositorSetup {
    static constexpr std::uint64_t config = 0x600000, picture = 0x500000, output = 0x700000;
    static constexpr std::uint32_t fd = 5, decoder = 9, syncpoint = 40;
    Harness h;

    static std::uint64_t matrix_word(std::int32_t row0, std::int32_t row1, std::int32_t row2, std::uint64_t top) {
        const auto bits = [](std::int32_t value) { return static_cast<std::uint64_t>(value) & 0xfffffu; };
        return bits(row0) | bits(row1) << 20 | bits(row2) << 40 | top << 60;
    }
    template <typename T> void put(std::vector<std::byte>& bytes, std::size_t offset, T value) {
        std::memcpy(bytes.data() + offset, &value, sizeof value);
    }

    explicit CompositorSetup(std::uint32_t format, std::uint32_t block_kind, std::size_t output_size) {
        auto frame = std::make_shared<DecodedFrame>();
        frame->width = 8;
        frame->height = 4;
        frame->luma_stride = 8;
        frame->chroma_stride = 8;
        frame->luma.assign(32, 40);
        frame->chroma.assign(16, 0);
        for (std::size_t i = 0; i < 16; i += 2) frame->chroma[i] = 50, frame->chroma[i + 1] = 60;
        h.hub.frames().open(decoder);
        h.hub.frames().push(decoder, picture, frame);

        auto& bytes = h.memory[config];
        bytes.assign(0x610, std::byte{});
        put<std::uint32_t>(bytes, 0x18, 7u << 16);                       // target rect x 0..7
        put<std::uint32_t>(bytes, 0x1c, 3u << 16);                       // target rect y 0..3
        put<std::uint32_t>(bytes, 0x20, format | block_kind << 11);     // output format, layout, 1 GOB block
        for (std::size_t offset : {0x24, 0x28}) put<std::uint32_t>(bytes, offset, 7u | 3u << 14);  // 8x4
        put<std::uint32_t>(bytes, 0x2c, 3u | 1u << 14);                  // chroma 4x2
        const std::size_t slot = 0x90;
        put<std::uint64_t>(bytes, slot + 0x00, 1);                        // enabled, progressive
        put<std::uint64_t>(bytes, slot + 0x10, 1023ull << 10 | 1023ull << 32);  // clamp 0..1023, alpha 1023
        put<std::uint64_t>(bytes, slot + 0x20, 7ull << 32);               // source x 0..7
        put<std::uint64_t>(bytes, slot + 0x28, 3ull << 32);               // source y 0..3
        put<std::uint64_t>(bytes, slot + 0x30, 7ull << 16 | 3ull << 48);  // destination 0..7, 0..3
        put<std::uint32_t>(bytes, slot + 0x44, 7u | 3u << 14);            // slot surface 8x4
        put<std::uint64_t>(bytes, slot + 0x60, matrix_word(256, 0, -256, 0));
        put<std::uint64_t>(bytes, slot + 0x68, matrix_word(0, 256, 0, 0x8));  // enable
        put<std::uint64_t>(bytes, slot + 0x70, matrix_word(0, 0, 512, 0));
        put<std::uint64_t>(bytes, slot + 0x78, matrix_word(0, 0, 8 << 8, 0));
        h.memory[output].assign(output_size, std::byte{0xcc});
        h.memory[output + 0x10000].assign(output_size, std::byte{0xcc});
        h.hub.start_device(fd, ChannelType::video_compositor, syncpoint);
    }

    void execute() {
        const auto method = [](std::uint32_t index, std::uint32_t value) {
            return std::vector<std::uint32_t>{header(SubmissionMode::incrementing, 0x10, 2), index, value};
        };
        std::vector<std::uint32_t> words{set_class(ClassId::video_compositor)};
        for (const auto& part : {method(0x100, picture >> 8), method(0x1c2, config >> 8),
                                 method(0x1c8, output >> 8), method(0x1c9, (output + 0x10000) >> 8),
                                 method(0xc0, 0)})
            words.insert(words.end(), part.begin(), part.end());
        words.push_back(header(SubmissionMode::immediate, 0x00, syncpoint));
        h.hub.push(fd, words);
        h.hub.drain(fd);
    }
    std::uint8_t at(std::uint64_t base, std::size_t offset) {
        const std::lock_guard lock(h.mutex);
        return static_cast<std::uint8_t>(h.memory[base][offset]);
    }
};

// A pitch RGBA output: every pixel converted, rows 32 bytes apart, and the
// syncpoint after the execution completes. A second execution without a new
// picture writes the previous output again.
void test_compositor_pitch() {
    CompositorSetup setup(31, 0, 32 * 4);
    setup.execute();
    const std::uint8_t expected_blue = (2 * 240 - 160 + 8) >> 2;
    bool all = true;
    for (std::size_t y = 0; y < 4; ++y)
        for (std::size_t x = 0; x < 8; ++x) {
            const auto offset = y * 32 + x * 4;
            all = all && setup.at(setup.output, offset) == 40 && setup.at(setup.output, offset + 1) == 50 &&
                  setup.at(setup.output, offset + 2) == expected_blue && setup.at(setup.output, offset + 3) == 255;
        }
    CHECK(all);
    {
        const std::lock_guard lock(setup.h.mutex);
        CHECK(setup.h.failures.empty() && setup.h.increments[CompositorSetup::syncpoint] == 1);
    }
    {
        const std::lock_guard lock(setup.h.mutex);
        setup.h.memory[CompositorSetup::output].assign(32 * 4, std::byte{0});
    }
    setup.execute();
    CHECK(setup.at(setup.output, 0) == 40 && setup.at(setup.output, 31 * 4 + 3) == 255);
}

// Block-linear BGRA output lands at GOB positions: row 1 starts 16 bytes in,
// pixel 4 of row 0 32 bytes in and row 2 64 bytes in.
void test_compositor_block_linear() {
    CompositorSetup setup(32, 1, 512);
    setup.execute();
    const std::uint8_t expected_blue = (2 * 240 - 160 + 8) >> 2;
    CHECK(setup.at(setup.output, 0) == expected_blue && setup.at(setup.output, 2) == 40);
    CHECK(setup.at(setup.output, 16) == expected_blue && setup.at(setup.output, 32) == expected_blue);
    CHECK(setup.at(setup.output, 64) == expected_blue);  // row 2 of the GOB
    const std::lock_guard lock(setup.h.mutex);
    CHECK(setup.h.failures.empty());
}

// 4:2:0 pitch output writes luma and the interleaved chroma plane.
void test_compositor_n420() {
    CompositorSetup setup(68, 0, 64);
    setup.execute();
    CHECK(setup.at(setup.output, 0) == 40 && setup.at(setup.output, 16 + 7) == 40);
    const auto chroma = setup.output + 0x10000;
    const std::uint8_t expected_blue = (2 * 240 - 160 + 8) >> 2;
    CHECK(setup.at(chroma, 0) == 50 && setup.at(chroma, 1) == expected_blue && setup.at(chroma, 16 + 6) == 50);
    const std::lock_guard lock(setup.h.mutex);
    CHECK(setup.h.failures.empty());
}

// Nothing is written before any picture arrives; an output format the
// compositor does not implement fails the channel explicitly.
// A new movie opens a new decoder channel: once the old one closes, the
// compositor takes its pictures from the channel that now has them.
void test_compositor_new_decoder() {
    CompositorSetup setup(31, 0, 32 * 4);
    setup.execute();
    const auto first = setup.at(CompositorSetup::output, 0);
    setup.h.hub.frames().close(CompositorSetup::decoder);
    constexpr std::uint32_t next = CompositorSetup::decoder + 1;
    auto frame = std::make_shared<DecodedFrame>();
    frame->width = 8;
    frame->height = 4;
    frame->luma_stride = 8;
    frame->chroma_stride = 8;
    frame->luma.assign(32, 200);
    frame->chroma.assign(16, 128);
    setup.h.hub.frames().open(next);
    setup.h.hub.frames().push(next, CompositorSetup::picture, frame);
    setup.execute();
    CHECK(setup.at(CompositorSetup::output, 0) != first);
    CHECK(setup.h.hub.frames().take(next, CompositorSetup::picture) == nullptr);  // the new channel's picture was used
}

// A new decoder channel takes over a surface while the earlier channel stays
// open: the compositor follows the channel holding the surface it names.
void test_compositor_open_channel_switch() {
    CompositorSetup setup(31, 0, 32 * 4);
    setup.execute();
    const auto first = setup.at(CompositorSetup::output, 0);
    constexpr std::uint32_t next = CompositorSetup::decoder + 1;
    auto frame = std::make_shared<DecodedFrame>();
    frame->width = 8;
    frame->height = 4;
    frame->luma_stride = 8;
    frame->chroma_stride = 8;
    frame->luma.assign(32, 200);
    frame->chroma.assign(16, 128);
    setup.h.hub.frames().open(next);
    setup.h.hub.frames().push(next, CompositorSetup::picture, frame);
    CHECK(setup.h.hub.frames().is_open(CompositorSetup::decoder));
    setup.execute();
    CHECK(setup.at(CompositorSetup::output, 0) != first);
    CHECK(setup.h.hub.frames().take(next, CompositorSetup::picture) == nullptr);
}

void test_compositor_failures() {
    {
        CompositorSetup setup(31, 0, 32 * 4);
        (void)setup.h.hub.frames().take(CompositorSetup::decoder, CompositorSetup::picture);
        setup.execute();
        CHECK(setup.at(setup.output, 0) == 0xcc);
        const std::lock_guard lock(setup.h.mutex);
        CHECK(setup.h.failures.empty());
    }
    CompositorSetup setup(4, 0, 32 * 4);  // R8
    setup.execute();
    const std::lock_guard lock(setup.h.mutex);
    CHECK(setup.h.failures.size() == 1 && setup.h.failures[0].first == CompositorSetup::syncpoint &&
          setup.h.failures[0].second.find("output format 4") != std::string::npos);
}

} // namespace

int main() {
    test_command_stream();
    test_decoder_codec();
    test_frame_queue();
    test_range_writer();
    test_composer();
    test_platform_decoder();
    test_compositor_pitch();
    test_compositor_block_linear();
    test_compositor_n420();
    test_compositor_new_decoder();
    test_compositor_open_channel_switch();
    test_compositor_failures();
    if (failures) return 1;
    std::puts("Multimedia engine checks passed");
    return 0;
}
