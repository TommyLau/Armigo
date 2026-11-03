// ============================================================================
// Armigo NVN - Texel Flush Certificates
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
//
// Certificates for guest texel flushes: see texflush.h. One record per texture
// serial holds a certificate per layer; a flush over a range of layers verifies
// them before anything is uploaded.

#include <armigo/graphics/texflush.h>

#include <armigo/memory/write_tracker.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace armigo::graphics::texflush {
namespace {

struct LayerCertificate {
    std::uint64_t stamp = 0;
    std::uint64_t hash = 0;
    bool valid = false;
};

struct Record {
    const std::byte* storage = nullptr;
    std::uint64_t layer_size = 0;
    std::uint32_t layers = 0;
    std::vector<LayerCertificate> certificates;
};

std::mutex records_mutex;
std::unordered_map<std::uint64_t, Record> certificates_by_serial;

std::uint64_t rotate_left(std::uint64_t value, int amount) noexcept {
    return value << amount | value >> (64 - amount);
}

std::uint64_t lane(std::uint64_t accumulator, std::uint64_t word) noexcept {
    return rotate_left(accumulator + word * 0xc2b2ae3d27d4eb4full, 31) * 0x9e3779b97f4a7c15ull;
}

bool span_ok(const std::byte* storage, std::uint64_t layer_size, std::uint32_t layers,
             std::uint32_t first, std::uint32_t count) noexcept {
    if (!storage || !layer_size || !layers || first > layers || count > layers - first)
        return false;
    const auto bytes = layer_size * layers;
    return bytes / layers == layer_size &&
        bytes <= std::numeric_limits<std::uintptr_t>::max() -
            reinterpret_cast<std::uintptr_t>(storage);
}

bool same_record(const Record* record, const std::byte* storage, std::uint64_t layer_size,
                 std::uint32_t layers) noexcept {
    return record && record->storage == storage && record->layer_size == layer_size &&
        record->layers == layers;
}

const std::byte* layer_pointer(const std::byte* storage, std::uint64_t layer_size,
                               std::uint32_t layer) noexcept {
    return storage + static_cast<std::size_t>(std::uint64_t{layer} * layer_size);
}

} // namespace

std::uint64_t hash(const std::byte* bytes, std::uint64_t size) noexcept {
    std::uint64_t a = 0x60ea27eeadc0b5d6ull ^ size;
    std::uint64_t b = 0xc2b2ae3d27d4eb4full, c = 0x165667b19e3779f9ull,
                 e = 0x27d4eb2f165667c5ull;
    const auto* p = reinterpret_cast<const unsigned char*>(bytes);
    std::uint64_t remaining = size;
    for (; remaining >= 32; p += 32, remaining -= 32) {
        std::uint64_t words[4];
        std::memcpy(words, p, 32);
        a = lane(a, words[0]);
        b = lane(b, words[1]);
        c = lane(c, words[2]);
        e = lane(e, words[3]);
    }
    std::uint64_t result = rotate_left(a, 1) + rotate_left(b, 7) + rotate_left(c, 12) +
        rotate_left(e, 18);
    for (; remaining >= 8; p += 8, remaining -= 8) {
        std::uint64_t word = 0;
        std::memcpy(&word, p, 8);
        result = lane(result, word);
    }
    if (remaining) {
        std::uint64_t word = 0;
        std::memcpy(&word, p, static_cast<std::size_t>(remaining));
        result = lane(result ^ remaining, word);
    }
    result ^= result >> 33;
    result *= 0xff51afd7ed558ccdull;
    result ^= result >> 33;
    result *= 0xc4ceb9fe1a85ec53ull;
    result ^= result >> 33;
    return result;
}

std::uint32_t target_layers(std::int32_t target, std::uint32_t height,
                            std::uint32_t depth, bool& known) noexcept {
    known = true;
    switch (target) {
    case 0x0:   // one-dimensional
    case 0x1:   // two-dimensional
    case 0x5:   // multisample two-dimensional
    case 0x7:   // rectangle
        return 1;
    case 0x3:   // one-dimensional array
        return height;
    case 0x4:   // two-dimensional array
    case 0x6:   // multisample two-dimensional array
        return depth;
    case 0x8:   // cubemap
        return 6;
    case 0x9:   // cubemap array
        if (depth > std::numeric_limits<std::uint32_t>::max() / 6) {
            known = false;
            return 0;
        }
        return depth * 6;
    default:
        known = false;
        return 0;
    }
}

Status layers(std::int32_t target, std::uint32_t height, std::uint32_t depth,
              bool has_layers, std::int32_t min_layer, std::int32_t layer_count,
              std::uint32_t& first, std::uint32_t& count,
              std::string_view* why) noexcept {
    if (why) *why = {};
    first = 0;
    count = 0;
    const auto invalid = [why](std::string_view reason) {
        if (why) *why = reason;
        return Status::invalid;
    };
    // The three-dimensional target ignores the view and covers one layer.
    if (target == 0x2) {
        count = 1;
        return Status::ok;
    }
    bool known_target = true;
    const auto layers_of_texture = target_layers(target, height, depth, known_target);
    if (!known_target)
        return invalid(target == 0x9 ? "cube-array layer count overflows"
                                     : "unknown texture target");
    if (!layers_of_texture) return invalid("texture has no layers");
    if (!has_layers) {
        count = layers_of_texture;
        return Status::ok;
    }
    if (min_layer < 0 || layer_count < 0) return invalid("negative view layer selection");
    if (static_cast<std::uint32_t>(min_layer) > layers_of_texture)
        return invalid("view base layer outside the texture");
    const auto selected = layer_count ? static_cast<std::uint32_t>(layer_count)
                                      : layers_of_texture - static_cast<std::uint32_t>(min_layer);
    if (selected > layers_of_texture - static_cast<std::uint32_t>(min_layer))
        return invalid("view layers exceed the texture");
    first = static_cast<std::uint32_t>(min_layer);
    count = selected;
    return count ? Status::ok : Status::empty;
}

void note(std::uint64_t serial, const std::byte* storage, std::uint64_t layer_size,
          std::uint32_t layers_of_texture, std::uint32_t first, std::uint32_t count,
          bool establish) noexcept {
    if (!serial || !span_ok(storage, layer_size, layers_of_texture, first, count)) return;
    std::vector<LayerCertificate> next(count ? count : 1);
    if (establish) {
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto* layer = layer_pointer(storage, layer_size, first + index);
            const auto stamp = memory::WriteTracker::now();
            bool certified = memory::WriteTracker::tracked(layer, layer_size) &&
                memory::WriteTracker::protect_checked(layer, layer_size) &&
                memory::WriteTracker::tracked(layer, layer_size) &&
                !memory::WriteTracker::written_since(layer, layer_size, stamp);
            if (certified)
                certified = !memory::WriteTracker::written_since(layer, layer_size, stamp) &&
                    memory::WriteTracker::tracked(layer, layer_size);
            next[index] = certified
                ? LayerCertificate{stamp, hash(layer, layer_size), true}
                : LayerCertificate{};
        }
    }

    const std::lock_guard lock(records_mutex);
    auto& record = certificates_by_serial[serial];
    if (!same_record(&record, storage, layer_size, layers_of_texture)) {
        record.storage = storage;
        record.layer_size = layer_size;
        record.layers = layers_of_texture;
        record.certificates.assign(layers_of_texture, LayerCertificate{});
    }
    for (std::uint32_t index = 0; index < count; ++index)
        record.certificates[first + index] = next[index];
}

Status verify(std::uint64_t serial, const std::byte* storage, std::uint64_t layer_size,
              std::uint32_t layers_of_texture, std::uint32_t first,
              std::uint32_t count) noexcept {
    if (!serial || !span_ok(storage, layer_size, layers_of_texture, first, count))
        return Status::untracked;
    std::vector<LayerCertificate> saved(count ? count : 1);
    {
        const std::lock_guard lock(records_mutex);
        const auto found = certificates_by_serial.find(serial);
        if (found == certificates_by_serial.end() ||
            !same_record(&found->second, storage, layer_size, layers_of_texture))
            return Status::untracked;
        for (std::uint32_t index = 0; index < count; ++index) {
            saved[index] = found->second.certificates[first + index];
            if (!saved[index].valid || !saved[index].stamp) return Status::untracked;
        }
    }
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto* layer = layer_pointer(storage, layer_size, first + index);
        if (!memory::WriteTracker::tracked(layer, layer_size) ||
            memory::WriteTracker::written_since(layer, layer_size, saved[index].stamp) ||
            hash(layer, layer_size) != saved[index].hash ||
            memory::WriteTracker::written_since(layer, layer_size, saved[index].stamp) ||
            !memory::WriteTracker::tracked(layer, layer_size))
            return Status::changed;
    }
    return Status::clean;
}

Status content(std::uint64_t serial, const std::byte* storage, std::uint64_t layer_size,
               std::uint32_t layers_of_texture, std::uint32_t first,
               std::uint32_t count) noexcept {
    if (!serial || !span_ok(storage, layer_size, layers_of_texture, first, count))
        return Status::untracked;
    std::vector<std::uint64_t> expected(count ? count : 1);
    bool known = true;
    {
        const std::lock_guard lock(records_mutex);
        const auto found = certificates_by_serial.find(serial);
        if (found == certificates_by_serial.end() ||
            !same_record(&found->second, storage, layer_size, layers_of_texture))
            known = false;
        else
            for (std::uint32_t index = 0; index < count; ++index) {
                const auto& certificate = found->second.certificates[first + index];
                if (!certificate.valid || !certificate.stamp) known = false;
                expected[index] = certificate.hash;
            }
    }
    if (!known) return Status::untracked;
    for (std::uint32_t index = 0; index < count; ++index)
        if (hash(layer_pointer(storage, layer_size, first + index), layer_size) !=
            expected[index])
            return Status::changed;
    return Status::clean;
}

void forget(std::uint64_t serial) noexcept {
    if (!serial) return;
    const std::lock_guard lock(records_mutex);
    certificates_by_serial.erase(serial);
}

std::size_t certificate_count() noexcept {
    const std::lock_guard lock(records_mutex);
    return certificates_by_serial.size();
}

} // namespace armigo::graphics::texflush
