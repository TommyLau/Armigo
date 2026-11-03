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
// A guest texel flush publishes CPU stores of a texture's layers to the GPU.
// The host keeps the guest's pool bytes and a separate Metal texture, so before
// it copies anything it has to know whether anything changed. A per-layer
// certificate answers that: the write-tracker generation when the layer was last
// published plus a hash of its bytes then. A flush whose layers all verify clean
// publishes nothing: no upload, no GPU wait, no readback.
//
// The hash is a corruption check only; it never waives a dirty, busy or
// untracked layer. Dirty generations come from the write tracker, which keeps
// one canonical record per host page across every registration.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace armigo::graphics::texflush {

enum class Status : std::uint8_t {
    // The range was accepted, or it is empty and there is nothing to do.
    ok,
    empty,
    // The description or the layer range cannot be represented.
    invalid,
    // Every given layer is unchanged since it was published.
    clean,
    // A layer changed, or is busy or untracked: the caller must publish.
    changed,
    // No certificate covers the layers.
    untracked,
};

// Whole-layer range a flush covers, with the guest API's selection rules: a
// three-dimensional texture ignores the view, and a view with a layer range
// selects those layers.
// The layer count of a texture target: one for single-layer targets, the height
// for a one-dimensional array, the depth for two-dimensional and multisample
// arrays, six for a cubemap and six per face for a cubemap array.
[[nodiscard]] std::uint32_t target_layers(std::int32_t target, std::uint32_t height,
                                          std::uint32_t depth, bool& known) noexcept;

[[nodiscard]] Status layers(std::int32_t target, std::uint32_t height,
                            std::uint32_t depth, bool has_layers,
                            std::int32_t min_layer, std::int32_t layer_count,
                            std::uint32_t& first, std::uint32_t& count,
                            std::string_view* why = nullptr) noexcept;

// Certifies the covered layers from their current bytes (establish) or clears
// their certificates. Establishing protects the pages so a later guest store is
// observed; a layer whose bytes cannot be certified is left uncertified.
void note(std::uint64_t serial, const std::byte* storage, std::uint64_t layer_size,
          std::uint32_t layers, std::uint32_t first, std::uint32_t count,
          bool establish) noexcept;

// Whether the covered layers are unchanged since they were published.
[[nodiscard]] Status verify(std::uint64_t serial, const std::byte* storage,
                            std::uint64_t layer_size, std::uint32_t layers,
                            std::uint32_t first, std::uint32_t count) noexcept;

// Hash-only comparison against the certificates, for diagnostics.
[[nodiscard]] Status content(std::uint64_t serial, const std::byte* storage,
                             std::uint64_t layer_size, std::uint32_t layers,
                             std::uint32_t first, std::uint32_t count) noexcept;

void forget(std::uint64_t serial) noexcept;

// Textures that currently hold certificates.
[[nodiscard]] std::size_t certificate_count() noexcept;

// Layer hash used by the certificates (exposed for tests).
[[nodiscard]] std::uint64_t hash(const std::byte* bytes, std::uint64_t size) noexcept;

} // namespace armigo::graphics::texflush
