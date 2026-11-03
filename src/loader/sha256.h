// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Internal SHA-256 interface for the loader. The implementation is provided
// by the host platform's system cryptography; no framework type crosses this
// boundary.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <span>

namespace armigo::loader::detail {

using Sha256Digest = std::array<std::byte, 32>;

/// Computes the SHA-256 digest of `data`, or returns nothing if the host
/// implementation reports a failure.
std::optional<Sha256Digest> sha256(std::span<const std::byte> data) noexcept;

} // namespace armigo::loader::detail
