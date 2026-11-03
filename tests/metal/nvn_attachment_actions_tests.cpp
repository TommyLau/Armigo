// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "nvn_attachment_actions.h"

#include <array>
#include <cstdio>

namespace {

using namespace armigo::platform::metal::nvn_driver;

int failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)

template <std::size_t N>
std::size_t landing(const std::array<bool, N>& foldable) {
    return clears_to_land(N, [&](std::size_t index) { return foldable[index]; });
}

void clear_landing() {
    // Nothing pending, or every clear folds: nothing lands first.
    CHECK(landing(std::array<bool, 0>{}) == 0);
    CHECK(landing(std::array{true, true, true}) == 0);
    // A clear the pass cannot fold lands with every clear recorded before it,
    // so overlapping clears keep their order; later folding clears still fold.
    CHECK(landing(std::array{true, false, true}) == 2);
    CHECK(landing(std::array{true, true, false}) == 3);
    CHECK(landing(std::array{false, true, true}) == 1);
    CHECK(landing(std::array{false, false}) == 2);
}

void depth_use() {
    CHECK(!uses_depth_stencil({false, false, false, false}));
    CHECK(uses_depth_stencil({true, false, false, false}));
    CHECK(uses_depth_stencil({false, true, false, false}));
    CHECK(uses_depth_stencil({false, false, true, false}));
    CHECK(uses_depth_stencil({false, false, false, true}));
}

void depth_omission() {
    constexpr DepthStencilUse idle{false, false, false, false};
    constexpr DepthStencilUse testing{true, true, false, false};
    CHECK(may_omit_depth_stencil(true, 1, false, idle, false));
    // Any reason to keep the attachment keeps it.
    CHECK(!may_omit_depth_stencil(false, 1, false, idle, false));
    CHECK(!may_omit_depth_stencil(true, 0, false, idle, false));
    CHECK(!may_omit_depth_stencil(true, 1, true, idle, false));
    CHECK(!may_omit_depth_stencil(true, 1, false, testing, false));
    CHECK(!may_omit_depth_stencil(true, 1, false, idle, true));
}

} // namespace

int main() {
    clear_landing();
    depth_use();
    depth_omission();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("nvn attachment action tests passed");
    return 0;
}
