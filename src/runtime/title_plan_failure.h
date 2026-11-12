// ============================================================================
// Armigo Runtime - Native Process Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/compat/title_plan.h>
#include <armigo/runtime/launch_result.h>

namespace armigo::runtime {

// The launch error a title planning failure reports, for every launcher.
inline LaunchError launch_error(const compat::TitlePlanFailure& failure) noexcept {
    switch (failure.kind) {
        case compat::TitlePlanFailure::Kind::invalid_input: return LaunchError::invalid_input;
        case compat::TitlePlanFailure::Kind::artifact_mismatch: return LaunchError::artifact_mismatch;
        case compat::TitlePlanFailure::Kind::unsupported: return LaunchError::unsupported;
    }
    return LaunchError::invalid_input;
}

} // namespace armigo::runtime
