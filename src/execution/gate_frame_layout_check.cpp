// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Portable translation unit that only checks the gate frame layout. It keeps
// armigo_execution a real target on hosts without the AArch64 gate.
#include <armigo/execution/gate_frame.h>

namespace armigo::execution {
extern const unsigned kGateFrameLayoutChecked;
const unsigned kGateFrameLayoutChecked = ARMIGO_GATE_FRAME_SIZE;
}  // namespace armigo::execution
