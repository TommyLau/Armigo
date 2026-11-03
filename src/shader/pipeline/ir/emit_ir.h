// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Decoded blocks dispatch each instruction through the generated emitter
// table. Armigo's driver adds the frozen contract's
// predication rule: the instruction predicate is evaluated once and branches
// around the whole expansion. Flow/pop instructions are excluded from that
// wrapper for the flow
// lane's own handling.

#pragma once
#include "pipeline/decoder.h"
#include "pipeline/dispatch.h"

#include <span>

namespace armigo::shader::pipeline {
// emit_ir driven with lane-local bindings consulted before the checked global
// emitter table. Synthetic tests and staged bring-up use this entry point;
// the frozen emit_ir below dispatches only through the global table.
[[nodiscard]] Result<RegisterProgram> emit_ir_with(const DecodedProgram& decoded, FactReader& facts,
                                                   Limits limits, std::span<const EmitterBinding> local);
} // namespace armigo::shader::pipeline
