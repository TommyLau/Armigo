// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include "../facts.h"
#include "../resources.h"

namespace armigo::shader::pipeline::passes {
// These passes require valid SSA and publish changes only after validation succeeds.
// Cleanup preserves all effects and never evaluates floating arithmetic on the host.
[[nodiscard]] Status cleanup(Program&);
// Reconciles a fragment's depth with the same surface's stored texture value
// only for comparisons pairing those two data origins.
[[nodiscard]] Status snap_fragment_depth(Program&);
[[nodiscard]] Status legalize_nontexture(Program&, FactReader&, const MetalProfile&);
// Moves filtered texture samples out of the blocks their instruction
// predicate guards when their inputs are available before the guard;
// implicit-LOD and bias samples only when `implicit` is set.
[[nodiscard]] Status hoist_guarded_samples(Program&, bool implicit = false);
} // namespace armigo::shader::pipeline::passes
