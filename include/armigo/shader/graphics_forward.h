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

#include <armigo/shader/graphics.h>
#include <armigo/shader/maxwell_forward_program.h>

namespace armigo::shader::graphics {

// Lowers a decoded guarded forward program into a typed SSA graph Program.
// This is the control-flow companion of lower(): it accepts the decoder's
// structurally reachable ForwardProgram view, validates the public mutable
// metadata independently of decoder provenance, and emits blocks through the
// same Builder primitives the legacy path uses. The output is an ordinary
// graph Program (one entry block, strictly forward terminators, a single
// final Return block); nothing here executes guest control flow.
//
// Lowering model. Forward guest-PC order merges all incoming block states.
// Each register, predicate and deferred output has a concrete SSA carrier
// and separate initialization validity. Joins reuse identical dominating
// carriers or emit typed predecessor-labelled phis, selecting validity by
// the same incoming edge. Only validity-proved cells are exposed to Builder
// reads. Facts use immutable PredicateId identities and intersect at joins.
// Unknown instruction guards emit a real branch/body/shared-continuation
// diamond; known-false guards read no sources. Numeric instructions reuse
// the existing Builder lowering after their guard decision is made.
//
// Admitted input. Instructions of every family the Builder or a registered
// family lowerer implements, IntegerSetPredicateRegister in the decoded
// NE/AND/PT-source profile (raw-word comparison; the signed field does not
// affect it; destination A is written before destination B and PT writes
// discard), direct ForwardBranch with any preserved guard, and Exit with the
// always flow test. PredicateSetPredicate and every other family are
// rejected as unsupported even under a statically false guard.
//
// Exits. Every guest EXIT reachable on a path snapshots the declared export
// interface at that point and jumps to one final Return block appended after
// all CFG blocks. Distinct carriers become predecessor-labelled ScalarPhi
// outputs; identical dominating carriers are reused. Every reachable exit
// must prove initialization of the complete declared export interface.
//
// Error mapping (codes reuse Error::Code):
//   UnsupportedStage       stage is not a graphics stage;
//   InvalidInstruction     malformed mutable metadata: unsorted, duplicated,
//                          misaligned, or uncontained pcs, family/payload
//                          mismatch, bad guard index, bad branch target,
//                          a structural successor missing from the list, or
//                          an input node outside the structural closure;
//   UnsupportedModifier    a known decoder family whose lowering profile is
//                          not implemented, or an unsupported instruction
//                          field, checked regardless of guard outcome;
//   MissingOutput          a reachable exit does not define the complete
//                          declared export interface; the Builder's own
//                          error is preserved verbatim;
//   MissingExit            no entry instruction, or a structural or
//                          executing path leaves the program region;
//   UninitializedRegister  a scalar register or predicate read on an
//                          executing path whose initialization is unproved;
//   Limit                  node, graph block, or initialization DAG cap hit;
//   Allocation             host allocation failure.
// Graph validation of the produced specification is always performed by
// make_graph; its codes are reported verbatim.
[[nodiscard]] LowerResult lower_forward(const maxwell::ForwardProgram& program,
                                        Limits limits = {});

}  // namespace armigo::shader::graphics
