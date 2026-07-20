// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Structural validation checks that a Graph is a well-formed DAG
// description — name wiring, single assignment — *without* caring whether we
// have a kernel for each op_type yet (that is Session bind / PR5+).
//
// Spec: ONNX IR.md — Graphs are DAGs; values have unique definitions.

#pragma once

#include "eduort/graph.h"
#include "eduort/status.h"

namespace eduort {

// Check structural well-formedness of `graph`.
//
// Passes when:
//   - Every non-empty node input name is defined (graph input, initializer,
//     or some node's output)
//   - Each value name is produced by at most one node (SSA-style)
//   - Graph input / output / initializer names are non-empty where required
//
// Does *not* reject unknown op_types (design PR4: structural only).
// Cycles are reported by the topological sort (see topo_sort.h).
//
// MVP restriction: empty node *output* names are rejected. ONNX allows "" for
// omitted optional outputs (like empty inputs); no current MVP op needs that.
// Treat as skip when optional-output ops land (e.g. Dropout mask).
// Failures use ErrorCode::kModelLoad.
Status ValidateStructure(const Graph& graph);

}  // namespace eduort
