// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Topological order is a list of node indices such that every
// producer runs before every consumer of a value. Kahn's algorithm finds
// such an order for a DAG and detects cycles.
//
// Spec: ONNX IR.md — graphs are directed acyclic graphs for inference.

#pragma once

#include <vector>

#include "eduort/graph.h"
#include "eduort/status.h"

namespace eduort {

// Compute a topological order of graph.nodes as indices into that vector.
// Safe to call standalone: enforces SSA among node outputs, rejects undefined
// non-seed inputs and self-loops. Prefer PrepareGraphStructure for Session.
//
// On cycle, SSA violation, or missing producer → ErrorCode::kModelLoad.
StatusOr<std::vector<int>> ComputeTopoOrder(const Graph& graph);

// ValidateStructure + ComputeTopoOrder, then store result in graph.topo_order.
// Clears topo_order first so a failed prepare never leaves a stale order.
// This is what Session::Create will call after loading (design checklist).
// Failures use ErrorCode::kModelLoad.
Status PrepareGraphStructure(Graph& graph);

}  // namespace eduort
