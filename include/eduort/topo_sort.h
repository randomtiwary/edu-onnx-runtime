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
// Requires ValidateStructure-level name wiring; still re-checks producers.
//
// On cycle or missing producer → kInvalidArgument / kModelLoad with message.
StatusOr<std::vector<int>> ComputeTopoOrder(const Graph& graph);

// ValidateStructure + ComputeTopoOrder, then store result in graph.topo_order.
// This is what Session::Create will call after loading (design checklist).
Status PrepareGraphStructure(Graph& graph);

}  // namespace eduort
