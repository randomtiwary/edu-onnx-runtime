// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: The planner binds each topo-ordered node to a concrete kernel on
// some EP. Full Session::Run is PR9a; this PR only builds the binding list.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "eduort/execution_provider.h"
#include "eduort/graph.h"
#include "eduort/kernel.h"
#include "eduort/status.h"

namespace eduort {

struct NodeBinding {
  std::size_t node_index = 0;
  std::string ep_name;
  std::unique_ptr<IKernel> kernel;
};

// For each node in graph.topo_order (must be non-empty unless graph has no
// nodes), try EPs in order and create a kernel. Fails with kUnsupportedOperator
// if no EP can produce one. Constant nodes are still bound in PR5 if registered;
// PR9a will fold Constants before bind.
StatusOr<std::vector<NodeBinding>> BindKernels(
    const Graph& graph,
    const std::vector<IExecutionProvider*>& eps_in_priority_order);

}  // namespace eduort
