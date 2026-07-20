// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Bind walk — for each node in topo order, try EPs until one creates
// a kernel. This is the "planning" half of Session::Create without Run.

#include "eduort/planner.h"

#include "eduort/macros.h"

#include <utility>

namespace eduort {

StatusOr<std::vector<NodeBinding>> BindKernels(
    const Graph& graph,
    const std::vector<IExecutionProvider*>& eps_in_priority_order) {
  if (eps_in_priority_order.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "BindKernels: empty execution provider list");
  }

  // If topo_order empty but nodes exist, caller forgot PrepareGraphStructure.
  if (graph.topo_order.empty() && !graph.nodes.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "BindKernels: graph.topo_order is empty; call "
                         "PrepareGraphStructure first");
  }

  std::vector<NodeBinding> bindings;
  bindings.reserve(graph.topo_order.size());

  for (const std::size_t ni : graph.topo_order) {
    if (ni >= graph.nodes.size()) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "BindKernels: topo_order index out of range");
    }
    const Node& node = graph.nodes[ni];
    bool bound = false;
    for (IExecutionProvider* ep : eps_in_priority_order) {
      if (ep == nullptr) {
        continue;
      }
      if (!ep->CanProduceKernel(node, graph.opset_version)) {
        continue;
      }
      StatusOr<std::unique_ptr<IKernel>> k =
          ep->CreateKernel(node, graph.opset_version);
      if (!k.ok()) {
        continue;  // try next EP
      }
      NodeBinding b;
      b.node_index = ni;
      b.ep_name = ep->Name();
      b.kernel = std::move(k).value();
      bindings.push_back(std::move(b));
      bound = true;
      break;
    }
    if (!bound) {
      return Status::Error(
          ErrorCode::kUnsupportedOperator,
          "unsupported operator '" + node.op_type + "' (domain='" +
              node.domain + "', model_opset=" +
              std::to_string(graph.opset_version) + "). See docs/operators.md.");
    }
  }

  return bindings;
}

}  // namespace eduort
