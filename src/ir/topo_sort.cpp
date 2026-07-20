// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Kahn's algorithm (1962) for topological sort:
//   1. Count incoming edges for each node.
//   2. Start with nodes that have in-degree 0 (ready).
//   3. "Emit" a ready node; reduce in-degree of its consumers; enqueue zeros.
//   4. If we emit fewer nodes than exist, there is a cycle.
//
// Edges come from *values*: if node A produces "T" and node B consumes "T",
// there is an edge A → B (A must run first).

#include "eduort/topo_sort.h"

#include "eduort/macros.h"
#include "eduort/validate.h"

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace eduort {
namespace {

// Build producer map: value name → node index that defines it.
// Enforces SSA (duplicate outputs → error) so standalone ComputeTopoOrder is safe.
StatusOr<std::unordered_map<std::string, int>> BuildProducerMap(
    const Graph& graph) {
  std::unordered_map<std::string, int> producer;
  for (int ni = 0; ni < static_cast<int>(graph.nodes.size()); ++ni) {
    for (const std::string& out :
         graph.nodes[static_cast<std::size_t>(ni)].outputs) {
      if (out.empty()) {
        continue;  // empty outputs: ValidateStructure rejects; skip here
      }
      auto [it, inserted] = producer.emplace(out, ni);
      if (!inserted) {
        return Status::Error(
            ErrorCode::kModelLoad,
            "value '" + out + "' is produced by multiple nodes (indices " +
                std::to_string(it->second) + " and " + std::to_string(ni) +
                "); ONNX graphs must be single-assignment");
      }
    }
  }
  return producer;
}

bool IsSeedValue(const Graph& graph, const std::string& name) {
  if (graph.initializers.count(name) != 0) {
    return true;
  }
  for (const std::string& gin : graph.graph_inputs) {
    if (gin == name) {
      return true;
    }
  }
  return false;
}

}  // namespace

StatusOr<std::vector<int>> ComputeTopoOrder(const Graph& graph) {
  const int n = static_cast<int>(graph.nodes.size());
  if (n == 0) {
    return std::vector<int>{};
  }

  EDUORT_ASSIGN_OR_RETURN(const auto producer, BuildProducerMap(graph));

  // adjacency[u] = list of nodes that depend on u (consumers).
  std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
  std::vector<int> indeg(static_cast<std::size_t>(n), 0);

  for (int ni = 0; ni < n; ++ni) {
    const Node& node = graph.nodes[static_cast<std::size_t>(ni)];
    // Unique predecessors so multi-input from same producer counts once.
    std::unordered_set<int> seen_pred;
    for (const std::string& in : node.inputs) {
      if (in.empty()) {
        continue;
      }
      auto it = producer.find(in);
      if (it == producer.end()) {
        // Must be graph input / initializer seed.
        if (!IsSeedValue(graph, in)) {
          return Status::Error(ErrorCode::kModelLoad,
                               "topo: undefined value '" + in + "'");
        }
        continue;
      }
      const int pred = it->second;
      if (pred == ni) {
        return Status::Error(ErrorCode::kModelLoad,
                             "node '" + node.name + "' consumes its own output '" +
                                 in + "' (self-loop)");
      }
      if (!seen_pred.insert(pred).second) {
        continue;  // already counted edge from this predecessor
      }
      adj[static_cast<std::size_t>(pred)].push_back(ni);
      indeg[static_cast<std::size_t>(ni)] += 1;
    }
  }

  // LEARNER: std::queue for the "ready set" — any order among ready nodes is a
  // valid topo order; we use FIFO for determinism.
  std::queue<int> ready;
  for (int i = 0; i < n; ++i) {
    if (indeg[static_cast<std::size_t>(i)] == 0) {
      ready.push(i);
    }
  }

  std::vector<int> order;
  order.reserve(static_cast<std::size_t>(n));
  while (!ready.empty()) {
    const int u = ready.front();
    ready.pop();
    order.push_back(u);
    for (int v : adj[static_cast<std::size_t>(u)]) {
      indeg[static_cast<std::size_t>(v)] -= 1;
      if (indeg[static_cast<std::size_t>(v)] == 0) {
        ready.push(v);
      }
    }
  }

  if (static_cast<int>(order.size()) != n) {
    // LEARNER: leftover nodes with indeg > 0 participate in a cycle.
    return Status::Error(
        ErrorCode::kModelLoad,
        "graph contains a cycle (topological sort could not order all " +
            std::to_string(n) + " nodes; ordered " +
            std::to_string(order.size()) + ")");
  }

  return order;
}

Status PrepareGraphStructure(Graph& graph) {
  // LEARNER: Always clear first so a failed re-prepare never leaves a stale
  // order that looks successful to Session.
  graph.topo_order.clear();
  EDUORT_RETURN_IF_ERROR(ValidateStructure(graph));
  EDUORT_ASSIGN_OR_RETURN(graph.topo_order, ComputeTopoOrder(graph));
  return Status::OK();
}

}  // namespace eduort
