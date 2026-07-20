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
//
// Node indices use std::size_t to match vector::size() / operator[] without
// repeated casts.

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
StatusOr<std::unordered_map<std::string, std::size_t>> BuildProducerMap(
    const Graph& graph) {
  std::unordered_map<std::string, std::size_t> producer;
  for (std::size_t ni = 0; ni < graph.nodes.size(); ++ni) {
    for (const std::string& out : graph.nodes[ni].outputs) {
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

StatusOr<std::vector<std::size_t>> ComputeTopoOrder(const Graph& graph) {
  const std::size_t n = graph.nodes.size();
  if (n == 0) {
    return std::vector<std::size_t>{};
  }

  EDUORT_ASSIGN_OR_RETURN(const auto producer, BuildProducerMap(graph));

  // adjacency[u] = list of nodes that depend on u (consumers).
  std::vector<std::vector<std::size_t>> adj(n);
  std::vector<std::size_t> indeg(n, 0);

  for (std::size_t ni = 0; ni < n; ++ni) {
    const Node& node = graph.nodes[ni];
    // Unique predecessors so multi-input from same producer counts once.
    std::unordered_set<std::size_t> seen_pred;
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
      const std::size_t pred = it->second;
      if (pred == ni) {
        return Status::Error(ErrorCode::kModelLoad,
                             "node '" + node.name + "' consumes its own output '" +
                                 in + "' (self-loop)");
      }
      if (!seen_pred.insert(pred).second) {
        continue;  // already counted edge from this predecessor
      }
      adj[pred].push_back(ni);
      indeg[ni] += 1;
    }
  }

  // LEARNER: std::queue for the "ready set" — any order among ready nodes is a
  // valid topo order; we use FIFO for determinism.
  std::queue<std::size_t> ready;
  for (std::size_t i = 0; i < n; ++i) {
    if (indeg[i] == 0) {
      ready.push(i);
    }
  }

  std::vector<std::size_t> order;
  order.reserve(n);
  while (!ready.empty()) {
    const std::size_t u = ready.front();
    ready.pop();
    order.push_back(u);
    for (const std::size_t v : adj[u]) {
      indeg[v] -= 1;
      if (indeg[v] == 0) {
        ready.push(v);
      }
    }
  }

  if (order.size() != n) {
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
