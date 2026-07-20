// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Structural validation checks *name wiring* only:
//   - Every non-empty node input is defined (graph input, initializer, or
//     some node's output).
//   - Each value name has at most one definition (SSA-style), including no
//     node output that redefines a seed (graph input / initializer).
//   - Unknown op_types are allowed here — kernels arrive later (PR5+).
// Cycle detection is *not* done here; see Kahn's algorithm in topo_sort.cpp.

#include "eduort/validate.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace eduort {

Status ValidateStructure(const Graph& graph) {
  // LEARNER: Seeds (graph inputs + initializers) are defined before any node
  // runs. Node outputs must not redefine a seed, and must not collide with
  // each other (single assignment).
  std::unordered_set<std::string> seeds;
  std::unordered_set<std::string> defined;
  seeds.reserve(graph.graph_inputs.size() + graph.initializers.size());
  defined.reserve(graph.graph_inputs.size() + graph.initializers.size() +
                  graph.nodes.size() * 2u);

  for (const std::string& name : graph.graph_inputs) {
    if (name.empty()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "graph has an empty input name");
    }
    seeds.insert(name);
    defined.insert(name);
  }
  for (const auto& kv : graph.initializers) {
    if (kv.first.empty()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "initializer with empty name");
    }
    seeds.insert(kv.first);
    defined.insert(kv.first);
  }

  // LEARNER: ONNX values are single-assignment — at most one producer per name.
  std::unordered_map<std::string, int> producer;  // value → node index

  for (int ni = 0; ni < static_cast<int>(graph.nodes.size()); ++ni) {
    const Node& node = graph.nodes[static_cast<std::size_t>(ni)];
    if (node.op_type.empty()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "node at index " + std::to_string(ni) +
                               " has empty op_type");
    }

    for (const std::string& out : node.outputs) {
      // LEARNER (MVP): empty output names are rejected. ONNX allows "" for
      // omitted optional outputs (like empty inputs), but no MVP op needs that
      // yet. Documented in validate.h / docs/04-graph-and-topo.md.
      if (out.empty()) {
        return Status::Error(ErrorCode::kModelLoad,
                             "node '" + node.name + "' (" + node.op_type +
                                 ") has an empty output name");
      }
      // Reject redefinition of a seed (graph input and/or initializer).
      if (seeds.count(out) != 0) {
        return Status::Error(
            ErrorCode::kModelLoad,
            "node output '" + out +
                "' redefines a graph input or initializer (single-assignment)");
      }
      auto [it, inserted] = producer.emplace(out, ni);
      if (!inserted) {
        return Status::Error(
            ErrorCode::kModelLoad,
            "value '" + out + "' is produced by multiple nodes (indices " +
                std::to_string(it->second) + " and " + std::to_string(ni) +
                "); ONNX graphs must be single-assignment");
      }
      defined.insert(out);
    }
  }

  // Every non-empty node input must be defined somewhere.
  for (int ni = 0; ni < static_cast<int>(graph.nodes.size()); ++ni) {
    const Node& node = graph.nodes[static_cast<std::size_t>(ni)];
    for (const std::string& in : node.inputs) {
      // LEARNER: empty string means "optional input omitted" (e.g. Gemm C).
      if (in.empty()) {
        continue;
      }
      if (defined.count(in) == 0) {
        return Status::Error(
            ErrorCode::kModelLoad,
            "node '" + node.name + "' (" + node.op_type +
                ") references undefined value '" + in +
                "' (not a graph input, initializer, or node output)");
      }
    }
  }

  // Graph outputs must be defined (else Run could never materialize them).
  for (const std::string& out : graph.graph_outputs) {
    if (out.empty()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "graph has an empty output name");
    }
    if (defined.count(out) == 0) {
      return Status::Error(ErrorCode::kModelLoad,
                           "graph output '" + out +
                               "' is not produced by any node, input, or "
                               "initializer");
    }
  }

  return Status::OK();
}

}  // namespace eduort
