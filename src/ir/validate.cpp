// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/validate.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace eduort {

Status ValidateStructure(const Graph& graph) {
  // LEARNER: "Defined" values at load time are graph inputs + initializers.
  // Node outputs become defined when we consider the whole graph (any order);
  // cycle detection is separate in topo sort.
  std::unordered_set<std::string> defined;
  defined.reserve(graph.graph_inputs.size() + graph.initializers.size() +
                  graph.nodes.size() * 2u);

  for (const std::string& name : graph.graph_inputs) {
    if (name.empty()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "graph has an empty input name");
    }
    defined.insert(name);
  }
  for (const auto& kv : graph.initializers) {
    if (kv.first.empty()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "initializer with empty name");
    }
    defined.insert(kv.first);
  }

  // LEARNER: ONNX values are single-assignment — at most one producer node
  // per name in a standard inference graph.
  std::unordered_map<std::string, int> producer;  // value → node index

  for (int ni = 0; ni < static_cast<int>(graph.nodes.size()); ++ni) {
    const Node& node = graph.nodes[static_cast<std::size_t>(ni)];
    if (node.op_type.empty()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "node at index " + std::to_string(ni) +
                               " has empty op_type");
    }

    for (const std::string& out : node.outputs) {
      if (out.empty()) {
        return Status::Error(ErrorCode::kModelLoad,
                             "node '" + node.name + "' (" + node.op_type +
                                 ") has an empty output name");
      }
      auto [it, inserted] = producer.emplace(out, ni);
      if (!inserted) {
        return Status::Error(
            ErrorCode::kModelLoad,
            "value '" + out + "' is produced by multiple nodes (indices " +
                std::to_string(it->second) + " and " + std::to_string(ni) +
                "); ONNX graphs must be single-assignment");
      }
      // Also cannot collide with a graph input name that is not the same
      // pattern as initializer-as-input (node must not redefine graph input).
      // ONNX allows initializer+input same name, but not node output = input.
      for (const std::string& gin : graph.graph_inputs) {
        if (gin == out && graph.initializers.count(out) == 0) {
          // Redefining a pure feed name is illegal.
          return Status::Error(ErrorCode::kModelLoad,
                               "node output '" + out +
                                   "' collides with graph input name");
        }
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
