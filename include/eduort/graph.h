// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: After loading protobuf, we convert ONNX into a teaching-friendly
// Graph IR. Kernels and Session never touch onnx::NodeProto directly.
//
// Spec: ONNX IR.md — Graphs, Nodes, Values; docs/design.md § Graph IR

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "eduort/tensor.h"

namespace eduort {

// LEARNER: ONNX attributes are typed key-value metadata on a node
// (e.g. Gemm's alpha, Softmax's axis). We store a small subset of kinds.
struct Attribute {
  enum class Kind {
    kFloat,
    kInt,
    kInts,
    kString,
    kTensor,
    // Unsupported kinds are rejected at load with a clear error (MVP).
  };

  std::string name;
  Kind kind = Kind::kFloat;
  // Payload fields: only the field matching `kind` is meaningful.
  float f = 0.f;                      // kind == kFloat
  int64_t i = 0;                      // kind == kInt
  std::vector<int64_t> ints;          // kind == kInts
  std::string s;                      // kind == kString
  std::optional<Tensor> tensor;       // kind == kTensor
};

struct Node {
  std::string name;      // optional diagnostic name from NodeProto
  std::string op_type;   // e.g. "MatMul", "Add"
  std::string domain;    // canonical "" (ai.onnx rewritten to "" at load)
  std::vector<std::string> inputs;   // value names; "" = optional absent
  std::vector<std::string> outputs;  // value names produced
  std::vector<Attribute> attributes;
};

struct Graph {
  std::string name;
  std::vector<Node> nodes;
  // topo_order filled in PR4; empty until then. Indices into nodes (size_t).
  std::vector<std::size_t> topo_order;

  // LEARNER: initializers are constant tensors (weights/biases) embedded in
  // the model. Keys are value names used as node inputs.
  std::unordered_map<std::string, Tensor> initializers;

  // Graph I/O names (from GraphProto.input / .output).
  std::vector<std::string> graph_inputs;
  std::vector<std::string> graph_outputs;

  // Optional shape hints from ValueInfo / inputs / outputs.
  std::unordered_map<std::string, TensorShape> value_shapes;

  // Resolved default-domain opset (ai.onnx / "") after load policy.
  int64_t opset_version = 0;

  // IR version from ModelProto (informational; warn if outside [3,9]).
  int64_t ir_version = 0;
};

// Opset acceptance window (design Key Decision K14: default domain + opset range).
constexpr int64_t kMinOpset = 11;
constexpr int64_t kMaxOpset = 17;

// IR versions we accept without hard-fail (warn outside).
constexpr int64_t kMinIrVersion = 3;
constexpr int64_t kMaxIrVersion = 9;

// Canonicalize ONNX domain spellings for the default opset.
// "ai.onnx" and "" both become "".
std::string CanonicalizeDomain(const std::string& domain);

}  // namespace eduort
