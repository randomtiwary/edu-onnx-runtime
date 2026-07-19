// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: onnx_loader maps protobuf ModelProto → eduort::Graph.
// This is the "compiler front-end" of the runtime: schema-level data becomes
// the structures Session will execute later.
//
// Spec: ONNX IR.md; design Key Decisions (full onnx.proto, opset/domain policy)

#pragma once

#include <cstddef>
#include <string>

#include "eduort/graph.h"
#include "eduort/model_proto.h"
#include "eduort/status.h"

#include "onnx.pb.h"

namespace eduort {

// file → ModelProto → Graph (opset / domain / ir_version policy applied).
StatusOr<Graph> LoadGraphFromFile(const std::string& path,
                                  std::size_t max_bytes = kDefaultMaxModelBytes);

// Map an already-parsed ModelProto into Graph IR.
StatusOr<Graph> LoadGraphFromModelProto(const onnx::ModelProto& model);

// Convert a TensorProto (initializer / Constant attr) into a host Tensor.
// Supported ONNX dtypes/payloads: see Tensor / DataType in tensor.h.
StatusOr<Tensor> TensorFromTensorProto(const onnx::TensorProto& tp);

}  // namespace eduort
