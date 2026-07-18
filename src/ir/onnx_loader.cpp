// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Field-by-field mapping from ONNX protobuf messages into Graph IR.
// Keep this file as the single place that "understands" onnx::NodeProto, etc.

#include "eduort/onnx_loader.h"

#include "eduort/macros.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace eduort {
namespace {

// ONNX TensorProto.DataType enum values we care about (from onnx.proto).
constexpr int kOnnxFloat = 1;   // FLOAT
constexpr int kOnnxInt64 = 7;   // INT64

StatusOr<DataType> MapElemType(int onnx_dtype) {
  if (onnx_dtype == kOnnxFloat) {
    return DataType::kFloat32;
  }
  if (onnx_dtype == kOnnxInt64) {
    return DataType::kInt64;
  }
  return Status::Error(ErrorCode::kModelLoad,
                       "unsupported TensorProto data_type " +
                           std::to_string(onnx_dtype) +
                           " (MVP: FLOAT=1, INT64=7)");
}

// ValueInfoProto.type.tensor_type → TensorShape (static dims only).
StatusOr<TensorShape> ShapeFromValueInfo(const onnx::ValueInfoProto& vi) {
  if (!vi.has_type() || !vi.type().has_tensor_type()) {
    // Unknown / missing shape is OK — leave caller to skip.
    return TensorShape{};  // rank-0 scalar as empty marker? Better optional.
  }
  const auto& tt = vi.type().tensor_type();
  if (!tt.has_shape()) {
    return TensorShape{};
  }
  std::vector<int64_t> dims;
  for (const auto& d : tt.shape().dim()) {
    // LEARNER: dim_param (symbolic) is unsupported in MVP fixtures.
    if (d.has_dim_param() && !d.dim_param().empty() && !d.has_dim_value()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "symbolic dim_param not supported in MVP: " +
                               d.dim_param() + " (value " + vi.name() + ")");
    }
    if (d.has_dim_value()) {
      dims.push_back(d.dim_value());
    } else {
      return Status::Error(ErrorCode::kModelLoad,
                           "missing dim_value for value " + vi.name());
    }
  }
  return TensorShape::FromSignedDims(std::move(dims));
}

StatusOr<int64_t> ResolveOpset(const onnx::ModelProto& model) {
  // LEARNER: opset_import lists (domain, version) pairs. We only accept the
  // default ONNX domain ("" / "ai.onnx") in range [11, 17].
  int64_t found = -1;
  bool saw_default = false;
  for (const auto& id : model.opset_import()) {
    const std::string domain = CanonicalizeDomain(id.domain());
    if (domain != "") {
      // Non-default domain (e.g. ai.onnx.ml) — reject.
      return Status::Error(
          ErrorCode::kModelLoad,
          "unsupported opset domain '" + id.domain() +
              "' (MVP only supports default ONNX domain \"\" / ai.onnx)");
    }
    if (saw_default && found != id.version()) {
      // Conflicting default-domain versions.
      return Status::Error(ErrorCode::kModelLoad,
                           "conflicting default-domain opset versions in model");
    }
    saw_default = true;
    found = id.version();
  }
  if (!saw_default) {
    // ONNX says models should import the default opset; be strict for learners.
    return Status::Error(ErrorCode::kModelLoad,
                         "model missing opset_import for default ONNX domain");
  }
  if (found < kMinOpset || found > kMaxOpset) {
    return Status::Error(ErrorCode::kModelLoad,
                         "opset version " + std::to_string(found) +
                             " outside supported range [" +
                             std::to_string(kMinOpset) + ", " +
                             std::to_string(kMaxOpset) + "]");
  }
  return found;
}

StatusOr<Attribute> MapAttribute(const onnx::AttributeProto& ap) {
  Attribute a;
  a.name = ap.name();

  // LEARNER: AttributeProto.type discriminates which field is populated.
  // We implement the kinds needed for MVP ops; others fail clearly.
  using A = onnx::AttributeProto;
  switch (ap.type()) {
    case A::FLOAT:
      a.kind = Attribute::Kind::kFloat;
      a.f = ap.f();
      return a;
    case A::INT:
      a.kind = Attribute::Kind::kInt;
      a.i = ap.i();
      return a;
    case A::INTS:
      a.kind = Attribute::Kind::kInts;
      a.ints.assign(ap.ints().begin(), ap.ints().end());
      return a;
    case A::STRING:
      a.kind = Attribute::Kind::kString;
      a.s = ap.s();
      return a;
    case A::TENSOR: {
      a.kind = Attribute::Kind::kTensor;
      EDUORT_ASSIGN_OR_RETURN(Tensor tv, TensorFromTensorProto(ap.t()));
      a.tensor = std::move(tv);
      return a;
    }
    case A::UNDEFINED:
      // Some producers omit type and set a single field — probe common ones.
      if (ap.has_t()) {
        a.kind = Attribute::Kind::kTensor;
        EDUORT_ASSIGN_OR_RETURN(Tensor tv, TensorFromTensorProto(ap.t()));
        a.tensor = std::move(tv);
        return a;
      }
      if (ap.has_f()) {
        a.kind = Attribute::Kind::kFloat;
        a.f = ap.f();
        return a;
      }
      if (ap.has_i()) {
        a.kind = Attribute::Kind::kInt;
        a.i = ap.i();
        return a;
      }
      if (!ap.ints().empty()) {
        a.kind = Attribute::Kind::kInts;
        a.ints.assign(ap.ints().begin(), ap.ints().end());
        return a;
      }
      if (ap.has_s()) {
        a.kind = Attribute::Kind::kString;
        a.s = ap.s();
        return a;
      }
      return Status::Error(ErrorCode::kModelLoad,
                           "attribute '" + ap.name() +
                               "' has UNDEFINED type and no data fields");
    default:
      return Status::Error(ErrorCode::kModelLoad,
                           "unsupported attribute type for '" + ap.name() +
                               "' (type=" + std::to_string(ap.type()) + ")");
  }
}

StatusOr<Node> MapNode(const onnx::NodeProto& np) {
  Node n;
  n.name = np.name();
  n.op_type = np.op_type();
  if (n.op_type.empty()) {
    return Status::Error(ErrorCode::kModelLoad, "node missing op_type");
  }
  n.domain = CanonicalizeDomain(np.domain());
  // Only default domain nodes allowed (same policy as opset_import).
  if (n.domain != "") {
    return Status::Error(ErrorCode::kModelLoad,
                         "node '" + n.name + "' op " + n.op_type +
                             " has unsupported domain '" + np.domain() + "'");
  }
  n.inputs.assign(np.input().begin(), np.input().end());
  n.outputs.assign(np.output().begin(), np.output().end());
  n.attributes.reserve(static_cast<std::size_t>(np.attribute_size()));
  for (const auto& ap : np.attribute()) {
    EDUORT_ASSIGN_OR_RETURN(Attribute attr, MapAttribute(ap));
    n.attributes.push_back(std::move(attr));
  }
  return n;
}

}  // namespace

std::string CanonicalizeDomain(const std::string& domain) {
  // LEARNER: ONNX default domain is historically either "" or "ai.onnx".
  // We store one spelling everywhere so registry lookup is simple.
  if (domain.empty() || domain == "ai.onnx") {
    return "";
  }
  return domain;
}

StatusOr<Tensor> TensorFromTensorProto(const onnx::TensorProto& tp) {
  EDUORT_ASSIGN_OR_RETURN(const DataType dt, MapElemType(tp.data_type()));
  std::vector<int64_t> signed_dims(tp.dims().begin(), tp.dims().end());
  EDUORT_ASSIGN_OR_RETURN(TensorShape shape,
                          TensorShape::FromSignedDims(std::move(signed_dims)));
  EDUORT_ASSIGN_OR_RETURN(const uint64_t ne, shape.NumElementsChecked());
  const std::size_t elem = SizeOfDataType(dt);
  const std::size_t nbytes = static_cast<std::size_t>(ne) * elem;

  EDUORT_ASSIGN_OR_RETURN(Tensor t, Tensor::Create(dt, shape, DeviceKind::kCPU));

  if (nbytes == 0) {
    return t;
  }

  // Prefer raw_data when present (common for weight dumps).
  if (!tp.raw_data().empty()) {
    if (tp.raw_data().size() != nbytes) {
      return Status::Error(ErrorCode::kModelLoad,
                           "TensorProto raw_data size mismatch for '" +
                               tp.name() + "'");
    }
    std::memcpy(t.mutable_data(), tp.raw_data().data(), nbytes);
    return t;
  }

  if (dt == DataType::kFloat32) {
    if (static_cast<std::size_t>(tp.float_data_size()) != ne) {
      return Status::Error(ErrorCode::kModelLoad,
                           "TensorProto float_data size mismatch for '" +
                               tp.name() + "'");
    }
    float* out = t.mutable_data_f32();
    for (int i = 0; i < tp.float_data_size(); ++i) {
      out[i] = tp.float_data(i);
    }
    return t;
  }

  if (dt == DataType::kInt64) {
    if (static_cast<std::size_t>(tp.int64_data_size()) != ne) {
      return Status::Error(ErrorCode::kModelLoad,
                           "TensorProto int64_data size mismatch for '" +
                               tp.name() + "'");
    }
    int64_t* out = t.mutable_data_i64();
    for (int i = 0; i < tp.int64_data_size(); ++i) {
      out[i] = tp.int64_data(i);
    }
    return t;
  }

  return Status::Error(ErrorCode::kModelLoad,
                       "TensorProto has no usable data payload for '" +
                           tp.name() + "'");
}

StatusOr<Graph> LoadGraphFromModelProto(const onnx::ModelProto& model) {
  if (!model.has_graph()) {
    return Status::Error(ErrorCode::kModelLoad, "ModelProto has no graph");
  }

  Graph g;
  g.ir_version = model.ir_version();
  // LEARNER: IR version outside [3,9] is a warning-only policy in the design.
  // We do not fail solely on ir_version; log via stderr for now (Env later).
  if (g.ir_version < kMinIrVersion || g.ir_version > kMaxIrVersion) {
    std::fprintf(stderr,
                 "eduort warning: ir_version=%lld outside preferred [%lld,%lld]\n",
                 static_cast<long long>(g.ir_version),
                 static_cast<long long>(kMinIrVersion),
                 static_cast<long long>(kMaxIrVersion));
  }

  EDUORT_ASSIGN_OR_RETURN(g.opset_version, ResolveOpset(model));

  const onnx::GraphProto& gp = model.graph();
  g.name = gp.name();

  // --- Initializers (weights) ----------------------------------------------
  for (const auto& tp : gp.initializer()) {
    if (tp.name().empty()) {
      return Status::Error(ErrorCode::kModelLoad,
                           "initializer with empty name");
    }
    EDUORT_ASSIGN_OR_RETURN(Tensor tensor, TensorFromTensorProto(tp));
    if (g.initializers.count(tp.name()) != 0) {
      return Status::Error(ErrorCode::kModelLoad,
                           "duplicate initializer name: " + tp.name());
    }
    g.initializers.emplace(tp.name(), std::move(tensor));
  }

  // --- Graph inputs / outputs + shape table --------------------------------
  for (const auto& vi : gp.input()) {
    g.graph_inputs.push_back(vi.name());
    StatusOr<TensorShape> sh = ShapeFromValueInfo(vi);
    if (!sh.ok()) {
      return sh.status();
    }
    // Only record non-empty rank shapes or rank-0 from explicit empty shape.
    // Rank-0 from missing type: skip storing to avoid fake scalars.
    if (vi.has_type() && vi.type().has_tensor_type() &&
        vi.type().tensor_type().has_shape()) {
      g.value_shapes[vi.name()] = std::move(sh).value();
    }
  }
  for (const auto& vi : gp.output()) {
    g.graph_outputs.push_back(vi.name());
    StatusOr<TensorShape> sh = ShapeFromValueInfo(vi);
    if (!sh.ok()) {
      return sh.status();
    }
    if (vi.has_type() && vi.type().has_tensor_type() &&
        vi.type().tensor_type().has_shape()) {
      g.value_shapes[vi.name()] = std::move(sh).value();
    }
  }
  for (const auto& vi : gp.value_info()) {
    StatusOr<TensorShape> sh = ShapeFromValueInfo(vi);
    if (!sh.ok()) {
      return sh.status();
    }
    if (vi.has_type() && vi.type().has_tensor_type() &&
        vi.type().tensor_type().has_shape()) {
      g.value_shapes[vi.name()] = std::move(sh).value();
    }
  }

  // --- Nodes ----------------------------------------------------------------
  g.nodes.reserve(static_cast<std::size_t>(gp.node_size()));
  for (const auto& np : gp.node()) {
    EDUORT_ASSIGN_OR_RETURN(Node node, MapNode(np));
    g.nodes.push_back(std::move(node));
  }

  return g;
}

StatusOr<Graph> LoadGraphFromFile(const std::string& path,
                                  std::size_t max_bytes) {
  EDUORT_ASSIGN_OR_RETURN(std::unique_ptr<onnx::ModelProto> model,
                          LoadModelProtoFromFile(path, max_bytes));
  return LoadGraphFromModelProto(*model);
}

}  // namespace eduort
