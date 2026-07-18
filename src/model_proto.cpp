// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/model_proto.h"

#include "eduort/file_utils.h"
#include "eduort/macros.h"

#include <limits>
#include <utility>
#include <vector>

namespace eduort {

StatusOr<std::unique_ptr<onnx::ModelProto>> LoadModelProtoFromBytes(
    const void* data, std::size_t size) {
  // LEARNER: Validate inputs *before* constructing ModelProto so we do not
  // allocate a heavy protobuf object on paths that will only return an error.
  if (size > 0 && data == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "LoadModelProtoFromBytes: null data with size > 0");
  }
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "model byte buffer too large for ParseFromArray");
  }

  // LEARNER: ModelProto is the root message of every .onnx file.
  // ParseFromArray walks the protobuf wire encoding into C++ fields.
  auto model = std::make_unique<onnx::ModelProto>();
  const bool ok =
      model->ParseFromArray(data, static_cast<int>(size));  // protobuf API
  if (!ok) {
    return Status::Error(ErrorCode::kModelLoad,
                         "protobuf failed to parse bytes as onnx.ModelProto");
  }
  return model;
}

StatusOr<std::unique_ptr<onnx::ModelProto>> LoadModelProtoFromFile(
    const std::string& path, std::size_t max_bytes) {
  EDUORT_ASSIGN_OR_RETURN(std::vector<char> bytes,
                          ReadFileLimited(path, max_bytes));
  return LoadModelProtoFromBytes(bytes.data(), bytes.size());
}

}  // namespace eduort
