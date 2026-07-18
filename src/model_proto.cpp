// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/model_proto.h"

#include "eduort/macros.h"

#include <fstream>
#include <limits>
#include <utility>
#include <vector>

namespace eduort {
namespace {

StatusOr<std::vector<char>> ReadFileLimited(const std::string& path,
                                            std::size_t max_bytes) {
  if (path.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument, "model path is empty");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::Error(ErrorCode::kModelLoad, "failed to open model file: " + path);
  }

  in.seekg(0, std::ios::end);
  const std::streamoff end = in.tellg();
  if (end < 0) {
    return Status::Error(ErrorCode::kModelLoad, "failed to stat model file: " + path);
  }
  const auto file_size = static_cast<std::uint64_t>(end);
  if (file_size > max_bytes) {
    return Status::Error(
        ErrorCode::kInvalidArgument,
        "model file exceeds max_bytes (" + std::to_string(file_size) + " > " +
            std::to_string(max_bytes) + "): " + path);
  }

  in.seekg(0, std::ios::beg);
  std::vector<char> buf(static_cast<std::size_t>(file_size));
  if (file_size > 0) {
    in.read(buf.data(), static_cast<std::streamsize>(file_size));
    if (!in) {
      return Status::Error(ErrorCode::kModelLoad,
                           "failed to read model file: " + path);
    }
  }
  return buf;
}

}  // namespace

StatusOr<std::unique_ptr<onnx::ModelProto>> LoadModelProtoFromBytes(
    const void* data, std::size_t size) {
  // LEARNER: ModelProto is the root message of every .onnx file.
  // ParseFromArray walks the protobuf wire encoding into C++ fields.
  auto model = std::make_unique<onnx::ModelProto>();
  if (size > 0 && data == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "LoadModelProtoFromBytes: null data with size > 0");
  }
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "model byte buffer too large for ParseFromArray");
  }
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
