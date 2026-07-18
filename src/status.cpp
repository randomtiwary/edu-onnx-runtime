// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/status.h"

namespace eduort {

const char* ErrorCodeName(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kOk:
      return "OK";
    case ErrorCode::kInvalidArgument:
      return "InvalidArgument";
    case ErrorCode::kModelLoad:
      return "ModelLoad";
    case ErrorCode::kUnsupportedOperator:
      return "UnsupportedOperator";
    case ErrorCode::kShapeMismatch:
      return "ShapeMismatch";
    case ErrorCode::kRuntime:
      return "Runtime";
    case ErrorCode::kCudaUnavailable:
      return "CudaUnavailable";
  }
  return "UnknownErrorCode";
}

Status::Status() noexcept : code_(ErrorCode::kOk), message_() {}

Status::Status(ErrorCode code, std::string message) noexcept
    : code_(code), message_(std::move(message)) {}

Status Status::OK() noexcept { return Status(); }

Status Status::Error(ErrorCode code, std::string message) {
  // LEARNER: Constructing Error(kOk, ...) would be contradictory. Coerce to
  // kRuntime so the API cannot produce a lying Status.
  if (code == ErrorCode::kOk) {
    code = ErrorCode::kRuntime;
    if (message.empty()) {
      message = "Status::Error called with ErrorCode::kOk";
    }
  }
  return Status(code, std::move(message));
}

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }
  std::string out = ErrorCodeName(code_);
  if (!message_.empty()) {
    out += ": ";
    out += message_;
  }
  return out;
}

}  // namespace eduort
