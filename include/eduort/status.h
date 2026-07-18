// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Recoverable errors in eduort use Status / StatusOr instead of
// C++ exceptions on the public API (design Key Decision K16).
//
// Why? Explicit checking teaches the control flow of a runtime: every fallible
// step returns a code you must look at. Exceptions can skip layers and are
// harder to reason about when learning how Session::Create / Run fail.
//
// Spec / design: docs/design.md § Public C++ API Sketch

#pragma once

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace eduort {

// LEARNER: ErrorCode is a small closed set of failure categories.
// Prefer a specific code when it helps the caller branch (e.g. kCudaUnavailable
// → fall back to CPU). Use kRuntime for unexpected compute failures.
enum class ErrorCode {
  kOk = 0,
  kInvalidArgument,
  kModelLoad,
  kUnsupportedOperator,
  kShapeMismatch,
  kRuntime,
  kCudaUnavailable,
};

// Human-readable name for logging / tests (never localized).
const char* ErrorCodeName(ErrorCode code) noexcept;

// ---------------------------------------------------------------------------
// Status — "did this operation succeed?"
// ---------------------------------------------------------------------------
// LEARNER: A Status is either OK or an error with a code + message.
// It does *not* carry a success value; use StatusOr<T> for that.
class Status {
 public:
  Status() noexcept;  // same as OK()

  static Status OK() noexcept;
  static Status Error(ErrorCode code, std::string message);

  bool ok() const noexcept { return code_ == ErrorCode::kOk; }
  ErrorCode code() const noexcept { return code_; }
  const std::string& message() const noexcept { return message_; }

  // Short form: "OK" or "InvalidArgument: dims must be non-negative"
  std::string ToString() const;

 private:
  Status(ErrorCode code, std::string message) noexcept;

  ErrorCode code_;
  std::string message_;
};

// ---------------------------------------------------------------------------
// StatusOr<T> — "success value OR error"
// ---------------------------------------------------------------------------
// LEARNER: Main return type for factories (Tensor::Create, Session::Create, …).
//
//   StatusOr<Tensor> t = Tensor::Create(...);
//   if (!t.ok()) { /* use t.status() */ return; }
//   Tensor tensor = std::move(t).value();
//
// Templates must live in the header so every .cpp that uses StatusOr<T> can
// instantiate the methods.
template <typename T>
class StatusOr {
 public:
  // Success path.
  StatusOr(T value)  // NOLINT(google-explicit-constructor)
      : status_(Status::OK()), value_(std::move(value)) {}

  // Failure path. Status::OK() without a value is a programmer bug → kRuntime.
  StatusOr(Status status)  // NOLINT(google-explicit-constructor)
      : status_(std::move(status)), value_(std::nullopt) {
    if (status_.ok()) {
      status_ = Status::Error(
          ErrorCode::kRuntime,
          "StatusOr constructed from Status::OK() without a value");
    }
  }

  bool ok() const noexcept { return status_.ok() && value_.has_value(); }

  const Status& status() const noexcept { return status_; }

  const T& value() const& {
    CheckOk();
    return *value_;
  }
  T& value() & {
    CheckOk();
    return *value_;
  }
  T&& value() && {
    CheckOk();
    return std::move(*value_);
  }

  const T& operator*() const& { return value(); }
  T& operator*() & { return value(); }
  T&& operator*() && { return std::move(*this).value(); }

  const T* operator->() const {
    CheckOk();
    return &(*value_);
  }
  T* operator->() {
    CheckOk();
    return &(*value_);
  }

  // LEARNER: For tests/demos only. Production code must check ok() and
  // propagate Status. On failure this prints and aborts.
  T ValueOrDie() && {
    if (!ok()) {
      std::fprintf(stderr, "eduort::StatusOr::ValueOrDie failed: %s\n",
                   status_.ToString().c_str());
      std::abort();
    }
    return std::move(*value_);
  }

 private:
  void CheckOk() const {
    if (!ok()) {
      std::fprintf(stderr,
                   "eduort::StatusOr::value() called on error StatusOr: %s\n",
                   status_.ToString().c_str());
      std::abort();
    }
  }

  Status status_;
  // LEARNER: std::optional so T need not be default-constructible (unique_ptr).
  std::optional<T> value_;
};

}  // namespace eduort
