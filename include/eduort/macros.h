// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Fallible APIs in eduort return Status / StatusOr. Checking every
// result with boilerplate:
//
//   StatusOr<T> x = Foo();
//   if (!x.ok()) {
//     return x.status();
//   }
//   T v = std::move(x).value();
//
// gets noisy fast (and easy to forget). These macros implement the standard
// "early return on error" pattern used by many C++ codebases (similar spirit to
// Abseil's RETURN_IF_ERROR / ASSIGN_OR_RETURN, and ONNX Runtime helpers).
//
// Use them in .cpp files that return Status or StatusOr<U>. Prefer them over
// hand-rolled if (!ok()) blocks so the project stays consistent.
//
// Design: docs/design.md § Public C++ API Sketch (K16)

#pragma once

#include "eduort/status.h"

#include <utility>

// Internal token-paste helpers (not for call sites).
#define EDUORT_MACRO_CONCAT_IMPL(a, b) a##b
#define EDUORT_MACRO_CONCAT(a, b) EDUORT_MACRO_CONCAT_IMPL(a, b)

// ---------------------------------------------------------------------------
// EDUORT_RETURN_IF_ERROR(status_expr)
// ---------------------------------------------------------------------------
// LEARNER: Evaluate `status_expr` (a Status, or anything convertible to Status).
// If it is not OK, return it from the current function. Works in functions that
// return Status *or* StatusOr<T> (Status converts to a failed StatusOr).
//
// Example:
//   EDUORT_RETURN_IF_ERROR(ValidateShape(shape));
//
//   StatusOr<int64_t> n = shape.NumElementsChecked();
//   EDUORT_RETURN_IF_ERROR(n.status());
//
#define EDUORT_RETURN_IF_ERROR(status_expr)                      \
  do {                                                           \
    const ::eduort::Status _eduort_status = (status_expr);       \
    if (!_eduort_status.ok()) {                                  \
      return _eduort_status;                                     \
    }                                                            \
  } while (0)

// ---------------------------------------------------------------------------
// EDUORT_ASSIGN_OR_RETURN(lhs, status_or_expr)
// ---------------------------------------------------------------------------
// LEARNER: Evaluate a StatusOr expression. On failure, return its Status.
// On success, move the value into `lhs`.
//
// `lhs` may be an existing variable *or* a declaration:
//   EDUORT_ASSIGN_OR_RETURN(const std::size_t nbytes, ComputeNBytes(dt, shape));
//   EDUORT_ASSIGN_OR_RETURN(n, shape.NumElementsChecked());  // n already declared
//
// Implementation note: not wrapped in do/while so that `lhs` can introduce a
// new variable in the surrounding scope (same idea as Abseil ASSIGN_OR_RETURN).
//
#define EDUORT_ASSIGN_OR_RETURN(lhs, status_or_expr)                         \
  EDUORT_ASSIGN_OR_RETURN_IMPL(                                              \
      EDUORT_MACRO_CONCAT(_eduort_status_or_, __LINE__), lhs, status_or_expr)

#define EDUORT_ASSIGN_OR_RETURN_IMPL(status_or_name, lhs, status_or_expr)    \
  auto status_or_name = (status_or_expr);                                    \
  if (!status_or_name.ok()) {                                                \
    return status_or_name.status();                                          \
  }                                                                          \
  lhs = std::move(status_or_name).value()

