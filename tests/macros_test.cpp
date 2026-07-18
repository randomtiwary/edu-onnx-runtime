// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Smoke-test the error macros so we do not regress the early-return
// pattern used throughout .cpp files (see include/eduort/macros.h).

#include "eduort/macros.h"
#include "eduort/status.h"

#include <string>

#include <gtest/gtest.h>

namespace eduort {
namespace {

Status FailWith(ErrorCode code, const char* msg) {
  return Status::Error(code, msg);
}

Status PropagateFailure() {
  EDUORT_RETURN_IF_ERROR(FailWith(ErrorCode::kInvalidArgument, "nope"));
  return Status::OK();
}

StatusOr<int> MakeValue(int v) { return v; }

StatusOr<int> MakeError() {
  return Status::Error(ErrorCode::kShapeMismatch, "bad");
}

StatusOr<int> AssignSuccess() {
  EDUORT_ASSIGN_OR_RETURN(const int x, MakeValue(7));
  return x + 1;
}

StatusOr<int> AssignFailure() {
  EDUORT_ASSIGN_OR_RETURN(const int x, MakeError());
  return x;
}

TEST(MacrosTest, ReturnIfErrorPropagates) {
  Status s = PropagateFailure();
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(s.message(), "nope");
}

TEST(MacrosTest, AssignOrReturnSuccess) {
  StatusOr<int> v = AssignSuccess();
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value(), 8);
}

TEST(MacrosTest, AssignOrReturnFailure) {
  StatusOr<int> v = AssignFailure();
  EXPECT_FALSE(v.ok());
  EXPECT_EQ(v.status().code(), ErrorCode::kShapeMismatch);
}

}  // namespace
}  // namespace eduort
