// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: GoogleTest uses TEST(Suite, Case) macros. Prefer checking both the
// boolean ok() path and the ErrorCode so regressions in either surface.

#include "eduort/status.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>

namespace eduort {
namespace {

TEST(StatusTest, DefaultIsOk) {
  Status s;
  EXPECT_TRUE(s.ok());
  EXPECT_EQ(s.code(), ErrorCode::kOk);
  EXPECT_EQ(s.ToString(), "OK");
}

TEST(StatusTest, ErrorCarriesCodeAndMessage) {
  Status s = Status::Error(ErrorCode::kInvalidArgument, "bad dim");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), ErrorCode::kInvalidArgument);
  EXPECT_EQ(s.message(), "bad dim");
  EXPECT_EQ(s.ToString(), "InvalidArgument: bad dim");
}

TEST(StatusTest, ErrorWithOkCodeIsCoerced) {
  Status s = Status::Error(ErrorCode::kOk, "nope");
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.code(), ErrorCode::kRuntime);
}

TEST(StatusOrTest, SuccessValue) {
  StatusOr<int> v = 42;
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value(), 42);
  EXPECT_EQ(*v, 42);
}

TEST(StatusOrTest, FailureStatus) {
  StatusOr<int> v = Status::Error(ErrorCode::kShapeMismatch, "rank");
  EXPECT_FALSE(v.ok());
  EXPECT_EQ(v.status().code(), ErrorCode::kShapeMismatch);
  EXPECT_EQ(v.status().message(), "rank");
}

TEST(StatusOrTest, OkStatusWithoutValueBecomesError) {
  StatusOr<int> v = Status::OK();
  EXPECT_FALSE(v.ok());
  EXPECT_EQ(v.status().code(), ErrorCode::kRuntime);
}

TEST(StatusOrTest, MoveOnlyType) {
  StatusOr<std::unique_ptr<int>> v = std::make_unique<int>(7);
  ASSERT_TRUE(v.ok());
  std::unique_ptr<int> p = std::move(v).value();
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(*p, 7);
}

TEST(StatusOrTest, ValueOrDieSuccess) {
  StatusOr<std::string> v = std::string("hi");
  EXPECT_EQ(std::move(v).ValueOrDie(), "hi");
}

TEST(StatusOrTest, ArrowOperator) {
  StatusOr<std::string> v = std::string("abc");
  EXPECT_EQ(v->size(), 3u);
}

TEST(ErrorCodeNameTest, CoversKnownCodes) {
  EXPECT_STREQ(ErrorCodeName(ErrorCode::kOk), "OK");
  EXPECT_STREQ(ErrorCodeName(ErrorCode::kCudaUnavailable), "CudaUnavailable");
}

}  // namespace
}  // namespace eduort
