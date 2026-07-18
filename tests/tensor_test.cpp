// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: These tests pin down the ownership and shape contracts that Session
// will rely on later (feeds as FromHostBlob, owns_data(), nbytes, etc.).

#include "eduort/tensor.h"

#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace eduort {
namespace {

TEST(TensorShapeTest, RankAndElements) {
  TensorShape s({2, 3, 4});
  EXPECT_EQ(s.rank(), 3);
  EXPECT_EQ(s.NumElements(), 24u);
  EXPECT_EQ(s.ToString(), "[2,3,4]");
}

TEST(TensorShapeTest, ScalarRank0) {
  // LEARNER: ONNX scalar = empty shape, one element.
  TensorShape s;  // rank 0
  EXPECT_EQ(s.rank(), 0);
  EXPECT_EQ(s.NumElements(), 1u);
}

TEST(TensorShapeTest, ZeroDim) {
  TensorShape s({2, 0, 3});
  EXPECT_EQ(s.NumElements(), 0u);
}

TEST(TensorShapeTest, FromSignedDimsRejectsNegative) {
  // LEARNER: uint64_t dims cannot be negative; the ONNX bridge rejects d < 0.
  StatusOr<TensorShape> s = TensorShape::FromSignedDims({2, -1});
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.status().code(), ErrorCode::kInvalidArgument);
}

TEST(TensorShapeTest, FromSignedDimsAcceptsNonNegative) {
  StatusOr<TensorShape> s = TensorShape::FromSignedDims({2, 3});
  ASSERT_TRUE(s.ok()) << s.status().ToString();
  EXPECT_EQ(s->NumElements(), 6u);
}

TEST(TensorShapeTest, OverflowFailsChecked) {
  // Two huge dims whose product does not fit in uint64.
  TensorShape s({std::numeric_limits<uint64_t>::max(), 2});
  StatusOr<uint64_t> n = s.NumElementsChecked();
  EXPECT_FALSE(n.ok());
  EXPECT_EQ(n.status().code(), ErrorCode::kInvalidArgument);
}

TEST(TensorCreateTest, FloatMatrix) {
  StatusOr<Tensor> t = Tensor::Create(DataType::kFloat32, TensorShape({2, 3}));
  ASSERT_TRUE(t.ok()) << t.status().ToString();
  EXPECT_EQ(t->dtype(), DataType::kFloat32);
  EXPECT_EQ(t->shape().NumElements(), 6u);
  EXPECT_EQ(t->nbytes(), 6 * sizeof(float));
  EXPECT_EQ(t->device(), DeviceKind::kCPU);
  EXPECT_TRUE(t->owns_data());
  ASSERT_NE(t->mutable_data_f32(), nullptr);

  float* p = t->mutable_data_f32();
  for (int i = 0; i < 6; ++i) {
    p[i] = static_cast<float>(i);
  }
  EXPECT_FLOAT_EQ(t->data_f32()[5], 5.0f);
}

TEST(TensorCreateTest, Int64Vector) {
  StatusOr<Tensor> t = Tensor::Create(DataType::kInt64, TensorShape({4}));
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->nbytes(), 4 * sizeof(int64_t));
  t->mutable_data_i64()[0] = 42;
  EXPECT_EQ(t->data_i64()[0], 42);
}

TEST(TensorCreateTest, EmptyTensor) {
  StatusOr<Tensor> t = Tensor::Create(DataType::kFloat32, TensorShape({0}));
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->nbytes(), 0u);
  EXPECT_EQ(t->data(), nullptr);
}

TEST(TensorCreateTest, RejectsCudaDeviceInPr2) {
  StatusOr<Tensor> t =
      Tensor::Create(DataType::kFloat32, TensorShape({1}), DeviceKind::kCUDA);
  EXPECT_FALSE(t.ok());
  EXPECT_EQ(t.status().code(), ErrorCode::kInvalidArgument);
}

TEST(TensorCreateTest, RejectsOverflowingShape) {
  StatusOr<Tensor> t = Tensor::Create(
      DataType::kFloat32,
      TensorShape({std::numeric_limits<uint64_t>::max(), 2}));
  EXPECT_FALSE(t.ok());
  EXPECT_EQ(t.status().code(), ErrorCode::kInvalidArgument);
}

TEST(TensorCreateTest, RejectsNegativeViaFromSignedDims) {
  StatusOr<TensorShape> shape = TensorShape::FromSignedDims({1, -5});
  ASSERT_FALSE(shape.ok());
  EXPECT_EQ(shape.status().code(), ErrorCode::kInvalidArgument);
}

TEST(TensorFromHostBlobTest, WrapsWithoutOwning) {
  std::vector<float> host = {1.f, 2.f, 3.f, 4.f};
  StatusOr<Tensor> t = Tensor::FromHostBlob(
      DataType::kFloat32, TensorShape({2, 2}), host.data(),
      host.size() * sizeof(float));
  ASSERT_TRUE(t.ok()) << t.status().ToString();
  EXPECT_FALSE(t->owns_data());
  EXPECT_EQ(t->data_f32(), host.data());
  EXPECT_FLOAT_EQ(t->data_f32()[3], 4.f);

  // Mutating host is visible through the tensor (alias).
  host[0] = 9.f;
  EXPECT_FLOAT_EQ(t->data_f32()[0], 9.f);
}

TEST(TensorFromHostBlobTest, RejectsByteMismatch) {
  float host[2] = {0.f, 1.f};
  StatusOr<Tensor> t = Tensor::FromHostBlob(
      DataType::kFloat32, TensorShape({2, 2}), host, sizeof(host));
  EXPECT_FALSE(t.ok());
  EXPECT_EQ(t.status().code(), ErrorCode::kInvalidArgument);
}

TEST(TensorFromHostBlobTest, RejectsNullNonEmpty) {
  StatusOr<Tensor> t =
      Tensor::FromHostBlob(DataType::kFloat32, TensorShape({1}), nullptr, 4);
  EXPECT_FALSE(t.ok());
}

TEST(TensorCopyShareTest, SharedPtrAlias) {
  // LEARNER: Copying a Tensor shares the buffer (shared_ptr). Session will use
  // this when shallow-copying the value map at the start of Run.
  StatusOr<Tensor> a = Tensor::Create(DataType::kFloat32, TensorShape({2}));
  ASSERT_TRUE(a.ok());
  a->mutable_data_f32()[0] = 3.5f;
  Tensor b = a.value();  // copy
  EXPECT_EQ(b.data_f32(), a->data_f32());
  EXPECT_FLOAT_EQ(b.data_f32()[0], 3.5f);
}

// Death tests: typed accessor with wrong dtype aborts (programmer error).
TEST(TensorDeathTest, TypedAccessorWrongDtypeAborts) {
  StatusOr<Tensor> t = Tensor::Create(DataType::kInt64, TensorShape({1}));
  ASSERT_TRUE(t.ok());
  EXPECT_DEATH(
      { (void)t->mutable_data_f32(); },
      "dtype mismatch");
}

}  // namespace
}  // namespace eduort
