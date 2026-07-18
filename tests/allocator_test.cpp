// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/allocator.h"

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

namespace eduort {
namespace {

TEST(CpuAllocatorTest, AllocateAndFree) {
  CpuAllocator alloc;
  EXPECT_EQ(alloc.device(), DeviceKind::kCPU);
  void* p = alloc.Allocate(128);
  ASSERT_NE(p, nullptr);
  std::memset(p, 0xAB, 128);
  alloc.Free(p);
}

TEST(CpuAllocatorTest, AllocateZeroReturnsNull) {
  CpuAllocator alloc;
  EXPECT_EQ(alloc.Allocate(0), nullptr);
}

TEST(DefaultCpuAllocatorTest, SingletonWorks) {
  IAllocator* a = DefaultCpuAllocator();
  IAllocator* b = DefaultCpuAllocator();
  EXPECT_EQ(a, b);
  void* p = a->Allocate(16);
  ASSERT_NE(p, nullptr);
  a->Free(p);
}

TEST(BufferPtrTest, OwningBufferFrees) {
  IAllocator* alloc = DefaultCpuAllocator();
  void* raw = alloc->Allocate(32);
  ASSERT_NE(raw, nullptr);
  {
    BufferPtr buf = MakeAllocatedBuffer(raw, alloc);
    EXPECT_EQ(buf.get(), raw);
  }
  // Freed when buf left scope — no leak under ASan; no double-free crash.
}

TEST(BufferPtrTest, NonOwningDoesNotFree) {
  // Stack buffer must not be freed by BufferPtr.
  alignas(float) unsigned char stack[16] = {};
  BufferPtr buf = MakeNonOwningBuffer(stack);
  EXPECT_EQ(buf.get(), static_cast<void*>(stack));
  buf.reset();
  // If we wrongly used delete[], this would crash. Still here → no-op deleter.
  stack[0] = 1;
  EXPECT_EQ(stack[0], 1);
}

TEST(DeviceKindNameTest, CpuAndCuda) {
  EXPECT_STREQ(DeviceKindName(DeviceKind::kCPU), "CPU");
  EXPECT_STREQ(DeviceKindName(DeviceKind::kCUDA), "CUDA");
}

}  // namespace
}  // namespace eduort
