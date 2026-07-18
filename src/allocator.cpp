// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/allocator.h"

#include <new>

namespace eduort {

const char* DeviceKindName(DeviceKind device) noexcept {
  switch (device) {
    case DeviceKind::kCPU:
      return "CPU";
    case DeviceKind::kCUDA:
      return "CUDA";
  }
  return "UnknownDevice";
}

void* CpuAllocator::Allocate(std::size_t nbytes) {
  // LEARNER: operator new[] throws std::bad_alloc on failure by default.
  // We use the nothrow overload so the rest of the runtime can speak Status
  // instead of mixing exception styles (K16).
  if (nbytes == 0) {
    // Return a non-null sentinel? Simpler: nullptr for empty allocations.
    return nullptr;
  }
  return ::operator new[](nbytes, std::nothrow);
}

void CpuAllocator::Free(void* ptr) noexcept {
  // operator delete[] on nullptr is defined as a no-op.
  ::operator delete[](ptr);
}

IAllocator* DefaultCpuAllocator() noexcept {
  // LEARNER: Function-local static = thread-safe init since C++11, lives for
  // the process lifetime. Fine for a default host allocator.
  static CpuAllocator instance;
  return &instance;
}

BufferPtr MakeAllocatedBuffer(void* ptr, IAllocator* alloc) {
  if (ptr == nullptr) {
    return BufferPtr{};
  }
  // Capture alloc by value (pointer). Allocator must outlive the buffer —
  // true for DefaultCpuAllocator and for Session-owned EP allocators later.
  return BufferPtr(ptr, [alloc](void* p) {
    if (alloc != nullptr) {
      alloc->Free(p);
    }
  });
}

BufferPtr MakeNonOwningBuffer(void* ptr) {
  // LEARNER: empty deleter = "I am only borrowing this pointer".
  return BufferPtr(ptr, [](void*) noexcept {});
}

}  // namespace eduort
