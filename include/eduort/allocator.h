// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: An Allocator owns the strategy for obtaining raw bytes on a device.
// Real runtimes (ORT, TensorFlow) use arenas / pools; we start with the simplest
// possible host allocator (new[] / delete[]) so the ownership story is obvious.
//
// CUDA will add a device allocator in PR10. The IAllocator interface is the
// seam that lets Tensor (and later Session) not hard-code malloc forever.
//
// Spec / design: docs/design.md § Memory model

#pragma once

#include <cstddef>
#include <memory>

namespace eduort {

// LEARNER: DeviceKind tags *where* a buffer lives. Freeing a CUDA pointer with
// host delete[] (or vice versa) is undefined behaviour — we store the kind
// next to every buffer so deleters can assert they match.
enum class DeviceKind {
  kCPU = 0,
  kCUDA = 1,
};

const char* DeviceKindName(DeviceKind device) noexcept;

// ---------------------------------------------------------------------------
// IAllocator — abstract "give me N bytes on this device"
// ---------------------------------------------------------------------------
class IAllocator {
 public:
  virtual ~IAllocator() = default;

  virtual DeviceKind device() const noexcept = 0;
  virtual const char* Name() const noexcept = 0;

  // Allocate `nbytes` of storage. Returns nullptr only on failure (OOM).
  // nbytes == 0: implementation may return a non-null unique placeholder or
  // nullptr; Tensor treats 0-byte tensors as empty and never dereferences.
  virtual void* Allocate(std::size_t nbytes) = 0;

  // Free a pointer previously returned by Allocate on *this* allocator.
  // Free(nullptr) is a no-op.
  virtual void Free(void* ptr) noexcept = 0;
};

// ---------------------------------------------------------------------------
// CpuAllocator — host RAM via ::operator new[]
// ---------------------------------------------------------------------------
class CpuAllocator final : public IAllocator {
 public:
  DeviceKind device() const noexcept override { return DeviceKind::kCPU; }
  const char* Name() const noexcept override { return "CpuAllocator"; }

  void* Allocate(std::size_t nbytes) override;
  void Free(void* ptr) noexcept override;
};

// Process-wide default host allocator (never null).
// LEARNER: Session will later hold per-EP allocators; for PR2 this default is
// enough for Tensor::Create on CPU.
IAllocator* DefaultCpuAllocator() noexcept;

// Shared ownership of a raw buffer with a custom deleter.
// Used by Tensor so host / CUDA / external blobs share one representation.
using BufferPtr = std::shared_ptr<void>;

// Build a BufferPtr that frees via `alloc->Free` when the last owner drops.
// `ptr` may be nullptr (empty tensor).
BufferPtr MakeAllocatedBuffer(void* ptr, IAllocator* alloc);

// Build a BufferPtr with a **no-op** deleter (FromHostBlob).
// LEARNER: The caller still owns the memory; Tensor must not free it.
BufferPtr MakeNonOwningBuffer(void* ptr);

}  // namespace eduort
