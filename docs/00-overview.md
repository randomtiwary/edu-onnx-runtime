# 00 — Overview: what is this project?

## In one sentence

`eduort` loads an ONNX model file, walks its computational graph in a legal order, and runs each operator’s **kernel** on CPU (later optionally on GPU).

## Why ONNX?

**ONNX** (Open Neural Network Exchange) is a common file format for neural networks. Training tools export a `.onnx` file; a **runtime** reads that file and performs **inference** (forward pass only — no training).

The file is **Protocol Buffers** data. The root message is `ModelProto`, which contains a `GraphProto` of `NodeProto` operators connected by named tensors.

Primary specs (bookmark these):

- [ONNX IR](https://github.com/onnx/onnx/blob/main/docs/IR.md) — what a model *is*
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md) — what each op *means*
- [onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto) — the protobuf schema

## How a runtime is usually structured

```text
.onnx file
   │  protobuf parse
   ▼
Graph IR  (nodes, edges, weights)
   │  validate + topological sort
   ▼
Session plan  (which kernel, which device)
   │  for each node in order
   ▼
Kernel::Compute  (CPU or CUDA)
   │
   ▼
Output tensors (host memory)
```

Production systems (e.g. ONNX Runtime) add fusions, pools, many EPs, and huge opsets. We keep the **same shape**, tiny size.

## What “execution provider” means

An **Execution Provider (EP)** is a backend that can create kernels and allocate memory for a device class:

| EP | Device | Role in eduort |
|----|--------|----------------|
| CPU | Host RAM + CPU instructions | Always available; full MVP opset |
| CUDA | NVIDIA GPU | Optional; subset of ops; may fail probe |

The **Session** chooses an EP per node (design Key Decision K13) and inserts host↔device copies when producer and consumer disagree on device.

## What is *not* in PR1

PR1 only proves the **build system and documentation spine**. There is no model loading yet. That starts in PR3.

## Next chapter

- Building: [02-building.md](02-building.md)
- Full design: [design.md](design.md)
- Roadmap: [milestones.md](milestones.md)
