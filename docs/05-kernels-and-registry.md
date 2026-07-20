# 05 — Kernels and the registry

**Lands with:** PR5  
**Code:** `kernel.h`, `registry.h` / `registry.cpp`, CPU Identity

## Mental model

| Term | Meaning |
|------|---------|
| **Operator** | ONNX name in the graph (`Identity`, `Add`, …) |
| **Kernel** | C++ code that implements that op on a device |
| **Registry** | Table of factories: how to build a kernel |
| **since_version** | Oldest opset for which this kernel is valid |

```text
Node(op_type=Identity)
        │
        ▼
KernelRegistry.Create(domain, op, model_opset, ep="CPU")
        │  pick highest since_version ≤ model_opset
        ▼
IKernel::Compute(OpKernelContext)
```

## Lookup rule

From design **Key Decision K14** (domain + opset + `since_version` policy — see [`docs/design.md`](design.md) § Key Decisions):

Among registrations matching `(canonical domain, op_type, ep_name)` with  
`since_version ≤ model_opset`, choose the **largest** `since_version`.

Example: kernels registered at since 1 and 10; model opset 13 → use since 10.

## OpKernelContext

- `Input(i)` — const tensor from the value map (read-only)  
- `Output(i, dtype, shape)` — allocate via the **EP allocator** passed into the context (`Tensor::Create(dtype, shape, allocator)`), then store under the output name  
- `GetAttr(name)` — node attributes  

## Registration style

```cpp
KernelRegistry reg;
RegisterCpuKernels(reg);  // explicit — no static init order surprises
```

## PR5 scope

Only **Identity** is registered on CPU. Other MVP ops land in PR6–8.

See also: [07-execution-providers.md](07-execution-providers.md)
