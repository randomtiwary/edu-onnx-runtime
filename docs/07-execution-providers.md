# 07 — Execution providers

**Lands with:** PR5 (CPU shell)  
**Code:** `execution_provider.h`, `cpu_provider.h` / `cpu_provider.cpp`, `planner.h`

## What is an EP?

An **Execution Provider** is a backend:

- **Name** — `"CPU"`, later `"CUDA"`  
- **Allocator** — where `Output()` buffers live  
- **CreateKernel** — build a device-specific kernel for a node  

Session will try EPs in priority order (CUDA if ready, else CPU).

## CPU provider (always available)

```cpp
KernelRegistry reg;
RegisterCpuKernels(reg);
CpuExecutionProvider cpu(&reg);
```

`CanProduceKernel` / `CreateKernel` delegate to the registry with `ep_name="CPU"`.

## Planner bind (skeleton)

```cpp
PrepareGraphStructure(graph);  // topo_order
auto bindings = BindKernels(graph, {&cpu});
// bindings[i].kernel ready for Session::Run (PR9a)
```

If no EP can produce a kernel → `kUnsupportedOperator` with a clear message.

## CUDA

Optional EP + probe state machine arrive in **PR10–11**. Until then only CPU
appears in the EP list.

## Related

- [05-kernels-and-registry.md](05-kernels-and-registry.md)  
- Design: EP interface + mixed-device policy (K13)
