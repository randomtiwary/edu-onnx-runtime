# 04 — Graph structure and topological order

**Lands with:** PR4  
**Code:** `validate.h` / `validate.cpp`, `topo_sort.h` / `topo_sort.cpp`

## Why structure before kernels?

A model can be **structurally broken** even if every op name looks familiar:

- An input name that nothing produces (typo)
- Two nodes writing the same output name
- A cycle (A needs B’s output and B needs A’s)

PR4 checks those **wiring** problems only. Unknown `op_type` values are still
allowed — kernels arrive in PR5+.

## Value names

ONNX uses a **flat string namespace** of tensor names in one graph:

```text
graph inputs / initializers  ──►  node inputs
node outputs                 ──►  later node inputs / graph outputs
```

Empty string `""` on a node input means **optional input omitted** (legal).

## Single assignment (SSA-ish)

Each value name should have **at most one producer node**.  
`ValidateStructure` rejects duplicate producers.

## Topological order (Kahn)

Execution must visit producers before consumers. **Kahn’s algorithm:**

1. Build edges: if node A produces `T` and node B consumes `T`, edge **A → B**.
2. Graph inputs and initializers are free “seeds” (no producing node required).
3. Nodes with no pending producers are **ready**.
4. Emit a ready node; unlock its consumers.
5. If some nodes never become ready → **cycle**.

```text
X, W (seeds) ──► Add1 ──► T ──► Add2 ──► Y
topo_order: [Add1, Add2]
```

Result is stored in `Graph::topo_order` as **indices** into `Graph::nodes`.

## API

```cpp
Status st = eduort::ValidateStructure(graph);
StatusOr<std::vector<int>> order = eduort::ComputeTopoOrder(graph);
// or both + assign:
st = eduort::PrepareGraphStructure(graph);  // fills graph.topo_order
```

`PrepareGraphStructure` is what `Session::Create` will call after load.

## What is *not* checked here

| Concern | When |
|---------|------|
| Op is supported / kernel exists | PR5 / Session bind |
| Shapes and broadcast | Ops PRs + Session shape pass |
| Constant folding | Session Create (PR9a) |

## Try it

```bash
./build/tests/topo_sort_test
```
