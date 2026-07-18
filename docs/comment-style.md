# Comment style guide (pedagogy gates)

Educational value is a **first-class** requirement, not an afterthought. Reviewers (including future-you) should reject PRs that dump clever code without explanation.

## The `LEARNER:` tag

Use a short block comment when introducing a concept a newcomer might not know:

```cpp
// LEARNER: A topological order of a DAG visits every node only after all of
// its inputs have been produced. That is exactly the order we must run ops
// so MatMul never sees an uninitialized weight tensor.
// Spec: ONNX IR.md — "Graphs"
std::vector<const Node*> TopoSort(const Graph& g);
```

### When to use `LEARNER:`

| Use it | Skip it |
|--------|---------|
| ONNX IR / operator semantics | Obvious C++ syntax |
| Why an algorithm exists | Restating the next line of code |
| Device / memory ownership rules | Logging noise |
| Spec citations (`Operators.md`, `IR.md`) | Copy-pasting the entire spec |

## Spec citations

Prefer a stable anchor:

```text
// Spec: ONNX Operators.md — Gemm (opset 11+)
// Spec: ONNX IR.md — Tensor shapes
```

## File headers

Every non-trivial `.h` / `.cpp` should start with:

```cpp
// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: (optional one-liner for the file's job)
```

## Docs chapters

Feature PRs update the matching chapter under `docs/` in the **same** PR (see design “Pedagogy acceptance”). Progressive index:

| Doc | Lands with |
|-----|------------|
| `00-overview.md` | PR1 |
| `01-building.md` | PR1 |
| `02-what-is-onnx.md` | PR3b |
| `03-tensors-and-memory.md` | PR2 |
| `04-graph-and-topo.md` | PR4 |
| `05` / `07` registry + EP | PR5 |
| `06-session-run.md` | PR9a |
| `08-cuda-path.md` | PR10–11 |
| `09-testing.md` | PR2 |
| `operators.md` | PR6+ |
| `walkthrough-mlp.md` | PR9a |

## What “done” means for comments in a PR

- [ ] Public headers explain types in plain language  
- [ ] Non-obvious algorithms have `LEARNER:` + complexity note where relevant  
- [ ] Spec links for ops/IR touched in the PR  
- [ ] Matching `docs/` chapter updated

## Status early-return macros

Do **not** hand-roll repeated blocks like:

```cpp
if (!st.ok()) {
  return st;
}
// or
if (!x.ok()) {
  return x.status();
}
```

Use `include/eduort/macros.h` instead:

- `EDUORT_RETURN_IF_ERROR(status_expr)`
- `EDUORT_ASSIGN_OR_RETURN(lhs, status_or_expr)`

This keeps fallible call chains readable and consistent across PRs.

