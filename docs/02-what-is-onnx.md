# 02 — What is ONNX? (and how we load it)

**Lands with:** PR3b  
**Code:** `include/eduort/graph.h`, `include/eduort/onnx_loader.h`, `src/ir/onnx_loader.cpp`

## The big picture

**ONNX** (Open Neural Network Exchange) is a **file format + operator catalog** for neural networks. A `.onnx` file is **not** executable machine code — it is a **description** of a computation graph that a **runtime** (like eduort) interprets.

```text
Training framework  ──export──►  model.onnx  ──load──►  Runtime  ──Run──►  outputs
   (PyTorch, …)                 (protobuf)            (eduort)
```

## Inside a `.onnx` file

Wire format = **Protocol Buffers**. Root message: `ModelProto`.

| Field | Meaning |
|-------|---------|
| `ir_version` | ONNX IR version of the file |
| `opset_import` | Which operator set version(s) the graph expects |
| `graph` | The `GraphProto`: nodes, initializers, inputs, outputs |

### GraphProto

| Field | Meaning |
|-------|---------|
| `node[]` | Operators (`Add`, `MatMul`, …) with inputs/outputs/attributes |
| `initializer[]` | Constant tensors (weights) as `TensorProto` |
| `input[]` / `output[]` | Named graph I/O (`ValueInfoProto` + type/shape) |

**Specs to bookmark:**

- [ONNX IR](https://github.com/onnx/onnx/blob/main/docs/IR.md)
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- Vendored schema: `third_party/onnx/onnx.proto` (pin v1.14.1)

## eduort load path (PR3a + PR3b)

```text
.onnx bytes
    │  LoadModelProtoFromFile   (PR3a)
    ▼
onnx::ModelProto
    │  LoadGraphFromModelProto  (PR3b)
    ▼
eduort::Graph   { nodes, initializers, graph_inputs/outputs, opset_version, … }
```

We **do not** keep raw protobuf in Session. The Graph IR is the teaching surface.

## Policies applied at load (K14)

| Check | Behavior |
|-------|----------|
| Default domain | Only `""` / `ai.onnx`; rewrite both to `""` |
| Other domains | Fail (`ai.onnx.ml`, custom, …) |
| Opset version | Must be in **[11, 17]** |
| `ir_version` | Prefer **[3, 9]**; **warn** if outside, do not hard-fail |

## Initializers vs graph inputs

ONNX allows a name to be **both** a graph input and an initializer (default value). eduort stores:

- `graph_inputs` — all `GraphProto.input` names  
- `initializers` — map name → host `Tensor`  

Session (later) decides which feeds are required vs optional defaults.

## What PR3b does *not* do

- Topological sort / cycle detection → **PR4**  
- Kernel registry / execution → **PR5+**  
- Shape inference engine → gradual with ops  

## Try it

```bash
# After build:
./build/tests/loader_test

# Optional Python helpers (pip install onnx):
python3 scripts/export_mlp.py
python3 scripts/check_model_ops.py testdata/models/add_two_nodes.onnx
```
