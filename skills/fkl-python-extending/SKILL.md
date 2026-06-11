---
name: fkl-python-extending
description: Add a new FKL operation to the fkl-python front-end (operations.py descriptor, codegen, tests). Use when wrapping a C++ FKL Operation struct for Python, when adding a new op to fkl_python, or when codegen produces invalid C++ for a new op.
---

# Extending fkl-python with a new FKL operation

## Architecture recap

- `fkl/operations.py` - one Python descriptor class per C++ Operation.
- `fkl/codegen.py`    - threads ChainState (dtype+shape) through the chain
                        and emits ONE host+device .cu with extern "C" fkl_entry.
- `fkl/backend.py`    - clang/nvcc single-step compile + disk cache.
- `fkl/jit.py`        - compose() + FusedKernel hot path (ctypes).

The kernel is defined by C++ TYPES; runtime VALUES go through a flat
float params[] array. Your descriptor must keep that separation.

## Step 1: read the C++ struct first

Find it under FusedKernelLibrary/include/fused_kernel/algorithms/. Identify:
- Parent type: UnaryOperation (no params), BinaryOperation (1 params),
  TernaryOperation (params+backIOp), ReadBack/IncompleteReadBack (BVF),
  Read/WriteOperation (memory).
- The available build() overloads (gives you the expression to emit).
- InputType -> OutputType transform (drives out_dtype/out_shape).

## Step 2: write the descriptor

```python
class MyOp(Op):
    name = "MyOp"                  # exact C++ struct name

    def __init__(self, value):
        self._v = float(value)

    @property
    def values(self):              # scalars appended to params[], chain order
        return [self._v]

    def out_dtype(self, dt):       # only if the op changes type
        return dt.with_base("float32")

    def out_shape(self, shape):    # only if the op changes geometry
        return shape

    def cpp(self, state, pbase):   # the C++ build() expression
        # NEVER inline values; reference params[pbase + i]
        return f"MyOp<{state.dtype}>::build((float)params[{pbase}])"

    def token(self, state):        # cache key contribution: TYPES ONLY
        return f"MyOp<{state.dtype}>"
```

Rules:
- `token()` must contain everything that changes the GENERATED C++ and
  nothing that doesn't. Values in token() = cache misses on every value
  change = recompile storm. Values NOT in params[] = silently stale kernels.
- Vector values: use `pack_value(dt, raw)` + `make_expr(dt, pbase)` from
  fkl.types (see _BinaryValueOp) and implement `bind(dt)`.
- If a wrapper op delegates to an inner op (see StaticLoop), forward
  `bind()` to it, or collect_params will crash with missing `_vals`.

## Step 3: register and test

1. Export in `fkl/__init__.py` (import + __all__).
2. Add a test in tests/test_operations.py with a CPU reference:
```python
def t_myop():
    src = [float(i) for i in range(64)]
    out = fkl.compose(fkl.TensorRead(), fkl.MyOp(2.0), fkl.TensorWrite())(
        dev_f32(src, 64))
    check("MyOp", unf32(out.copy_to_host(), 64), [ref(v) for v in src])
```
3. Debug codegen without compiling:
   `print(fkl.compose(...).source_for("float32", (64, 1, 1)))`
4. On compile failure the .cu is kept in ~/.cache/fkl/ - compile it
   manually with nvcc to iterate fast:
   `nvcc -std=c++20 -arch=sm_120 -shared -Xcompiler -fPIC -I$FKL_INCLUDE <file>.cu -o /tmp/t.so`

## Known C++ traps (cost us real debugging time)

1. Tensor has NO PtrDims ctor: use
   `Tensor<T>(ptr, w, h, planes, 1, MemType::Device)`.
2. Batch ReadBack ops (Crop with std::array<Rect,B>) return Read<BatchRead>
   whose InstanceType is ReadType -> BackFuser does NOT auto-fuse them with
   the previous read. Codegen wraps them as `fuse(read_expr, batch_expr)`.
   This is keyed off the op having a truthy `_batch` attribute.
3. FKL's LTS-C++17 branch is the supported base (it carries the fix for
   issue #244, ColorConversion FusedOperation aliases). On stale `main`
   the BGR2GRAY-family aliases fail to compile (see #244/#248).
4. Ops whose alias resolves to a type WITHOUT build() exist; always check
   the alias chain in the header, not just the struct name.
5. CUDA 13.3 deprecation warnings for long4/ulong4 are noise; only grep
   stderr for "error".
6. TensorSplit semantics: planes = BATCH (thread.z), color_planes =
   channels. Build the output Tensor as
   `Tensor<base>(ptr, w, h, BATCH, channels, ...)` — NOT (channels, 1).
   exec() offsets by planePixels per channel within each z-plane.
7. Executor<DivergentBatchTransformDPP> does NOT compile on main:
   fuseBackSequence calls fuse_back<IOps...> with explicit template args
   over a const tuple -> rvalue-ref binding error (upstream bug). The
   divergent codegen launches launchDivergentBatchTransformDPP_Kernel
   directly with buildOperationSequence(...) lvalues and a generated
   PySequenceSelector (1-based at(z), FKL convention).
8. IOpSequences passed to the divergent kernel must be LVALUES (const auto
   seqN = buildOperationSequence(...)), not temporaries inlined in the
   launch expression.
9. When changing what generate_cu emits for the SAME inputs, bump
   CODEGEN_VERSION (part of the cache key) or stale .so files will be
   reused silently.

## Backend/shim notes (clang)

clang <= 21 lags CUDA 13.x. backend.py auto-creates shims in ~/.cache/fkl:
- empty texture_fetch_functions.h / texture_indirect_functions.h
- a fatbinary wrapper translating --image= to --image3= syntax
- -D_NV_RSQRT_SPECIFIER=noexcept(true) for glibc >= 2.42
If a future CUDA breaks clang again, extend `_ensure_clang_shims()`.
If clang fails entirely, compile() silently falls back to nvcc and the
choice is remembered in ~/.cache/fkl/.backend.
