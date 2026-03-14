# TXL (Triton Xtra Language) - Project Workflow

**Important**: Do NOT commit or push changes without explicit user permission.

## Initial Build

First time building TXL wheel for Linux:

```bash
# 1. Initialize git submodules
git submodule update --init --recursive

# 2. Download LLVM (if not already present)
# Place llvm-7d5de303-ubuntu-x64/ in project root

# 3. Build Docker image and create wheel
./tools/build-wheel-docker.sh -n

# Wheel will be in: output/txl-3.5.1-cp312-cp312-manylinux_2_35_x86_64.whl
```

Build flags:
- `-n` - Apply TXL patches to Triton (run once, first time only)
- `-j N` - Number of parallel jobs (default: 8)
- `-c` - Clean build directories before build
- `--no-cache` - Rebuild Docker image without cache

## Rebuild

After making code changes, use incremental rebuild (much faster):

```bash
# Fast incremental rebuild (uses ninja, only rebuilds changed files)
./tools/build-wheel-docker.sh -r

# Rebuild with clean (full rebuild)
./tools/build-wheel-docker.sh -r -c
```

Notes:
- Conda environment is persisted in `txl-conda/` directory
- Build artifacts are persisted in `thirdparty/triton/build/`
- Uses clang by default (less memory than gcc)

## Test

Test TXL wheel on Modal's cloud H100 GPUs:

```bash
# Run test with default volume (txl-dump)
./tools/modal_tests.sh flash_attention.py
./tools/modal_tests.sh mla_decoding.py
./tools/modal_tests.sh nsa_prefill.py

# Run with custom test name
./tools/modal_tests.sh flash_attention.py my-test

# Run with custom volume name
./tools/modal_tests.sh flash_attention.py my-test txl-dump

# Output files:
# - docker/dumps/{test_name}_{timestamp}.log - Console output
# - docker/dumps/{test_name}_{timestamp}/ - Dump files (kernel caches)
```

Available Modal test scripts:
- `docker/flash_attention.py` - Flash attention benchmark
- `docker/mla_decoding.py` - MLA decoding benchmark
- `docker/nsa_prefill.py` - NSA prefill benchmark (1800s timeout)

### Debug Environment Variables

Pass debug environment variable to Modal container:

```bash
# Run with TXLGPU pipeliner debug
TRITON_LLVM_DEBUG_ONLY=txlgpu-pipeliner ./tools/modal_tests.sh nsa_prefill.py debug-test txl-dump
```

The debug output will be in `docker/dumps/{test_name}_{timestamp}.log`.

Notes:
- All tests save dump files to Modal volume `txl-dump`
- Dump files are automatically downloaded after test completes
- Use `--force` flag in volume get to overwrite existing local directories

## Debug Tips

### Local Debug (if you have GPU)
```bash
TRITON_LLVM_DEBUG_ONLY="triton-gpu-taskid-propagate" \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=dump \
TRITON_ALWAYS_COMPILE=1 \
python python/txl/tests/fused-attention.py
```

### CUDA Debug
```bash
CUDA_COREDUMP_SHOW_PROGRESS=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 \
CUDA_LAUNCH_BLOCKING=1 \
cuda-gdb
```

### Memory Issues
If build runs out of memory:
```bash
# Reduce parallel jobs
./tools/build-wheel-docker.sh -j 4
```

### GLIBCXX Error
If you get 'GLIBCXX_3.4.30' not found:
```bash
conda install -c conda-forge gcc=12.1.0
```

## Other Notes

- **Don't push for every change** - Only push when user explicitly asks
- Use `./tools/build-wheel-docker.sh -r` for incremental rebuilds after code changes
- Use `./tools/build-wheel-docker.sh -r -c` for clean rebuilds when build issues occur
- The `-n` flag should only be used once when setting up the project for the first time
- LLVM and conda directories are excluded from git (too large)

## Code Development

### Important: Never modify patch/triton directly

This repo has two copies of Triton code:
- `thirdparty/triton/` - Git submodule (main Triton codebase) - **MODIFY HERE**
- `patch/triton/` - TXL patches applied to Triton - **DO NOT MODIFY**

**Correct workflow**:
1. Edit code in `thirdparty/triton/` (the submodule)
2. Build and test with `./tools/build-wheel-docker.sh -r`
3. Repeat steps 1-2 until fix is verified
4. Only when explicitly requested by user, copy changes to `patch/triton`:
   ```bash
   bash tools/cp_from_triton.sh
   ```
5. Commit the patch changes

### Workflow: Making Changes

1. **Edit in submodule** (thirdparty/triton):
   ```bash
   # Make changes to code in thirdparty/triton/
   ```

2. **Build and test**:
   ```bash
   ./tools/build-wheel-docker.sh -r
   # Run tests
   ```

3. **Repeat** until fix is verified

4. **Copy to patch/triton** (only when user requests):
   ```bash
   bash tools/cp_from_triton.sh
   ```

5. **Commit** (only when user requests)

### Scripts

- `tools/cp_to_triton.sh` - Copy patch/triton → thirdparty/triton
- `tools/cp_from_triton.sh` - Copy thirdparty/triton → patch/triton
- `tools/diff_triton.py` - Compare patch/triton vs thirdparty/triton

### Git Tips

- Remove trailing slashes from `cp -r` commands to avoid issues
- Submodule changes are tracked separately from main repo
- Use `.gitignore` patterns like `llvm-*`, `txl-conda/` for large build artifacts
- **Never modify patch/triton directly** - only copy from thirdparty/triton

## Debugging Pass Failures

When encountering `RuntimeError: PassManager::run failed`, follow this workflow:

### Step 1: Identify the Failing Pass

Run test with TXLGPU pipeliner debug flag to see which stage fails:

```bash
# Set the debug env var BEFORE running the test script
TRITON_LLVM_DEBUG_ONLY=txlgpu-pipeliner ./tools/modal_tests.sh nsa_prefill.py debug-test txl-dump
```

The log will show stages like:
```
[txlgpu-pipeliner]: SoftwarePipeliner After SmemAllocs
[txlgpu-pipeliner]: DONE
[txlgpu-pipeliner]: SoftwarePipeliner After TmemAllocs
[txlgpu-pipeliner]: DONE
...
[txlgpu-pipeliner]: SoftwarePipeliner After lowerLoads
[txlgpu-pipeliner]: DONE

python: ...Assertion failed...
```

The crash happens AFTER the last `DONE` printed.

### Step 2: Find the Failing Stage

TXLGPU SoftwarePipeliner.cpp stages (in order):
1. After SmemAllocs
2. After TmemAllocs
3. After Removing RedundantTMEMAllocs
4. After Mbars
5. After lowerLoads ← Crash happens after this
6. After lowerSmemLoadStores
7. After MemDesc
8. After lowerDotXOps
9. After wgmma

The bug is in the pass between the last printed stage and the next stage.

### Step 3: Add Debug Prints

Edit the failing pass in `thirdparty/triton/third_party/nvidia/lib/Dialect/TXLGPU/Transforms/SoftwarePipeliner.cpp`:

```cpp
void lowerSmemLoadStores(ModuleOp moduleOp) {
  LDBG("[DEBUG] lowerSmemLoadStores: Processing SmemLoadOps\n");
  moduleOp->walk([&](tt::SmemLoadOp op) {
      lowerSmemLoad(op);
  });
  LDBG("[DEBUG] lowerSmemLoadStores: Processing SmemStoreOps\n");
  // ... add more debug prints
}
```

Then rebuild:
```bash
./tools/build-wheel-docker.sh -r
```

### Step 4: Check the Log

The log file will be at `docker/dumps/{test_name}_{timestamp}.log`. Look for:
- `[txlgpu-pipeliner]:` - pipeliner stages
- `[DEBUG] lowerSmemLoadStores:` - our custom debug prints
- `python: ...Assertion failed` - the actual error

### Step 5: Extract MLIR at Failing Stage

To analyze the IR that causes the crash:

1. Find line numbers in the log:
```bash
grep -n "SoftwarePipeliner After lowerLoads\|DONE" docker/dumps/{test_name}.log
```

2. Extract the MLIR between stages:
```bash
# Extract lines between "After lowerLoads" and "DONE"
sed -n '5737,6843p' docker/dumps/{test_name}.log > docker/dumps/lowerLoads.mlir
```

3. Upload to gist for analysis:
```bash
gh gist create docker/dumps/lowerLoads.mlir --public -d "MLIR after lowerLoads"
```

### Step 6: Analyze Type Mismatch

Common issue: `DenseElementsAttr` type mismatch - when MLIR tries to create a constant with float attribute type that doesn't match tensor element type.

Look for:
- Operations with operands of mixed types (bf16 vs f32)
- Constants where the value type doesn't match the tensor element type
- e.g., `dense<0xFF800000>` with type `tensor<64xf32>` - the hex value is negative infinity in f32 bit pattern

### Real Example: NSA topk=2048 Bug

In practice:
1. Crash happened after `SoftwarePipeliner After lowerLoads` → `DONE`
2. This means bug is in `lowerSmemLoadStores` function
3. Error: `floatAttr.getType() == eltType` assertion failed
4. The issue was in how `getRegType()` determines types for SmemLoadOp - wrong type caused DenseElementsAttr creation to fail

## Debug Helpers

TXL provides debug utilities in `TXLUtility.h`:

```cpp
#include "triton/Analysis/TXLUtility.h"

txlDebugMsg("message", operation);
txlDebugMsg("message", value);
txlDebugMsg("message", type);
txlDebugMsg("message", SmallVector<Value>{...});
txlDebugMsg("message", SmallVector<Operation*>{...});
```

## Debugging smem_load vs frag_smem_load

### Problem Summary

When replacing `frag_smem_load` with `smem_load` in NSA kernel, the compilation fails with encoding mismatch errors.

### Root Cause

1. **Python API difference**:
   - `smem_load(mem_desc, layout=None)` - layout is optional, can be None
   - `frag_smem_load(mem_desc, shape, layout)` - layout is required

2. **C++ backend behavior**:
   - When `layout` is None: Python creates a dummy `regType` (tensor<1x1xi32>) and sets `txl.with_reg_type = 0`
   - When `layout` is provided: Python creates real regType and sets `txl.with_reg_type = 1`

3. **RemoveLayoutConversions pass**:
   - `FragSmemLoadOp` has canonicalization in Ops.cpp: `cvt(frag_smem_load) -> frag_smem_load`
   - `SmemLoadOp` did NOT have this canonicalization
   - This causes redundant layout conversion paths to be created

4. **TXLGPUPipeline crash**:
   - When `with_reg_type = 0`, `lowerSmemLoad` uses dummy regType which has NO encoding
   - This causes `llvm::dyn_cast<DistributedEncodingTrait>` assertion failure

### Solution (Two Parts)

1. **Add canonicalization in Ops.cpp** (fold ConvertLayout into SmemLoad):
   ```cpp
   // In lib/Dialect/TritonGPU/IR/Ops.cpp
   // cvt(smem_load) -> smem_load.
   if (auto smemLoad = dyn_cast<SmemLoadOp>(arg)) {
     rewriter.setInsertionPoint(arg);
     auto newOp = rewriter.replaceOpWithNewOp<SmemLoadOp>(op, op->getResult(0).getType(),
                                                          smemLoad.getSrc(),
                                                          smemLoad.getRegType(),
                                                          smemLoad.getCtaId());
     if (auto attr = smemLoad->getAttrOfType<IntegerAttr>("txl.with_reg_type")) {
       newOp->setAttr("txl.with_reg_type", attr);
     }
     return success();
   }
   ```

2. **Fix lowerSmemLoad in SoftwarePipeliner.cpp**:
   ```cpp
   // When with_reg_type = 0, use result type instead of dummy regType
   if (withRegType == 0) {
       retType = op.getResult().getType();  // Has actual encoding
   }
   ```

### Using diff_select to Find Bug Origin

Use `diff_mode='ttgir'` and `diff_select=N` in txl.jit to see IR at each pass:

```python
@txl.jit(diff_mode="ttgir", diff_select=10, log_dir="/workspace/dump/smem/")
def txl_mla0(...):
```

- `diff_select=10` shows pass 10's diff
- SoftwarePipeliner is pass 22
- Binary search between passes to find which introduces the bug

### Key Files Modified

- `thirdparty/triton/lib/Dialect/TritonGPU/IR/Ops.cpp` - Add SmemLoadOp canonicalization
- `thirdparty/triton/third_party/nvidia/lib/Dialect/TXLGPU/Transforms/SoftwarePipeliner.cpp` - Fix lowerSmemLoad

# CuTeDSL Dense GEMM Tutorial

This section documents the conventions and notation used in CuTeDSL dense GEMM kernels for Blackwell B200 GPU.

## CuTe Tensor Naming Conventions

In NVIDIA CUTLASS (especially CuTe), variable names like `tCrC` follow a rigorous naming convention. The name encodes four pieces of information: `<prefix><partitioner><storage><data_role>`.

### Naming Components

| Component | Symbol | Meaning |
|-----------|--------|---------|
| **Prefix** | `t` | Thread-level tensor (this thread's private slice) |
| **Partitioner** | `C`, `A`, `B` | Thread mapping rules (tC, tA, tB) |
| **Storage** | `r`, `s`, `g` | `r`=Register, `s`=Shared Memory, `g`=Global Memory |
| **Data Role** | `A`, `B`, `C` | GEMM operands: A=first factor, B=second factor, C=accumulator |

### Common Examples

| Variable | Meaning |
|----------|---------|
| `tCsA` | Thread's view of A matrix in Shared Memory (partitioned by C rules) |
| `tCrA` | Thread's fragment of A matrix in Registers (for MMA compute) |
| `tAgA` | Thread's view of A matrix in Global Memory (for TMA loads) |
| `tCrC` | Thread's accumulator fragment in Registers |
| `tCgA` | Thread's partition of A for MMA (Global memory view) |
| `tDtC` | TMEM source tensor for epilogue copy |
| `tDgC` | Global memory destination tensor for epilogue copy |

## My Notation (Custom Notation System)

This section documents the custom notation used in the dense_gemm_2.py tutorial to describe tensor shapes at different levels.

### Terminology

| Term | Meaning |
|------|---------|
| `per_mma_atom` | Elements within one MMA instruction (spatial) |
| `per_mma_tile` | Tile in the MMA grid (spatial) |
| `per_tma_atom` | Elements within one TMA copy instruction |
| `per_tma_tile` | Tile for TMA operations (spatial) |
| `per_wave` | Pipeline stages that can run in parallel (stages dimension) |
| `per_tide` | Full K loop (runtime iterations) |
| `per_tmem_atom` | Elements within one TMEM copy instruction |
| `per_tmem_tile` | Tile for TMEM operations |

### Shape Notation Examples

```python
# SMEM tensor A: ((128,16),1,4,1)
# My Notation: per_mma_atom((128,16)), per_tma_tile(1,4), per_wave(1)
#   - (128,16) = per_mma_atom (MMA instruction shape)
#   - (1,4) = per_tma_tile (1 MMA_M_tile, 4 MMA_K_tiles)
#   - 1 = per_wave (1 pipeline stage)

# MMA fragment A: (1,1,4,1)
# My Notation: per_mma_atom(1), per_tma_tile(1,4), per_wave(1)
#   - 1 = 1 smem descriptor for whole block

# Accumulator: ((128,256),1,1)
# My Notation: per_mma_atom((128,256)), per_tma_tile(1,1)
#   - (128,256) = MMA instruction produces 128x256 accumulator

# TMEM source: (((64,32),1),1,((1,4),1,1))
# My Notation: per_tmem_atom((64,32)), per_tmem_tile(1), per_mma_tile((1,4)) per_tma_tile(1,1)

# Register tensor: ((64,1),1)
# My Notation: per_tmem_atom((64,1)), per_tmem_tile(1)
```

## Dense GEMM Kernel Structure

The typical dense GEMM kernel follows this structure:

### 1. Setup and Allocation
```python
# Allocate SMEM tensors
sA = smem.allocate_tensor(element_type=io_dtype, layout=a_smem_layout.outer, ...)
sB = smem.allocate_tensor(element_type=io_dtype, layout=b_smem_layout.outer, ...)

# Allocate TMEM for accumulator
tmem = utils.TmemAllocator(...)
tmem.allocate(num_tmem_cols)
```

### 2. Create MMA and TMA
```python
# Tiled MMA
tiled_mma = cute.make_tiled_mma(op)

# TMA atoms
tma_atom_a = cute.make_tma_atom(...)
tma_atom_b = cute.make_tma_atom(...)
```

### 3. Partition and Create Fragments
```python
# Partition GMEM tensors for MMA
tCgA = thr_mma.partition_A(gA)
tCgB = thr_mma.partition_B(gB)
tCgC = thr_mma.partition_C(gC)

# Create MMA fragments (from SMEM)
tCrA = tiled_mma.make_fragment_A(sA)
tCrB = tiled_mma.make_fragment_B(sB)
tCtAcc = tiled_mma.make_fragment_C(acc_shape)

# Create TMA descriptors
tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(tma_atom_a, ...)
tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(tma_atom_b, ...)
```

### 4. Main Loop
```python
for k_tile_idx in range(num_k_tiles):
    # Issue TMA loads
    ab_empty = ab_producer.acquire_and_advance()
    cute.copy(tma_atom_a, tAgA[(None, ab_empty.count)], tAsA[(None, ab_empty.index)], ...)
    cute.copy(tma_atom_b, ...)

    # Execute MMA
    ab_full = ab_consumer.wait_and_advance()
    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
        cute.gemm(tiled_mma, tCtAcc, tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
```

### 5. Epilogue (TMEM → Register → GMEM)
```python
# Sub-tiling for ILP
for i in cutlass.range(cute.size(tDtC, mode=[2])):
    cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)  # TMEM -> Reg
    tCrC.store(tCrAcc.load().to(io_dtype))                    # Convert dtype
    cute.autovec_copy(tCrC, tDgC[None, None, i])             # Reg -> GMEM
```

## Running the Tests

```bash
# Run default version (dense_gemm_2.py - simplified)
./tools/modal_tests.sh tutorials/cuteDSL/01_dense_gemm.py

# Available versions:
# - 0: dense_gemm_0.py (4-stage pipelining)
# - 1: dense_gemm_1.py (1-stage pipelining)
# - 2: dense_gemm_2.py (simplified with detailed comments)
# - original: dense_gemm.py (full-featured)
```

## Key Files

```
docker/tutorials/cuteDSL/
├── 01_dense_gemm.py              # Modal test script
└── blackwell/
    ├── dense_gemm.py             # Original full-featured
    ├── dense_gemm_0.py           # 4-stage pipelining
    ├── dense_gemm_1.py           # 1-stage pipelining
    └── dense_gemm_2.py           # Simplified with detailed comments

thirdparty/cutlass/python/CuTeDSL/cutlass/
├── utils/blackwell_helpers.py    # make_smem_layout_a/b
└── cute/nvgpu/cpasync/helpers.py # tma_partition
```

## Debug Tips

### Add Shape Verification
Use `cute.printf` to verify tensor shapes during kernel execution:
```python
if tidx == 0:
    cute.printf("tensor shape: {}\n", tensor.shape)
```

### Important Notes
- `smem_desc` represents the whole block, not decomposed like MMA fragments
- `tma_partition` requires input tensors folded in shape `(Each_Iter, Num_Iters)`
- `subtile_cnt` in epilogue controls ILP (typically 4 for fp16)
