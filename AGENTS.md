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
