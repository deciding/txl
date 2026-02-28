# TXL (Triton Xtra Language) - Project Workflow

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
- `docker/nsa_prefill.py` - NSA prefill benchmark (30s timeout)

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
