<p align="center">
  <img src="docs/images/logo.png" alt="TeraXLang" width="400"/>
</p>

# TeraXLang

TeraXLang is a CUDA kernel-specific DSL built on top of Triton that achieves **SOTA GPU kernel performance** on both **Hopper** (H100) and **Blackwell** (B200) architectures.

## Why TeraXLang?

### Why build another DSL when there are already Tilelang, Gluon, etc.?

I wanted to understand what really happens in Triton:
- What optimizations has Triton done?
- Why do many DSLs claim they can easily outperform Triton?
- What if we add a few more APIs that might harm Triton's generality, but bring superior performance in exchange?

### Why not use Gluon (provided by Triton itself)?

Gluon has removed some auto-optimizations of Triton to be more low-level friendly, and may not support all features needed for new hardware architectures.

## Key Features

- **Minimal Extensions**: Adds only essential methods to Triton (smem, tmem, mbar, TMA operations)
- **Warp-level Primitives**: Efficient warpgroup synchronization and reduction
- **TMA Support**: Hardware-accelerated tensor memory operations
- **Multi-Architecture**: Optimized for both Hopper and Blackwell GPUs

## Performance

### Matmul (H100 80GB HBM3)
M=8192, N=8192, K=1024

| Kernel | TFLOPS |
|--------|--------|
| cuBLAS | 710.4 |
| TXL (hopper_txl_ws_persistent) | 697.7 |

**~2% slower than cuBLAS**

### Flash Attention (H100 80GB HBM3)
batch=16, heads=32, seq_len=16384, head_dim=128

| Kernel | TFLOPS |
|--------|--------|
| FlashAttention3 | 640 |
| TXL (hopper_txl_ws_fa3) | 676.26 |

**~6% faster than FlashAttention3**

### MLA Decoding (H100 80GB HBM3)
TestParam(b=132, s_q=1, s_k=32768, is_varlen=False, is_causal=False, is_fp8=False, topk=None, test_performance=True, is_all_indices_invalid=False, have_zero_seqlen_k=False, block_size=64, h_q=128, h_kv=1, d=576, dv=512, seed=0)

| Kernel | Time (ms) | TFLOPS | GB/s |
|--------|-----------|--------|------|
| HuggingFace MLA | 2.030 | 593 | 2472 |
| TXL MLA | 2.227 | 541 | 2254 |

### NSA Prefill (H100 80GB HBM3)
TestParam(b=1, s_q=16384, s_kv=16384, topk=128, h_q=128, h_kv=1, d_qk=576, d_v=512, seed=0, check_correctness=True, benchmark=True)

| Kernel | Time (us) | TFLOPS |
|--------|-----------|--------|
| Prefill FlashNSA | 2352 | 248.4 |
| TXL NSA | 2193 | 266.4 |

## Quick Start

### Install

```bash
# Clone and build
git clone https://github.com/deciding/txl.git
cd txl
git submodule update --init --recursive

# Build wheel (requires Docker)
./tools/build-wheel-docker.sh -n

# Or install from prebuilt wheel
pip install thirdparty/triton/dist/txl-*.whl
```

### Run Tests on Modal

```bash
# Matmul benchmark
./tools/modal_tests.sh matmul.py

# Flash Attention benchmark
./tools/modal_tests.sh flash_attention.py
```

## Available Kernels

### Matmul
- `hopper_triton_ws_persistent` - Triton persistent matmul on Hopper
- `hopper_txl_ws_persistent` - TXL warp-specialized persistent matmul on Hopper
- `hopper_txl_naive` - TXL naive TMA matmul on Hopper
- `blackwell_txl_ws_persistent` - TXL warp-specialized persistent matmul on Blackwell

### Flash Attention
- `hopper_txl_ws_naive` - TXL warp-specialized FA without ping-pong
- `hopper_txl_ws_pingpong` - TXL warp-specialized FA with ping-pong
- `hopper_txl_ws_fa3` - TXL warp-specialized FA3-style kernel

## Documentation

See [docs/](https://deciding.github.io/txl/) for detailed API documentation.

## Architecture

TeraXLang adds these key operations to Triton:

| Operation | Description |
|-----------|-------------|
| `smem_alloc` | Shared memory allocation |
| `tmem_alloc` | Tensor memory (TMEM) allocation |
| `mbar_alloc` | Mailbox for warpgroup synchronization |
| `tma_load` / `tma_store` | Tensor Memory Accelerator operations |
| `frag_smem_load` / `frag_smem_store` | Fragment-based smem operations |

## License

MIT
