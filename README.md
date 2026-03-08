<img src="docs/images/logo.png" alt="TXL" width="200"/>

# TXL (Triton Xtra Language)

TXL is a CUDA kernel-specific DSL built on top of Triton that achieves **SOTA GPU kernel performance** on both **Hopper** (H100) and **Blackwell** (B200) architectures.

## Key Features

- **Minimal Extensions**: Adds only essential methods to Triton (smem, tmem, mbar, TMA operations)
- **Warp-level Primitives**: Efficient warpgroup synchronization and reduction
- **TMA Support**: Hardware-accelerated tensor memory operations
- **Multi-Architecture**: Optimized for both Hopper and Blackwell GPUs

## Performance

### Matmul (H100 80GB HBM3)
M=8192, N=8192, K=1024

| Kernel | Time (ms) |
|--------|-----------|
| cuBLAS | 710.4 |
| TXL (hopper_txl_ws_persistent) | 697.7 |

**~2% faster than cuBLAS**

### Flash Attention (H100 80GB HBM3)
batch=16, heads=32, seq_len=16384, head_dim=128

| Kernel | TFLOPS |
|--------|--------|
| FlashAttention3 | 640 |
| TXL (hopper_txl_ws_fa3) | 676.26 |

**~6% faster than FlashAttention3**

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

See [docsdocs/) for detailed API documentation.

## Architecture

/](TXL adds these key operations to Triton:

| Operation | Description |
|-----------|-------------|
| `smem_alloc` | Shared memory allocation |
| `tmem_alloc` | Tensor memory (TMEM) allocation |
| `mbar_alloc` | Mailbox for warpgroup synchronization |
| `tma_load` / `tma_store` | Tensor Memory Accelerator operations |
| `frag_smem_load` / `frag_smem_store` | Fragment-based smem operations |

## License

MIT
