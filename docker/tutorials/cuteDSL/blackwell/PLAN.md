# GEMM Implementation Comparison

## Versions Overview

| Version | File | Pipeline Stages | Complexity | Use Case |
|---------|------|----------------|-----------|----------|
| gemm_0 | `dense_gemm_0.py` | 4 AB, 1 acc | Tutorial | 4-stage pipelining example |
| gemm_1 | `dense_gemm_1.py` | 1 AB, 1 acc | Tutorial | Simple 1-stage pipeline |
| gemm_2 | `dense_gemm_2.py` | 1 AB, 1 acc | Tutorial | Low-level mbarrier API |
| full | `dense_gemm.py` | Configurable | Production | High-performance production |

## gemm_0 vs dense_gemm.py Comparison

| Feature | gemm_0 (Tutorial) | dense_gemm.py (Full-featured) |
|---------|-------------------|------------------------------|
| **Pipeline Stages** | Fixed (4 AB, 1 acc) | Configurable (`num_ab_stage`, `num_acc_stage`) |
| **Cluster Shape** | Fixed (1,1) | Configurable (`cluster_shape_mn`) |
| **2-CTA MMA** | No | Yes (`use_2cta_instrs`) |
| **TMA Store** | No (autovec) | Yes (`use_tma_store`) |
| **TMA Multicast** | No | Yes (reduces L2 traffic) |
| **Prefetch** | Simple (`prefetch_stages`) | Advanced (with `try_wait`) |
| **Data Types** | fp16 only | fp16, bf16, tf32, int8, fp8 |
| **Epilogue** | Simple | Flexible (epilogue_op lambda) |
| **Code Lines** | ~500 | ~1800 |
| **Configuration** | Hardcoded | Full CLI arguments |

## Key Additional Features in dense_gemm.py

### 1. TMA Multicast
- Reduces L2 cache traffic by broadcasting loads to multiple CTAs in cluster
- Uses `cpasync.create_tma_multicast_mask()`

### 2. 2-CTA MMA
- Uses `cta_group=2` for larger MMA tiles
- 128x256 instead of 128x128 with cta_group=1

### 3. TMA Store
- Uses TMA to store output directly to GMEM
- vs autovec store in tutorial versions

### 4. Cluster Support
- Multiple CTAs working together
- Proper synchronization with multicast

### 5. Deferred Sync
- Uses `defer_sync=True` for better pipeline control

### 6. Try-Wait (Speculative Prefetch)
- Uses `try_wait()` for speculative prefetching
- Allows overlapping computation with memory operations

## Quick Reference: Which Version to Use?

| Scenario | Recommended Version |
|----------|-------------------|
| Learning GEMM internals | gemm_1 or gemm_2 |
| Understanding 4-stage pipeline | gemm_0 |
| Low-level mbarrier API study | gemm_2 |
| Production high-performance | dense_gemm.py |
| Quick benchmark | Any tutorial version |
