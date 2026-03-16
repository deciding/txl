# GEMM Optimization Plan

Current: dense_gemm_2 (Pipeline API) → Target: dense_gemm.py (full-featured)

## Versions Overview

| Version | File | Description |
|---------|------|-------------|
| dense_gemm | `dense_gemm.py` | Full-featured production kernel |
| dense_gemm_1 | `dense_gemm_1.py` | Low-level mbarrier API |
| dense_gemm_2 | `dense_gemm_2.py` | Pipeline API (4-stage, no cluster) |

## Benchmark Results (8192x8192x4096)

| Kernel | TFLOPS |
|--------|--------|
| torch.matmul (cuBLAS) | ~1700 |
| dense_gemm.py | ~1730 |
| dense_gemm_2.py | ~620 |
| dense_gemm_1.py | ~340 |

## Key Differences: dense_gemm_2 vs dense_gemm.py

| Feature | dense_gemm_2 | dense_gemm.py |
|---------|-------------|---------------|
| Cluster Shape | Fixed (1,1) | Configurable (e.g., (2,1), (2,2)) |
| 2-CTA MMA | No (cta_group=1) | Yes (cta_group=2, 256x128 tile) |
| TMA Store | Autovec store | TMA direct store |
| TMA Multicast | No | Yes (reduces L2 traffic) |
| Prefetch | Basic | try_wait() speculative prefetch |
| Data Types | fp16 only | fp16, bf16, tf32, int8, fp8 |

## Optimization Roadmap

### Step 1: Add Cluster Support
- Add `cluster_shape_mn` parameter
- Modify grid computation for cluster
- Add cluster synchronization
- **Expected gain**: ~20-30% (more parallelism)

### Step 2: Add 2-CTA MMA
- Use `cta_group=2` in MMA
- Increase MMA tile from 128x128 to 256x128
- Update SMEM layouts accordingly
- **Expected gain**: ~50-100% (larger tiles)

### Step 3: Add TMA Store
- Replace autovec store with TMA store
- Add TMEM for accumulator
- **Expected gain**: ~10-20% (better memory coalescing)

### Step 4: Add TMA Multicast
- Add multicast for A/B loads within cluster
- Reduces L2 cache pressure
- **Expected gain**: ~5-10%

### Step 5: Add try_wait Prefetch
- Speculative prefetching
- Better overlap of compute and memory
- **Expected gain**: ~5-10%

### Final: Add Multi-Dtype Support
- bf16, tf32, int8, fp8
- Parameterize all the things
- **Expected**: Full performance

## Target Performance

| Stage | Expected TFLOPS |
|-------|-----------------|
| Start (dense_gemm_2) | ~620 |
| After Step 1 | ~750 |
| After Step 2 | ~1100 |
| After Step 3 | ~1400 |
| After Step 4 | ~1600 |
| After Step 5 | ~1700+ |
