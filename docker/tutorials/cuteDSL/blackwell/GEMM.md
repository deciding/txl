# Blackwell GEMM Kernel Development Guide

This document captures the key patterns and routines for building high-performance GEMM kernels on NVIDIA Blackwell (B200) GPU using CuTeDSL.

## Table of Contents

1. [Shared Memory (SMEM) Usage](#1-shared-memory-smem-usage)
2. [Mbarrier Synchronization](#2-mbarrier-synchronization)
3. [Pipeline Operations](#3-pipeline-operations)
4. [TMA Load Operations](#4-tma-load-operations)
5. [GEMM Computation](#5-gemm-computation)
6. [Epilogue (TMEM → GMEM)](#6-epilogue-tmem--gmem)
7. [Key Performance Notes](#7-key-performance-notes)

---

## 1. Shared Memory (SMEM) Usage

### 1.1 Define SharedStorage Structure

```python
@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]  # mbarrier pointers
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]  # accumulator mbarrier
    tmem_holding_buf: cutlass.Int32  # TMEM allocation metadata
```

- `ab_mbar_ptr`: 2 mbarriers per stage (full + empty) for AB pipeline
- `acc_mbar_ptr`: 2 mbarriers per stage for accumulator pipeline
- `tmem_holding_buf`: Buffer for TMEM allocator

### 1.2 Allocate SMEM

```python
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage)  # Allocate struct in SMEM
```

### 1.3 Allocate Tensor in SMEM

```python
sA = smem.allocate_tensor(
    element_type=io_dtype,          # e.g., cutlass.Float16
    layout=a_smem_layout.outer,       # Outer layout (tiled shape)
    byte_alignment=128,
    swizzle=a_smem_layout.inner,      # Inner swizzle for bank conflict avoidance
)
```

**Key concepts:**
- `layout`: The outer tiled shape from `make_smem_layout_a/b`
- `swizzle`: Bank conflict avoidance pattern (INTER, SW32, SW64, SW128)
- `byte_alignment`: Typically 128 bytes for TMA

### 1.4 SMEM Layout Creation (Host)

```python
# make_smem_layout_a/b(tiled_mma, mma_tiler_mnk, dtype, num_stages)
a_smem_layout = sm100_utils.make_smem_layout_a(
    tiled_mma, mma_tiler_mnk, a.element_type, ab_stages
)

# Layout shape: ((M_atom, K_atom), MMA_M_tiles, MMA_K_tiles, num_stages)
# Example for M=128, K=64, stages=4: ((128,16), 1, 4, 4)
```

---

## 2. Mbarrier Synchronization

### 2.1 Mbarrier Layout

```
ab_mbar_ptr: [ab_mbar_full, ab_mbar_empty]
                  │              │
                  ▼              ▼
              index 0        index 1
```

- **Full mbarrier**: Signals data is ready (TMA→MMA), uses `expect_tx` with bytes
- **Empty mbarrier**: Signals buffer is consumed (MMA→TMA), just sync signal

### 2.2 Initialize Mbarriers

```python
# Get mbarrier pointers
ab_mbar_full = storage.ab_mbar_ptr.data_ptr()      # index 0
ab_mbar_empty = storage.ab_mbar_ptr.data_ptr() + 1  # index 1

# Initialize (warp 0 typically)
if warp_idx == 0:
    cute.arch.mbarrier_init(ab_mbar_full, 1)
    cute.arch.mbarrier_init(ab_mbar_empty, 1)
    cute.arch.mbarrier_init(acc_mbar_ptr, 1)

# Ensure visibility
cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()
```

### 2.3 Mbarrier Operations

| Operation | Function | elect_one? |
|-----------|----------|------------|
| TMA Load arrives on full | `mbarrier_arrive_and_expect_tx(mbar, bytes)` | **YES** |
| MMA arrives on empty | `tcgen05.commit(mbar)` | **YES** |
| Wait for full | `mbarrier_wait(mbar, phase)` | N/A |
| Wait for empty | `mbarrier_wait(mbar, phase)` | N/A |

### 2.4 Synchronization Pattern (Single-Stage)

```python
phase = 1  # Toggle between 0 and 1

for k_tile_idx in range(num_k_tiles):
    # TMA Load: wait for empty buffer
    cute.arch.mbarrier_wait(ab_mbar_empty, phase)
    
    # TMA loads (GMEM → SMEM)
    cute.copy(tma_atom_a, tAgA[(None, k_tile_idx)], tAsA[(None, 0)], 
              tma_bar_ptr=ab_mbar_full)
    
    # TMA arrives on full: elect_one + expect_tx
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(ab_mbar_full, num_tma_copy_bytes)
    
    # MMA: wait for full buffer
    cute.arch.mbarrier_wait(ab_mbar_full, 1 - phase)
    
    # MMA compute...
    
    # MMA arrives on empty: elect_one + tcgen05.commit
    with cute.arch.elect_one():
        cute.nvgpu.tcgen05.commit(ab_mbar_empty)
    
    # Toggle phase
    phase = 1 - phase
```

---

## 3. Pipeline Operations

### 3.1 Pipeline Types (CuTeDSL)

| Pipeline Class | Producer | Consumer | Use Case |
|---------------|----------|----------|----------|
| `PipelineTmaUmma` | TMA | TCGen05MMA | Main GEMM loop |
| `PipelineUmmaAsync` | AsyncThread | TCGen05MMA | Accumulator pipeline |
| `PipelineAsyncUmma` | AsyncThread | TCGen05MMA | Input fusion |

### 3.2 Create Pipeline

```python
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    num_stages=ab_stages,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    tx_count=num_tma_copy_bytes,
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
).make_participants()

acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
    num_stages=acc_stage,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
    barrier_storage=storage.acc_mbar_ptr.data_ptr(),
).make_participants()
```

### 3.3 Pipeline Operations

| Operation | Meaning |
|-----------|---------|
| `producer.acquire_and_advance()` | Wait for empty buffer, get buffer index |
| `consumer.wait_and_advance()` | Wait for full buffer, get buffer index |
| `producer.release()` | Signal empty (buffer consumed) |
| `consumer.release()` | Signal full (data ready) |

### 3.4 Replace Pipeline with Low-Level Mbarrier

**Instead of:**
```python
ab_empty = ab_producer.acquire_and_advance()
cute.copy(..., tma_bar_ptr=ab_empty.barrier)
ab_full = ab_consumer.wait_and_advance()
# MMA...
ab_full.release()
```

**Use:**
```python
# Wait for empty, then load and arrive on full
cute.arch.mbarrier_wait(ab_mbar_empty, phase)
cute.copy(..., tma_bar_ptr=ab_mbar_full)
with elect_one(): mbarrier_arrive_and_expect_tx(ab_mbar_full, bytes)

# Wait for full, then compute
cute.arch.mbarrier_wait(ab_mbar_full, 1 - phase)
# MMA...
with elect_one(): tcgen05.commit(ab_mbar_empty)

# Toggle phase
phase = 1 - phase
```

---

## 4. TMA Load Operations

### 4.1 Create TMA Atoms (Host)

```python
# TMA Copy Atom
op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

# TMA Atom for A
a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
    op, a, a_smem_layout_one_stage, mma_tiler_mnk, tiled_mma
)

# TMA Atom for B
b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
    op, b, b_smem_layout_one_stage, mma_tiler_mnk, tiled_mma
)
```

### 4.2 Prefetch TMA Descriptor (Kernel)

```python
if warp_idx == 0:
    cpasync.prefetch_descriptor(tma_atom_a)
    cpasync.prefetch_descriptor(tma_atom_b)
```

### 4.3 Partition for TMA (Kernel)

```python
# Partition GMEM tensor for MMA first
tCgA = thr_mma.partition_A(gA)  # (MMA, MMA_M, MMA_K, RestK)
tCgB = thr_mma.partition_B(gB)  # (MMA, MMA_N, MMA_K, RestK)

# Then partition for TMA
tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
    tma_atom_a,      # TMA Copy Atom
    0,                # CTA coordinate
    cute.make_layout(1),  # CTA layout
    cute.group_modes(sA, 0, 3),   # SMEM tensor grouped
    cute.group_modes(tCgA, 0, 3),  # GMEM tensor grouped
)

tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
    tma_atom_b, 0, cute.make_layout(1),
    cute.group_modes(sB, 0, 3),
    cute.group_modes(tCgB, 0, 3),
)
```

**Output:**
- `tAsA`: TMA address descriptor for SMEM
- `tAgA`: TMA address descriptor for GMEM

### 4.4 Perform TMA Load

```python
# TMA load from GMEM to SMEM
cute.copy(
    tma_atom_a,
    tAgA[(None, k_tile_idx)],  # GMEM source (k_tile_idx)
    tAsA[(None, 0)],            # SMEM destination (buffer index)
    tma_bar_ptr=ab_mbar_full,   # Mbarrier for arrive
)
```

---

## 5. GEMM Computation

### 5.1 Create Tiled MMA (Host)

```python
op = tcgen05.MmaF16BF16Op(
    io_dtype,           # Input dtype (e.g., Float16)
    acc_dtype,          # Accumulator dtype (e.g., Float32)
    mma_inst_shape_mnk, # MMA instruction shape (128, 256, 16)
    tcgen05.CtaGroup.ONE,  # CTA group (ONE or TWO)
    tcgen05.OperandSource.SMEM,  # Source: SMEM or TMEM
    tcgen05.OperandMajorMode.K,
    tcgen05.OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(op)
```

### 5.2 Create MMA Fragments

```python
# MMA fragments from SMEM (single descriptor per block)
tCrA = tiled_mma.make_fragment_A(sA)  # Shape: (1, 1, 4, stages)
tCrB = tiled_mma.make_fragment_B(sB)  # Shape: (1, 1, 4, stages)

# Accumulator fragment (will be swapped with TMEM pointer later)
acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
tCtAcc = tiled_mma.make_fragment_C(acc_shape)  # Shape: (128, 256)
```

### 5.3 Perform MMA

```python
num_k_blocks = cute.size(tCrA, mode=[2])

for k_block_idx in cutlass.range_constexpr(num_k_blocks):
    k_block_coord = (None, None, k_block_idx, buffer_index)
    cute.gemm(
        tiled_mma,
        tCtAcc,                    # Accumulator (in/out)
        tCrA[k_block_coord],       # A fragment
        tCrB[k_block_coord],       # B fragment
        tCtAcc,                    # Output accumulator
    )
    # Enable accumulate after first k-block
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
```

---

## 6. Epilogue (TMEM → GMEM)

### 6.1 TMEM Allocation

```python
tmem_alloc_barrier = pipeline.NamedBarrier(
    barrier_id=1,
    num_threads=threads_per_cta,
)
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf,
    barrier_for_retrieve=tmem_alloc_barrier,
)
num_tmem_cols = 512
tmem.allocate(num_tmem_cols)

# After sync, retrieve pointer
tmem.wait_for_alloc()
tmem_ptr = tmem.retrieve_ptr(acc_dtype)

# Swap pointer into accumulator
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)
```

### 6.2 Sub-tiling for ILP

```python
subtile_cnt = 4  # Typical for fp16

# Epilogue tiler: divide each MMA tile into subtiles
epi_tiler = (
    (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
)
# epi_tiler = ((128, 64)) for 128x256 MMA tile with 4 subtiles
```

### 6.3 Partition for TMEM Copy

```python
# Divide accumulator and output into epilogue tiles
tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
gC_epi = cute.zipped_divide(tCgC, epi_tiler)

# TMEM copy atom
tmem_atom = cute.make_copy_atom(
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
    cutlass.Float32,
)
tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

# Partition for TMEM copy
tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)  # Source: TMEM
tDgC = tmem_thr_copy.partition_D(gC_epi)         # Destination: GMEM
```

### 6.4 Register Tensors

```python
# Accumulator register (Float32)
tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)

# Output register (convert to output dtype)
tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)
```

### 6.5 Epilogue Copy Loop

```python
# Wait for MMA to complete
with cute.arch.elect_one():
    cute.nvgpu.tcgen05.commit(acc_mbar_ptr)
cute.arch.mbarrier_wait(acc_mbar_ptr, phase=0)

# TMEM → Register → GMEM
for i in cutlass.range(cute.size(tDtC, mode=[2])):
    cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)  # TMEM → Reg
    tCrC.store(tCrAcc.load().to(io_dtype))                    # Convert dtype
    cute.autovec_copy(tCrC, tDgC[None, None, i])              # Reg → GMEM
```

### 6.6 Deallocate TMEM

```python
pipeline.sync(barrier_id=1)
tmem.free(tmem_ptr)
```

---

## 7. Key Performance Notes

### 7.1 Pipeline Stages

| Stages | Latency Hiding | Use Case |
|--------|---------------|----------|
| 1 | Minimal | Simple kernels, small problem sizes |
| 4 | Good | Typical high-performance kernels |
| More | Higher | Large problem sizes |

### 7.2 Tensor Shape Notation

Use "My Notation" to understand data layout:

| Term | Meaning |
|------|---------|
| `per_mma_atom` | Elements within one MMA instruction |
| `per_mma_tile` | Tile in the MMA grid |
| `per_tma_atom` | Elements within one TMA copy |
| `per_tma_tile` | Tile for TMA operations |
| `per_wave` | Pipeline stages |
| `per_tide` | Full K loop iterations |
| `per_tmem_atom` | Elements within one TMEM copy |
| `per_tmem_tile` | Tile for TMEM operations |

### 7.3 Common Patterns

1. **SMEM tensor**: Single descriptor per block, shape `(1, 1, K_tiles, stages)`
2. **MMA fragment**: Single descriptor, shape `(1, 1, K_tiles, stages)`
3. **Single-stage**: Always use index 0 for SMEM, `k_tile_idx` for GMEM
4. **Multi-stage**: Use `phase` to toggle between buffers

### 7.4 Synchronization Rules

1. **Always use `elect_one()`** for arrive operations
2. **Use `tcgen05.commit`** for MMA→producer signaling (empty mbarrier)
3. **Use `mbarrier_arrive_and_expect_tx`** for TMA→consumer signaling (full mbarrier)
4. **Wait phase** = `1 - phase` (opposite of what was just arrived)

### 7.5 TMEM vs Register

- **TMEM**: High-capacity accumulator storage (512 columns typical)
- **Register**: Smaller, faster access for epilogue conversion
- **Epilogue pattern**: TMEM → Register (Float32) → Convert → GMEM (Float16)

---

## Quick Reference: Kernel Structure

```python
@cute.kernel
def kernel(tiled_mma, tma_atom_a, mA, tma_atom_b, mB, mC, a_smem_layout, b_smem_layout):
    # 1. Get thread/warp/block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    # 2. Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(...)
    sB = smem.allocate_tensor(...)
    
    # 3. Allocate TMEM
    tmem = utils.TmemAllocator(...)
    tmem.allocate(num_tmem_cols)
    
    # 4. Prefetch TMA
    cpasync.prefetch_descriptor(tma_atom_a/b)
    
    # 5. Initialize mbarriers
    ab_mbar_full = storage.ab_mbar_ptr.data_ptr()
    ab_mbar_empty = storage.ab_mbar_ptr.data_ptr() + 1
    if warp_idx == 0:
        mbarrier_init(ab_mbar_full, 1)
        mbarrier_init(ab_mbar_empty, 1)
    mbarrier_init_fence()
    sync_threads()
    
    # 6. Partition tensors
    gA = local_tile(mA, tiler, coord, proj=(1, None, 1))
    gB = local_tile(mB, tiler, coord, proj=(None, 1, 1))
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    tAsA, tAgA = tma_partition(...)
    tBsB, tBgB = tma_partition(...)
    
    # 7. Get TMEM pointer
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = make_tensor(tmem_ptr, tCtAcc.layout)
    
    # 8. Setup epilogue
    tCtAcc_epi = zipped_divide(tCtAcc, epi_tiler)
    tDtC = tmem_thr_copy.partition_S(...)
    tDgC = tmem_thr_copy.partition_D(...)
    
    # 9. Main loop
    phase = 1
    for k_tile_idx in range(num_k_tiles):
        mbarrier_wait(ab_mbar_empty, phase)
        copy(tma_atom_a, tAgA[(None, k_tile_idx)], tAsA[(None, 0)], tma_bar_ptr=ab_mbar_full)
        with elect_one(): mbarrier_arrive_and_expect_tx(ab_mbar_full, bytes)
        
        mbarrier_wait(ab_mbar_full, 1 - phase)
        for k_block in range(num_k_blocks):
            gemm(tiled_mma, tCtAcc, tCrA[kb], tCrB[kb], tCtAcc)
        with elect_one(): tcgen05.commit(ab_mbar_empty)
        
        phase = 1 - phase
    
    # 10. Signal done
    with elect_one(): tcgen05.commit(acc_mbar_ptr)
    
    # 11. Epilogue
    mbarrier_wait(acc_mbar_ptr, phase=0)
    for i in range(subtile_cnt):
        copy(tmem_tiled_copy, tDtC[None,None,i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        autovec_copy(tCrC, tDgC[None,None,i])
    
    # 12. Cleanup
    sync(barrier_id=1)
    tmem.free(tmem_ptr)
```
