# Core API

TeraXLang extends Triton's core language with additional primitives for shared memory operations, tensor memory accelerator (TMA), and warp-level synchronization.

## Thread & Block Identity

```python
# Thread and block dimensions
tid()       # Thread ID within block
tdim()      # Block size
bid()       # Block ID
bdim()      # Block dimensions

# Warp-level
thread0()       # First thread in block
wg_thread0()    # First thread in warpgroup
warp_id()      # Warp ID within block
warpgroup_id() # Warpgroup ID
lane_id()      # Lane ID within warp
```

## Shared Memory Operations

```python
# Shared memory allocation
smem_alloc()      # Allocate shared memory
smem_load()        # Load from shared memory
smem_store()       # Store to shared memory
smem_index()       # Index into shared memory
smem_slice()       # Slice shared memory
smem_trans()       # Transpose shared memory
smem_reshape()     # Reshape shared memory

# Fragment operations
frag_smem_load()   # Load with fragment layout
frag_smem_store()  # Store with fragment layout
```

## Tensor Memory Accelerator (TMA)

```python
# TMA operations
tma_load()         # TMA load from global to shared
tma_store()        # TMA store from shared to global
tma_gather()       # TMA gather operation
tma_load_wait()    # Wait for TMA load
tma_store_wait()   # Wait for TMA store
```

## Mailbox (Mbar) Synchronization

```python
# Mailbox operations
mbar_alloc()      # Allocate mailbox
mbar_expect()     # Expect mailbox value
mbar_wait()       # Wait for mailbox
mbar_arrive()    # Arrive at mailbox
```

## Warp-Level Primitives

```python
# Warp-level reductions
warp_max()        # Warp-level max
warp_sum()        # Warp-level sum
```

## Memory & Synchronization

```python
# Register allocation
reg_alloc()       # Allocate registers
reg_dealloc()     # Deallocate registers

# Synchronization
fence_proxy_async()  # Async fence
bar_arrive()      # Barrier arrive
bar_wait()        # Barrier wait

# Layout
relayout()        # Relayout tensor
print_layout()    # Print tensor layout
```

## Additional Math Operations

```python
# Dot product
dotx()           # Extended dot product
dot_wait()       # Wait for dot operation

# Async operations
async_load()     # Async load
async_load_wait() # Wait for async load
```

For the full Triton core API, see [Triton Language Reference](https://triton-lang.org/).
