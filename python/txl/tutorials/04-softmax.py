import time

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from quack.cross_entropy import _cross_entropy, cross_entropy
from quack.softmax import softmax


## Triton

import math
import numpy as np
import triton
import triton.language as tl

# log2(e) constant (can be passed in instead)
LOG2_E = 1.0 / math.log(2.0)  # ~1.4426950408889634

@triton.jit
def triton_row_softmax(
    X_ptr,            # pointer to input float32 matrix
    Y_ptr,            # pointer to output float32 matrix
    N_COLS,           # number of columns in each row
    stride_row_X,     # row stride (elements) for input
    stride_col_X,     # column stride (elements) for input (usually 1)
    stride_row_Y,     # row stride for output
    stride_col_Y,     # col stride for output
    log2_e: tl.constexpr,           # constant: log2(e)
    BLOCK_SIZE: tl.constexpr,  # number of columns handled by a single block's vector
):
    """
    A Triton kernel where each program (program_id(0)) processes a single row.
    BLOCK_SIZE is a compile-time constant controlling how many columns we process per vectorized load.
    """

    row = tl.program_id(0)          # each program instance handles one row
    # offsets inside the row for the current vectorized block
    offs = tl.arange(0, BLOCK_SIZE)

    # pointer to beginning of this row in X and Y
    row_ptr_X = X_ptr + row * stride_row_X
    row_ptr_Y = Y_ptr + row * stride_row_Y

    # ---------- 1) compute max over the row (numerical stability) ----------
    # initialize max with -inf
    #minus_inf = tl.constant(-1e30, dtype=tl.float32)
    minus_inf = -1e30
    cur_max = minus_inf + tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # iterate over the row in chunks of BLOCK_SIZE
    col_start = 0
    while col_start < N_COLS:
        idx = col_start + offs                     # absolute column indices
        mask = idx < N_COLS                        # valid columns mask

        # address for loads: row_ptr_X + idx * stride_col_X
        load_addr = row_ptr_X + idx * stride_col_X
        x = tl.load(load_addr, mask=mask, other=minus_inf)

        # elementwise max across chunks; cur_max is a vector of length BLOCK_SIZE,
        # but we need a scalar max. We'll reduce cur_max later with tl.max(..., axis=0).
        cur_max = tl.maximum(cur_max, x)
        col_start += BLOCK_SIZE

    # reduce vector cur_max to single scalar m
    m = tl.max(cur_max, axis=0)   # scalar

    # ---------- 2) compute sum of exp2((x - m) * log2_e) ----------
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # will reduce later

    col_start = 0
    while col_start < N_COLS:
        idx = col_start + offs
        mask = idx < N_COLS

        load_addr = row_ptr_X + idx * stride_col_X
        x = tl.load(load_addr, mask=mask, other=0.0)

        # compute exp2((x - m) * log2_e) (elementwise)
        shifted = x - m
        val = tl.exp2(shifted * log2_e)   # vector
        # masked store into sum accumulator vector
        val = tl.where(mask, val, tl.zeros([BLOCK_SIZE], dtype=tl.float32))
        sum_exp = sum_exp + val
        col_start += BLOCK_SIZE

    # reduce sum_exp vector to scalar
    sum_exp_scalar = tl.sum(sum_exp, axis=0)  # scalar

    # compute reciprocal once
    inv_sum = 1.0 / sum_exp_scalar

    # ---------- 3) write final softmax values ----------
    col_start = 0
    while col_start < N_COLS:
        idx = col_start + offs
        mask = idx < N_COLS

        load_addr = row_ptr_X + idx * stride_col_X
        x = tl.load(load_addr, mask=mask, other=0.0)

        out = tl.exp2((x - m) * log2_e) * inv_sum

        store_addr = row_ptr_Y + idx * stride_col_Y
        tl.store(store_addr, out, mask=mask)
        col_start += BLOCK_SIZE


# Python wrapper
def softmax_triton(x: np.ndarray, block_size: int = 128, num_warps: int = 4):
    """
    Compute row-wise softmax of 2D numpy array x using Triton kernel.
    - block_size should be tuned for your hardware (common: 64, 128, 256).
    - num_warps is a Triton tuning parameter (common: 4).
    """

    assert x.dtype == torch.float32, "Only float32 supported in this example"
    assert x.ndim == 2, "Input must be 2D (rows x cols)"
    rows, cols = x.shape

    # allocate output
    y = torch.empty_like(x)

    # row-major strides in elements
    stride_row_X = x.stride()[0]
    stride_col_X = x.stride()[1]
    stride_row_Y = y.stride()[0]
    stride_col_Y = y.stride()[1]

    # Launch Triton kernel: each program_id(0) -> one row -> grid=(rows,)
    grid = (rows,)

    # note: pass BLOCK_SIZE as a python integer into the kernel as a compile-time constexpr
    triton_row_softmax[grid](
        x,                           # X_ptr (numpy supports buffer protocol)
        y,                           # Y_ptr
        (cols),              # N_COLS
        (stride_row_X),      # stride_row_X
        (stride_col_X),      # stride_col_X
        (stride_row_Y),      # stride_row_Y
        (stride_col_Y),      # stride_col_Y
        (LOG2_E),          # log2_e constant
        block_size,                  # BLOCK_SIZE as constexpr
        num_warps=num_warps
    )
    return y

def test_softmax():
    #M = 32*1024
    #N = 32*1024
    M = 8192
    N = 16384
    KERNEL="SM"

    dtype = cutlass.BFloat16
    warmup_iterations = 10
    iterations = 100

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x32 = 0.1 * torch.randn(M, N, device=device, dtype=torch.float32)
    x = x32.to(dtype=torch_dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)

    if KERNEL == "CE":
        # CE
        loss = _cross_entropy(x, target)
        compiled_func_ref = torch.compile(lambda x, target: F.cross_entropy(x, target, reduction='none'))
        fn = lambda: _cross_entropy(x, target)
        fn_mem_bytes = lambda: (M * N + M + M) * dtype.width // 8

    elif KERNEL == "SM":
        # Softmax
        out = softmax(x)
        compiled_func_ref = torch.compile(lambda x, target: F.softmax(x, dim=-1))
        fn = lambda: softmax(x)
        tri_fn = lambda x: softmax_triton(x)
        fn_mem_bytes = lambda: 2 * x.numel() * dtype.width // 8

    ################################################
    # Validate
    ################################################

    # NumPy reference
    a = x32.cpu().numpy()
    a_max = a.max(axis=1, keepdims=True)
    ref = np.exp(a - a_max)
    ref = ref / ref.sum(axis=1, keepdims=True)

    torch_out = compiled_func_ref(x, target).to(torch.float32).cpu().numpy()
    quack_out = fn().to(torch.float32).cpu().numpy()
    tri_out = tri_fn(x32).to(torch.float32).cpu().numpy()

    # relative error
    outs = []
    outs.append(torch_out)
    outs.append(quack_out)
    outs.append(tri_out)
    names = ['torch', 'quack', 'triton']

    for name, out in zip(names, outs):
        max_rel_err = np.max(np.abs(out - ref) / (ref + 1e-20))
        print("max relative error:", max_rel_err)
        print("sum per row -1 (should be 0):", np.max(np.abs(out.sum(axis=1) - 1.0)))


    ################################################
    # Bench
    ################################################

    mem_bytes = fn_mem_bytes()

    fn = lambda: compiled_func_ref(x, target)
    for _ in range(5): fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Torch kernel execution time: {avg_time:.4f} ms")
    print(f"Torch mem throughput: {mem_bw_ref:.2f} GB/s")

    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    # Memory bandwidth calculation: read x (M*N elements) + read target (M elements) + write loss (M elements)
    mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Quack execution time: {avg_time:.4f} ms")
    print(f"Quack mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: tri_fn(x32)
    for _ in range(5): fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Triton kernel execution time: {avg_time:.4f} ms")
    print(f"Triton mem throughput: {mem_bw_ref:.2f} GB/s")
