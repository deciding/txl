import time

import torch
import torch.nn.functional as F
from triton.testing import do_bench
import txl

import cutlass
import cutlass.torch as cutlass_torch

try:
    from quack.cross_entropy import _cross_entropy, cross_entropy
    from quack.softmax import softmax
    print("Quack detected")
    HAS_QUACK=True
except Exception as e:
    print("No quack")
    print(e)
    HAS_QUACK=False


## Triton

import math
import numpy as np
import triton
import triton.language as tl

# log2(e) constant (can be passed in instead)
LOG2_E = 1.0 / math.log(2.0)  # ~1.4426950408889634

import os;print(os.getpid())


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

"""
Totally triton
"""
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   LOG2_E: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)

        ## Subtract maximum for numerical stability
        #row_minus_max = row - tl.max(row, axis=0)
        ## Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        #numerator = tl.exp(row_minus_max)

        cur_max_scaled = tl.max(row, axis=0) * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)

        denominator = tl.sum(numerator, axis=0)

        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

@triton.jit
def cluster_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   LOG2_E: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_offsets = tl.reshape(col_offsets, (2, BLOCK_SIZE // 2))

        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)

        ## Subtract maximum for numerical stability
        #row_minus_max = row - tl.max(row, axis=0)
        ## Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        #numerator = tl.exp(row_minus_max)

        cur_max_scaled = tl.max(row, axis=0) * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)

        denominator = tl.sum(numerator, axis=0)

        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

"""
TXL reimpl triton
"""
@txl.jit
#@txl.jit(diff_mode='llir')
def softmax_kernel_txl2(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
        LOG2_E: tl.constexpr,
        num_stages: tl.constexpr, num_warps: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    #lid = txl.lane_id()
    tid = txl.tid(0)

    smem_buf = txl.smem_alloc([num_warps], dtype=tl.float32)
    reduction_smem = txl.get_buffer(smem_buf, 0)

    smem_buf = txl.smem_alloc([1], dtype=tl.float32)
    scalar_smem = txl.get_buffer(smem_buf, 0)

    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0])
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps], warps_per_cta=[1], order=[0])
    layout_all_reduce2: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[1], order=[0])
    layout_sum: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0]) # TODO: infer from reduced shape
    layout_full: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0]) # TODO: use default
    layout_all: tl.constexpr = txl.BlockedLayout(size_per_thread=[8], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0]) # TODO: use default

    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32) # reduce on higher precision

        # Subtract maximum for numerical stability

        # REPLACE with warp reduce
        #cur_max = tl.max(row, axis=0)
        cur_max = txl.warp_max(row, axis=0)
        txl.frag_smem_store(reduction_smem, cur_max, layout_warp_reduce) # only 1 val 1 lane, only filter lanes here
        #cur_max = txl.frag_smem_load(reduction_smem, layout_all_reduce)
        #cur_max = txl.frag_smem_load(reduction_smem, layout_all_reduce, layout_all_reduce2, -float('inf'))
        cur_max = txl.frag_smem_load(reduction_smem, [128], layout_all_reduce, -float('inf')) # fill other lanes
        all_max = txl.warp_max(cur_max, axis=0)
        #if  tid < num_warps:
        if  tid == 0:
            txl.frag_smem_store(scalar_smem, all_max, layout_sum) # only one value
        #cur_max = txl.frag_smem_load(scalar_smem, layout_full, layout_all)
        #cur_max = txl.frag_smem_load(scalar_smem, [BLOCK_SIZE], layout_full) # implicit broadcast
        #cur_max = txl.smem_load(scalar_smem, layout_full) # implicit broadcast
        cur_max = txl.smem_load(scalar_smem, layout_all) # implicit broadcast

        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        #row_minus_max = row - cur_max
        #numerator = tl.exp(row_minus_max)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)

        # REPLACE with warp reduce
        #denominator = tl.sum(numerator, axis=0)
        cur_denom = txl.warp_sum(numerator, axis=0)
        txl.frag_smem_store(reduction_smem, cur_denom, layout_warp_reduce) # 1 lane

        #cur_denom = txl.frag_smem_load(reduction_smem, layout_all_reduce, layout_full)
        cur_denom = txl.frag_smem_load(reduction_smem, [128], layout_all_reduce, 0.0) # 1 reg, 4 lanes, 1 warp
        all_denom = txl.warp_sum(cur_denom, axis=0)

        if  tid < num_warps:
            txl.frag_smem_store(scalar_smem, all_denom, layout_sum)
        #denominator = txl.frag_smem_load(scalar_smem, layout_full, layout_all)
        #denominator = txl.frag_smem_load(scalar_smem, [BLOCK_SIZE], layout_full) # implicit broadcast
        #denominator = txl.smem_load(scalar_smem, layout_full) # implicit broadcast
        denominator = txl.smem_load(scalar_smem, layout_all) # implicit broadcast

        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

"""
TXL reiml triton
"""
@txl.jit
#@txl.jit(diff_mode='llir')
def softmax_kernel_txl(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
        LOG2_E: tl.constexpr,
        num_stages: tl.constexpr, num_warps: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    #lid = txl.lane_id()
    tid = txl.tid(0)

    smem_buf = txl.smem_alloc([num_warps], dtype=tl.float32)
    reduction_smem = txl.get_buffer(smem_buf, 0)

    smem_buf = txl.smem_alloc([1], dtype=tl.float32)
    scalar_smem = txl.get_buffer(smem_buf, 0)

    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0])
    #layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[num_warps], threads_per_warp=[1], warps_per_cta=[1], order=[0])
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps], warps_per_cta=[1], order=[0])
    layout_all_reduce2: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[1], order=[0])
    layout_sum: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0]) # TODO: infer from reduced shape
    layout_full: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0]) # TODO: use default
    #layout_all: tl.constexpr = txl.BlockedLayout(size_per_thread=[BLOCK_SIZE//32//num_warps], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0]) # TODO: use default
    layout_all: tl.constexpr = txl.BlockedLayout(size_per_thread=[8], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0]) # TODO: use default

    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32) # reduce on higher precision

        # Subtract maximum for numerical stability

        # REPLACE with warp reduce
        #cur_max = tl.max(row, axis=0)
        cur_max = txl.warp_max(row, axis=0)
        txl.frag_smem_store(reduction_smem, cur_max, layout_warp_reduce) # only 1 val 1 lane, only filter lanes here
        #cur_max = txl.frag_smem_load(reduction_smem, layout_all_reduce)
        #cur_max = txl.frag_smem_load(reduction_smem, layout_all_reduce, layout_all_reduce2, -float('inf'))
        cur_max = txl.frag_smem_load(reduction_smem, [128], layout_all_reduce, -float('inf')) # fill other lanes and warps
        all_max = txl.warp_max(cur_max, axis=0)
        #if  tid < num_warps:
        if  tid == 0:
            txl.frag_smem_store(scalar_smem, all_max, layout_sum) # only one value
        #cur_max = txl.frag_smem_load(scalar_smem, layout_full, layout_all)
        #cur_max = txl.frag_smem_load(scalar_smem, [BLOCK_SIZE], layout_full) # implicit broadcast
        #cur_max = txl.smem_load(scalar_smem, layout_full) # implicit broadcast
        cur_max = txl.smem_load(scalar_smem, layout_all) # implicit broadcast

        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        #row_minus_max = row - cur_max
        #numerator = tl.exp(row_minus_max)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)

        # REPLACE with warp reduce
        #denominator = tl.sum(numerator, axis=0)
        cur_denom = txl.warp_sum(numerator, axis=0)
        txl.frag_smem_store(reduction_smem, cur_denom, layout_warp_reduce) # 1 lane
        #cur_denom = txl.frag_smem_load(reduction_smem, layout_all_reduce, layout_full)
        cur_denom = txl.frag_smem_load(reduction_smem, [BLOCK_SIZE], layout_all_reduce, 0.0)
        all_denom = txl.warp_sum(cur_denom, axis=0)
        if  tid < num_warps:
            txl.frag_smem_store(scalar_smem, all_denom, layout_sum)
        #denominator = txl.frag_smem_load(scalar_smem, layout_full, layout_all)
        #denominator = txl.frag_smem_load(scalar_smem, [BLOCK_SIZE], layout_full) # implicit broadcast
        #denominator = txl.smem_load(scalar_smem, layout_full) # implicit broadcast
        denominator = txl.smem_load(scalar_smem, layout_all) # implicit broadcast

        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

@txl.jit
def online_softmax_kernel_txl(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr,
        num_stages: tl.constexpr, num_warps: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    lid = txl.lane_id()
    tid = txl.tid(0)

    max_smem_buf = txl.smem_alloc([num_warps], dtype=tl.float32)
    max_smem = txl.get_buffer(max_smem_buf, 0)
    sum_smem_buf = txl.smem_alloc([num_warps], dtype=tl.float32)
    sum_smem = txl.get_buffer(sum_smem_buf, 0)

    smem_buf = txl.smem_alloc([1], dtype=tl.float32)
    scalar_smem = txl.get_buffer(smem_buf, 0)

    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0])
    #layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[num_warps], threads_per_warp=[1], warps_per_cta=[1], order=[0])
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps], warps_per_cta=[num_warps], order=[0])
    layout_sum: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0]) # TODO: infer from reduced shape
    #layout_full: tl.constexpr = txl.BlockedLayout(size_per_thread=[BLOCK_SIZE//32//num_warps//16], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0]) # TODO: use default

    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)

        # Subtract maximum for numerical stability

        cur_max = txl.warp_max(row, axis=0)

        #row_minus_max = row - cur_max
        #numerator = tl.exp(row_minus_max)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)

        cur_denom = txl.warp_sum(numerator, axis=0)

        txl.frag_smem_store(max_smem, cur_max, layout_warp_reduce)
        txl.frag_smem_store(sum_smem, cur_denom, layout_warp_reduce)

        #warp_max = txl.frag_smem_load(max_smem, layout_all_reduce, layout_sum, -float('inf'))
        #warp_denom = txl.frag_smem_load(sum_smem, layout_all_reduce, layout_sum, 0.0)
        warp_max = txl.frag_smem_load(max_smem, [128], layout_all_reduce, -float('inf'))
        warp_denom = txl.frag_smem_load(sum_smem, [128], layout_all_reduce, 0.0)
        #if lid >= num_warps: # shape problem
        #    warp_max = -float('inf')
        #    warp_denom = 0.0
        all_max = txl.warp_max(warp_max, axis=0) # 3rd, might have problem
        warp_max_minus_max = warp_max - all_max
        sum_scale = tl.exp(warp_max_minus_max)
        warp_denom *= sum_scale
        all_denom = txl.warp_sum(warp_denom, axis=0)

        numerator *= tl.exp(cur_max - all_max)
        softmax_output = numerator / all_denom

        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def fused_softmax(x, num_programs, LOG2_E, bench: bool=False):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    #num_stages = 4 if SIZE_SMEM > 200000 else 2
    num_stages = 1

    # Allocate output
    if bench:
        y = torch.empty_like(x) # for benchmark
    else:
        y = torch.zeros_like(x)

    ## pre-compile kernel to get register usage and compute thread occupancy.
    #kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
    #                               num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    #kernel._init_handles()
    #n_regs = kernel.n_regs
    #size_smem = kernel.metadata.shared

	#occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)

    #occupancy = min(occupancy, SIZE_SMEM // size_smem)
    #num_programs = NUM_SM * occupancy

    #num_programs = min(num_programs, n_rows)

    num_programs = n_rows

    # Create a number of persistent programs.
    #kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)

    #softmax_kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, LOG2_E, num_stages)
    #cluster_softmax_kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, LOG2_E, num_stages, num_ctas=2)
    #softmax_kernel_txl[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, LOG2_E, num_stages, num_warps=4)
    #softmax_kernel_txl2[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, LOG2_E, num_stages, num_warps=4)
    online_softmax_kernel_txl[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, LOG2_E, num_stages, num_warps=4)
    return y


################################################
# TEST
################################################

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def meta_fn(kernel_fn, x):
    properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]

    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)
    kernel = kernel_fn.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    n_spills = kernel.n_spills
    size_smem = kernel.metadata.shared

    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    print("META:")
    print(f"NUM_SM: {NUM_SM}")
    print(f"NUM_REGS: {NUM_REGS}")
    print(f"SIZE_SMEM: {SIZE_SMEM}")
    print(f"WARP_SIZE: {WARP_SIZE}")
    print()

    print(f"x.shape: {x.shape}")
    print(f"BLOCK_SIZE: {BLOCK_SIZE}")
    print(f"num_warps: {num_warps}")
    print(f"num_stages: {num_stages}")
    print()

    print(f"n_regs: {n_regs}")
    print(f"n_spills: {n_spills}")
    print(f"size_smem: {size_smem}")
    print()

    print(f"occupancy: {occupancy}")
    print(f"num_programs: {num_programs}")

    return num_programs, BLOCK_SIZE, num_warps, num_stages

def test_softmax(dump_dir=None):
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
        quack_fn = lambda: _cross_entropy(x, target)
        fn_mem_bytes = lambda: (M * N + M + M) * dtype.width // 8

    elif KERNEL == "SM":
        LOG2_E = math.log2(math.e)
        # Softmax
        compiled_func_ref = torch.compile(lambda x, target: F.softmax(x, dim=-1))
        if HAS_QUACK:
            quack_fn = lambda: softmax(x)

        #num_programs, BLOCK_SIZE, num_warps, num_stages = meta_fn(softmax_kernel, x)
        num_programs=228
        txl_fn = lambda x, bench: fused_softmax(x, num_programs, LOG2_E, bench)

        fn_mem_bytes = lambda: 2 * x.numel() * dtype.width // 8

    ################################################
    # Validate
    ################################################
    from triton import knobs
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "axis-info,tritongpu-coalesce"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-coalesce"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "ttg-utility"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"
    knobs.autotuning.print=True
    knobs.compilation.always_compile=True
    if dump_dir:
        knobs.compilation.dump_ir=True
        knobs.cache.dump_dir=dump_dir

    # NumPy reference
    a = x32.cpu().numpy()
    a_max = a.max(axis=1, keepdims=True)
    ref = np.exp(a - a_max)
    ref = ref / ref.sum(axis=1, keepdims=True)

    torch_out = compiled_func_ref(x, target).to(torch.float32).cpu().numpy()
    if HAS_QUACK:
        quack_out = quack_fn().to(torch.float32).cpu().numpy()
    txl_out = txl_fn(x, False).to(torch.float32).cpu().numpy()

    # relative error
    outs = []
    outs.append(('torch', torch_out))
    if HAS_QUACK:
        outs.append(('quack', quack_out))
    outs.append(('txl', txl_out))

    for name, out in outs:
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

    if HAS_QUACK:
        time.sleep(0.5)
        fn = quack_fn
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        # Memory bandwidth calculation: read x (M*N elements) + read target (M elements) + write loss (M elements)
        mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9)
        print(f"Quack execution time: {avg_time:.4f} ms")
        print(f"Quack mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: txl_fn(x, True)
    for _ in range(5): fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Triton kernel execution time: {avg_time:.4f} ms")
    print(f"Triton mem throughput: {mem_bw_ref:.2f} GB/s")

if __name__ == "__main__":
    #test_softmax('dump/1031sm')
    test_softmax()
