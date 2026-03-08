import time
import torch
import torch.nn.functional as F
from triton.testing import do_bench
try:
    import teraxlang
    import triton
    HAS_TXL = True
    from triton.tools.tensor_descriptor import TensorDescriptor
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    print('TXL')
except Exception as e:
    print(e)

    class txl:

        class Config:

            def __init__(self, config, num_stages=1, num_warps=1, num_warpgroups=1, pre_hook=None):
                pass

        @staticmethod
        def jit(use_txl=False, diff_mode='ttir', diff_select=-1, log_dir=''):

            def decorator(func):
                return func
            return decorator

        @staticmethod
        def autotune(configs=[], key=''):

            def decorator(func):
                return func
            return decorator
    HAS_TXL = False
    DEVICE = torch.device('cuda:0')
    print('No txl')
import cutlass
import cutlass.torch as cutlass_torch
try:
    from quack.cross_entropy import _cross_entropy, cross_entropy
    from quack.softmax import softmax
    print('Quack detected')
    HAS_QUACK = True
except Exception as e:
    print('No quack')
    print(e)
    HAS_QUACK = False
import math
import numpy as np
import triton
import triton.language as tl
LOG2_E = 1.0 / math.log(2.0)
import os
print(os.getpid())

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret
'\nTotally triton\n'

@txl.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)
        cur_max_scaled = tl.max(row, axis=0) * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

@txl.jit
def cluster_softmax_small_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr, num_warps: tl.constexpr, num_ctas: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    lid = txl.lane_id()
    tid = txl.tid(0)
    rid = txl.cta_rank()
    max_smem_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    max_smem = txl.get_buffer(max_smem_buf, 0)
    sum_smem_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    sum_smem = txl.get_buffer(sum_smem_buf, 0)
    max_smem1_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    max_smem1 = txl.get_buffer(max_smem1_buf, 0)
    sum_smem1_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    sum_smem1 = txl.get_buffer(sum_smem1_buf, 0)
    smem_buf = txl.smem_alloc([1 * 2], dtype=tl.float32)
    scalar_smem = txl.get_buffer(smem_buf, 0)
    mbar_buf = txl.mbar_alloc(1)
    mbar = txl.get_buffer(mbar_buf, 0)
    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0], ctas_per_cga=[2], cta_split_num=[2], cta_order=[0])
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps], warps_per_cta=[num_warps], order=[0], ctas_per_cga=[2], cta_split_num=[2], cta_order=[0])
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)
        cur_max = txl.warp_max(row, axis=0)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)
        cur_denom = txl.warp_sum(numerator, axis=0)
        txl.frag_smem_store(max_smem, cur_max, layout_warp_reduce)
        txl.frag_smem_store(sum_smem, cur_denom, layout_warp_reduce)
        txl.mbar_expect(mbar, num_warps * 4 * 2)
        if rid == 0:
            txl.frag_smem_store(max_smem1, cur_max, layout_warp_reduce, None, 1, mbar)
            txl.frag_smem_store(sum_smem1, cur_denom, layout_warp_reduce, None, 1, mbar)
        else:
            txl.frag_smem_store(max_smem1, cur_max, layout_warp_reduce, None, 0, mbar)
            txl.frag_smem_store(sum_smem1, cur_denom, layout_warp_reduce, None, 0, mbar)
        txl.mbar_wait(mbar, 0)
        warp_max = txl.frag_smem_load(max_smem, [128], layout_all_reduce, -float('inf'))
        warp_denom = txl.frag_smem_load(sum_smem, [128], layout_all_reduce, 0.0)
        warp_max1 = txl.frag_smem_load(max_smem1, [128], layout_all_reduce, -float('inf'))
        warp_denom1 = txl.frag_smem_load(sum_smem1, [128], layout_all_reduce, 0.0)
        all_max = txl.warp_max(warp_max, axis=0)
        all_max1 = txl.warp_max(warp_max1, axis=0)
        all_max = tl.maximum(all_max, all_max1)
        warp_max_minus_max = warp_max - all_max
        sum_scale = tl.exp(warp_max_minus_max)
        warp_denom *= sum_scale
        all_denom = txl.warp_sum(warp_denom, axis=0)
        warp_max_minus_max1 = warp_max1 - all_max
        sum_scale1 = tl.exp(warp_max_minus_max1)
        warp_denom1 *= sum_scale1
        all_denom += txl.warp_sum(warp_denom1, axis=0)
        numerator *= tl.exp(cur_max - all_max)
        softmax_output = numerator / all_denom
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

@txl.jit
def cluster_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr, num_warps: tl.constexpr, num_ctas: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    lid = txl.lane_id()
    tid = txl.tid(0)
    rid = txl.cta_rank()
    max_smem_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    max_smem = txl.get_buffer(max_smem_buf, 0)
    sum_smem_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    sum_smem = txl.get_buffer(sum_smem_buf, 0)
    max_smem1_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    max_smem1 = txl.get_buffer(max_smem1_buf, 0)
    sum_smem1_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    sum_smem1 = txl.get_buffer(sum_smem1_buf, 0)
    smem_buf = txl.smem_alloc([1 * 2], dtype=tl.float32)
    scalar_smem = txl.get_buffer(smem_buf, 0)
    mbar_buf = txl.mbar_alloc(1)
    mbar = txl.get_buffer(mbar_buf, 0)
    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0], ctas_per_cga=[2], cta_split_num=[2], cta_order=[0])
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps], warps_per_cta=[num_warps], order=[0], ctas_per_cga=[2], cta_split_num=[2], cta_order=[0])
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)
        cur_max = txl.warp_max(row, axis=0)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)
        cur_denom = txl.warp_sum(numerator, axis=0)
        txl.frag_smem_store(max_smem, cur_max, layout_warp_reduce)
        txl.frag_smem_store(sum_smem, cur_denom, layout_warp_reduce)
        txl.mbar_expect(mbar, num_warps * 4 * 2)
        if rid == 0:
            txl.frag_smem_store(max_smem1, cur_max, layout_warp_reduce, None, 1, mbar)
            txl.frag_smem_store(sum_smem1, cur_denom, layout_warp_reduce, None, 1, mbar)
        else:
            txl.frag_smem_store(max_smem1, cur_max, layout_warp_reduce, None, 0, mbar)
            txl.frag_smem_store(sum_smem1, cur_denom, layout_warp_reduce, None, 0, mbar)
        txl.mbar_wait(mbar, 0)
        warp_max = txl.frag_smem_load(max_smem, [128], layout_all_reduce, -float('inf'))
        warp_denom = txl.frag_smem_load(sum_smem, [128], layout_all_reduce, 0.0)
        warp_max1 = txl.frag_smem_load(max_smem1, [128], layout_all_reduce, -float('inf'))
        warp_denom1 = txl.frag_smem_load(sum_smem1, [128], layout_all_reduce, 0.0)
        all_max = txl.warp_max(warp_max, axis=0)
        all_max1 = txl.warp_max(warp_max1, axis=0)
        all_max = tl.maximum(all_max, all_max1)
        warp_max_minus_max = warp_max - all_max
        sum_scale = tl.exp(warp_max_minus_max)
        warp_denom *= sum_scale
        all_denom = txl.warp_sum(warp_denom, axis=0)
        warp_max_minus_max1 = warp_max1 - all_max
        sum_scale1 = tl.exp(warp_max_minus_max1)
        warp_denom1 *= sum_scale1
        all_denom += txl.warp_sum(warp_denom1, axis=0)
        numerator *= tl.exp(cur_max - all_max)
        softmax_output = numerator / all_denom
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
'\nTest\n'

@txl.jit(diff_mode='ttgir')
def reduce_test(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr, num_warps: tl.constexpr):
    x = tl.arange(0, BLOCK_SIZE)
    y = tl.sum(x, axis=0)
    if txl.thread0():
        txl.print(y)

@txl.jit
def frag_smem_store_test(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr, num_warps: tl.constexpr, num_ctas: tl.constexpr):
    rid = txl.cta_rank()
    max_smem1_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    max_smem1 = txl.get_buffer(max_smem1_buf, 0)
    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0], ctas_per_cga=[2], cta_split_num=[2], cta_order=[0])
    cur_max = tl.sum(tl.zeros([BLOCK_SIZE], dtype=tl.float32))
    if rid == 0:
        txl.frag_smem_store(max_smem1, cur_max, layout_warp_reduce, 1)
    else:
        txl.frag_smem_store(max_smem1, cur_max, layout_warp_reduce, 0)

@txl.jit
def frag_smem_store_test2(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr, num_warps: tl.constexpr, num_ctas: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    lid = txl.lane_id()
    tid = txl.tid(0)
    rid = txl.cta_rank()
    max_smem_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    max_smem = txl.get_buffer(max_smem_buf, 0)
    sum_smem_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    sum_smem = txl.get_buffer(sum_smem_buf, 0)
    max_smem1_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    max_smem1 = txl.get_buffer(max_smem1_buf, 0)
    sum_smem1_buf = txl.smem_alloc([num_warps * 2], dtype=tl.float32)
    sum_smem1 = txl.get_buffer(sum_smem1_buf, 0)
    smem_buf = txl.smem_alloc([1 * 2], dtype=tl.float32)
    scalar_smem = txl.get_buffer(smem_buf, 0)
    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0], ctas_per_cga=[2], cta_split_num=[2], cta_order=[0])
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps * 2], warps_per_cta=[num_warps], order=[0], ctas_per_cga=[2], cta_split_num=[2], cta_order=[0])
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)
        cur_max = txl.warp_max(row, axis=0)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)
        cur_denom = txl.warp_sum(numerator, axis=0)
        if rid == 0:
            txl.frag_smem_store(max_smem1, cur_max, layout_warp_reduce, 1)
            txl.frag_smem_store(sum_smem1, cur_denom, layout_warp_reduce, 1)
        else:
            txl.frag_smem_store(max_smem1, cur_max, layout_warp_reduce, 0)
            txl.frag_smem_store(sum_smem1, cur_denom, layout_warp_reduce, 0)
'\nTXL reimpl triton\n'

@txl.jit
def softmax_kernel_txl2(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr, num_warps: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    tid = txl.tid(0)
    smem_buf = txl.smem_alloc([num_warps], dtype=tl.float32)
    reduction_smem = txl.get_buffer(smem_buf, 0)
    smem_buf = txl.smem_alloc([1], dtype=tl.float32)
    scalar_smem = txl.get_buffer(smem_buf, 0)
    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0])
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps], warps_per_cta=[1], order=[0])
    layout_all_reduce2: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[1], order=[0])
    layout_sum: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0])
    layout_full: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0])
    layout_all: tl.constexpr = txl.BlockedLayout(size_per_thread=[8], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0])
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)
        cur_max = txl.warp_max(row, axis=0)
        txl.frag_smem_store(reduction_smem, cur_max, layout_warp_reduce)
        cur_max = txl.frag_smem_load(reduction_smem, [128], layout_all_reduce, -float('inf'))
        all_max = txl.warp_max(cur_max, axis=0)
        if tid == 0:
            txl.frag_smem_store(scalar_smem, all_max, layout_sum)
        cur_max = txl.smem_load(scalar_smem, layout_all)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)
        cur_denom = txl.warp_sum(numerator, axis=0)
        txl.frag_smem_store(reduction_smem, cur_denom, layout_warp_reduce)
        cur_denom = txl.frag_smem_load(reduction_smem, [128], layout_all_reduce, 0.0)
        all_denom = txl.warp_sum(cur_denom, axis=0)
        if tid < num_warps:
            txl.frag_smem_store(scalar_smem, all_denom, layout_sum)
        denominator = txl.smem_load(scalar_smem, layout_all)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
'\nTXL reiml triton\n'

@txl.jit
def softmax_kernel_txl(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr, num_warps: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    tid = txl.tid(0)
    smem_buf = txl.smem_alloc([num_warps], dtype=tl.float32)
    reduction_smem = txl.get_buffer(smem_buf, 0)
    smem_buf = txl.smem_alloc([1], dtype=tl.float32)
    scalar_smem = txl.get_buffer(smem_buf, 0)
    layout_warp_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[num_warps], order=[0])
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps], warps_per_cta=[1], order=[0])
    layout_all_reduce2: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[1], order=[0])
    layout_sum: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0])
    layout_full: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0])
    layout_all: tl.constexpr = txl.BlockedLayout(size_per_thread=[8], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0])
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)
        cur_max = txl.warp_max(row, axis=0)
        txl.frag_smem_store(reduction_smem, cur_max, layout_warp_reduce)
        cur_max = txl.frag_smem_load(reduction_smem, [128], layout_all_reduce, -float('inf'))
        all_max = txl.warp_max(cur_max, axis=0)
        if tid == 0:
            txl.frag_smem_store(scalar_smem, all_max, layout_sum)
        cur_max = txl.smem_load(scalar_smem, layout_all)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)
        cur_denom = txl.warp_sum(numerator, axis=0)
        txl.frag_smem_store(reduction_smem, cur_denom, layout_warp_reduce)
        cur_denom = txl.frag_smem_load(reduction_smem, [BLOCK_SIZE], layout_all_reduce, 0.0)
        all_denom = txl.warp_sum(cur_denom, axis=0)
        if tid < num_warps:
            txl.frag_smem_store(scalar_smem, all_denom, layout_sum)
        denominator = txl.smem_load(scalar_smem, layout_all)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

@txl.jit
def online_softmax_kernel_txl(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, LOG2_E: tl.constexpr, num_stages: tl.constexpr, num_warps: tl.constexpr):
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
    layout_all_reduce: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[num_warps], warps_per_cta=[num_warps], order=[0])
    layout_sum: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[num_warps], order=[0])
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row = tl.cast(row, tl.float32)
        cur_max = txl.warp_max(row, axis=0)
        cur_max_scaled = cur_max * LOG2_E
        numerator = tl.exp2(row * LOG2_E - cur_max_scaled)
        cur_denom = txl.warp_sum(numerator, axis=0)
        txl.frag_smem_store(max_smem, cur_max, layout_warp_reduce)
        txl.frag_smem_store(sum_smem, cur_denom, layout_warp_reduce)
        warp_max = txl.frag_smem_load(max_smem, [128], layout_all_reduce, -float('inf'))
        warp_denom = txl.frag_smem_load(sum_smem, [128], layout_all_reduce, 0.0)
        all_max = txl.warp_max(warp_max, axis=0)
        warp_max_minus_max = warp_max - all_max
        sum_scale = tl.exp(warp_max_minus_max)
        warp_denom *= sum_scale
        all_denom = txl.warp_sum(warp_denom, axis=0)
        numerator *= tl.exp(cur_max - all_max)
        softmax_output = numerator / all_denom
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def fused_softmax(x, num_programs, LOG2_E, bench: bool=False):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    num_stages = 1
    if bench:
        y = torch.empty_like(x)
    else:
        y = torch.zeros_like(x)
    num_programs = n_rows
    cluster_softmax_kernel[num_programs, 1, 1](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, LOG2_E, num_stages, num_warps=num_warps, num_ctas=2)
    return y
DEVICE = triton.runtime.driver.active.get_active_torch_device()

def meta_fn(kernel_fn, x):
    properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties['multiprocessor_count']
    NUM_REGS = properties['max_num_regs']
    SIZE_SMEM = properties['max_shared_mem']
    WARP_SIZE = properties['warpSize']
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    LOG2_E = math.log2(math.e)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)
    kernel = kernel_fn.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, LOG2_E=LOG2_E, num_stages=num_stages, num_warps=num_warps, grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    n_spills = kernel.n_spills
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)
    print('META:')
    print(f'NUM_SM: {NUM_SM}')
    print(f'NUM_REGS: {NUM_REGS}')
    print(f'SIZE_SMEM: {SIZE_SMEM}')
    print(f'WARP_SIZE: {WARP_SIZE}')
    print()
    print(f'x.shape: {x.shape}')
    print(f'BLOCK_SIZE: {BLOCK_SIZE}')
    print(f'num_warps: {num_warps}')
    print(f'num_stages: {num_stages}')
    print()
    print(f'n_regs: {n_regs}')
    print(f'n_spills: {n_spills}')
    print(f'size_smem: {size_smem}')
    print()
    print(f'occupancy: {occupancy}')
    print(f'num_programs: {num_programs}')
    return (num_programs, BLOCK_SIZE, num_warps, num_stages)

def test_softmax(dump_dir=None, M=32 * 1024, N=32 * 1024):
    KERNEL = 'SM'
    dtype = cutlass.BFloat16
    warmup_iterations = 10
    iterations = 100
    torch_dtype = cutlass_torch.dtype(dtype)
    device = 'cuda'
    x32 = 0.1 * torch.randn(M, N, device=device, dtype=torch.float32)
    x = x32.to(dtype=torch_dtype)
    print(f'x.shape: {x.shape}')
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)
    if KERNEL == 'CE':
        loss = _cross_entropy(x, target)
        compiled_func_ref = torch.compile(lambda x, target: F.cross_entropy(x, target, reduction='none'))
        quack_fn = lambda: _cross_entropy(x, target)
        fn_mem_bytes = lambda: (M * N + M + M) * dtype.width // 8
    elif KERNEL == 'SM':
        LOG2_E = math.log2(math.e)
        compiled_func_ref = lambda x, target: F.softmax(x, dim=-1)
        if HAS_QUACK:
            quack_fn = lambda: softmax(x)
        num_programs = 264
        if HAS_TXL:
            txl_fn = lambda x, bench: fused_softmax(x, num_programs, LOG2_E, bench)
        fn_mem_bytes = lambda: 2 * x.numel() * dtype.width // 8
    from triton import knobs
    knobs.autotuning.print = True
    knobs.compilation.always_compile = True
    if dump_dir:
        knobs.compilation.dump_ir = True
        knobs.cache.dump_dir = dump_dir
    if HAS_QUACK:
        quack_out = quack_fn().to(torch.float32).cpu().numpy()
    if HAS_TXL:
        txl_out = txl_fn(x, False).to(torch.float32).cpu().numpy()
    a = x32.cpu().numpy()
    a_max = a.max(axis=1, keepdims=True)
    ref = np.exp(a - a_max)
    ref = ref / ref.sum(axis=1, keepdims=True)
    torch_out = compiled_func_ref(x, target).to(torch.float32).cpu().numpy()
    outs = []
    outs.append(('torch', torch_out))
    if HAS_QUACK:
        outs.append(('quack', quack_out))
    if HAS_TXL:
        outs.append(('txl', txl_out))
    for name, out in outs:
        max_rel_err = np.max(np.abs(out - ref) / (ref + 1e-20))
        print('max relative error:', max_rel_err)
        print('sum per row -1 (should be 0):', np.max(np.abs(out.sum(axis=1) - 1.0)))
    mem_bytes = fn_mem_bytes()
    fn = lambda: compiled_func_ref(x, target)
    for _ in range(5):
        fn()
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round(mem_bytes / (avg_time / 1000) / 1000000000.0)
    print(f'Torch kernel execution time: {avg_time:.4f} ms')
    print(f'Torch mem throughput: {mem_bw_ref:.2f} GB/s')
    if HAS_QUACK:
        time.sleep(0.5)
        fn = quack_fn
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw = round(mem_bytes / (avg_time / 1000) / 1000000000.0)
        print(f'Quack execution time: {avg_time:.4f} ms')
        print(f'Quack mem throughput: {mem_bw:.2f} GB/s')
    if HAS_TXL:
        fn = lambda: txl_fn(x, True)
        for _ in range(5):
            fn()
        time.sleep(0.5)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw_ref = round(mem_bytes / (avg_time / 1000) / 1000000000.0)
        print(f'Triton kernel execution time: {avg_time:.4f} ms')
        print(f'Triton mem throughput: {mem_bw_ref:.2f} GB/s')
if __name__ == '__main__':
    test_softmax()