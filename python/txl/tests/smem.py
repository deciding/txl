import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import txl
import os; print(os.getpid())


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@txl.jit()
def txl_tma_kernel(
        desc_q,
        desc_o,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    #if txl.tid(0) == 0:
    #    txl.print('hello t0')
    #if txl.warp_id() == 0:
    #    txl.print('hello w0', txl.tidx.x())
    #if txl.warpgroup_id() == 0:
    #    txl.print('hello wg0', txl.tidx.x())
    qo_offset_y = 0
    q = desc_q.load([qo_offset_y, 0])
    acc = q + 1.0
    desc_o.store([qo_offset_y, 0], acc)

def test_tma_txl():
    dtype = torch.float16
    dummy_block = [128, 64]
    q = torch.randn((128, 64), dtype=dtype, device=DEVICE)
    o = torch.empty_like(q)

    desc_q = TensorDescriptor(q, shape=[128, 64], strides=[64, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[128, 64], strides=[64, 1], block_shape=dummy_block)

    grid = lambda meta: (1,)
    txl_tma_kernel[grid](desc_q, desc_o, BLOCK_SIZE_M=128, BLOCK_SIZE_N=64)
    print(q)
    print(o)

@txl.jit
#@txl.jit(diff_mode="ttgir")
def txl_smem_kernel(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + offsets)

    buf = txl.smem_alloc([BLOCK_SIZE], dtype=tl.float32)
    view = txl.get_buffer(buf, 0)
    x = x + 1

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[1], order=[0])

    txl.smem_store(view, x)
    a = txl.smem_load(view, layout_a)

    tl.store(out_ptr + offsets, a)

def test_smem_txl():
    # Example usage
    BLOCK_SIZE = 32
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel[(1,)](x, y, BLOCK_SIZE=32, num_warps=1)

    print("input :", x)
    print("output:", y)

@txl.jit
def txl_smem_kernel2(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + offsets)

    buf = txl.smem_alloc([4], dtype=tl.float32)
    view = txl.get_buffer(buf, 0)
    x = x + 1

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])

    layout_b: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[4], order=[0])

    layout_c: tl.constexpr = txl.BlockedLayout(size_per_thread=[4], threads_per_warp=[1], warps_per_cta=[1], order=[0])

    txl.frag_smem_store(view, x, layout_b) # TODO: make layout_a not necessary

    if txl.lane_id() == 0:
        b = txl.frag_smem_load(view, layout_b)
        txl.print("b:", b)

    if txl.tid(0) == 0:
        c = txl.frag_smem_load(view, layout_c)
        sum_c = txl.sum(c)
        txl.print("c:", c)
        txl.print("sum_c", sum_c)

def test_smem_txl2():
    # Example usage
    BLOCK_SIZE = 128
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel2[(1,)](x, y, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    print("input :", x)
    print("output:", y)

@txl.jit
def txl_smem_kernel3(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    layout_sum_c: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0])

    buf1 = txl.smem_alloc([1], dtype=tl.float32)
    view1 = txl.get_buffer(buf1, 0)

    if txl.tid(0) == 0:
        sum_c = tl.full((1,), 3.0, tl.float32)
        txl.frag_smem_store(view1, sum_c, layout_sum_c) # TODO: make layout_a not necessary

    a = txl.frag_smem_load(view1, layout_a)
    if txl.tid(0) == 33:
        txl.print("a:", a)

def test_smem_txl3():
    # Example usage
    BLOCK_SIZE = 128
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel3[(1,)](x, y, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    print("input :", x)
    print("output:", y)

@txl.jit
#@txl.jit(src_file='dump/smem4/txl_smem_kernel4.ptx')
def txl_smem_kernel4(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + offsets)

    buf = txl.smem_alloc([4], dtype=tl.float32)
    view = txl.get_buffer(buf, 0)
    buf1 = txl.smem_alloc([1], dtype=tl.float32)
    view1 = txl.get_buffer(buf1, 0)

    x = x + 1

    layout_b: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[4], order=[0])
    layout_c: tl.constexpr = txl.BlockedLayout(size_per_thread=[4], threads_per_warp=[1], warps_per_cta=[1], order=[0])
    layout_sum_c: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0])
    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])

    txl.frag_smem_store(view, x, layout_b) # TODO: make layout_a not necessary

    c = txl.frag_smem_load(view, layout_c) # should never put into threadlevel if
    if txl.tid(0) == 0:
        sum_c = txl.sum(c)
        #sum_c = tl.full((1,), sum_c, tl.float32)
        txl.frag_smem_store(view1, sum_c, layout_sum_c) # TODO: make layout_a not necessary

    a = txl.frag_smem_load(view1, layout_a)
    if txl.tid(0) == 33:
        txl.print("a:", a)

def test_smem_txl4():
    # Example usage
    BLOCK_SIZE = 128
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel4[(1,)](x, y, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    print("input :", x)
    print("output:", y)

@txl.jit
#@txl.jit(src_file='dump/smem4/txl_smem_kernel4.ptx')
def txl_smem_kernel5(
        a_desc, # 64, 64
        b_desc, # 64, 64
        out_desc, # 64, 64
        mask_ptr, # 64
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(mask_ptr + offsets) # mask

    buf = txl.smem_alloc([BLOCK_SIZE], dtype=tl.int1)
    view = txl.get_buffer(buf, 0)
    txl.smem_store(view, x)

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[BLOCK_SIZE], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    mask = txl.smem_load(view, layout_a)

    a = a_desc.load([0, 0])
    b = b_desc.load([0, 0])
    accumulator = tl.dot(a, b.T)

    masked = tl.where(mask, accumulator, 0.0)
    out_desc.store([0, 0], masked)

    #a = a + tl.full(a.shape, tid, a.dtype) # TODO

def test_smem_txl5():
    # Example usage
    BLOCK_SIZE = 64

    # 1D index tensor
    idx = torch.arange(BLOCK_SIZE, device="cuda")
    # Integer divide by 2 → groups of 2
    group = idx // 2             # [0,0,1,1,2,2,3,3,...]
    # Mod 2 → flip between 0 and 1
    flip = group % 2             # [0,0,1,1,0,0,1,1,...]
    # Convert to bool
    mask = flip == 0             # [True,True,False,False,True,True,False,False,...]

    a = torch.randn((BLOCK_SIZE, BLOCK_SIZE), device="cuda", dtype=torch.float32)
    b = torch.randn((BLOCK_SIZE, BLOCK_SIZE), device="cuda", dtype=torch.float32)

    out = torch.empty((BLOCK_SIZE, BLOCK_SIZE), device="cuda", dtype=torch.float32)

    dummy_block = [64, 64]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    out_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel5[(1,)](a_desc, b_desc, out_desc, mask, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    print("input :", mask)
    print("output:", out)

@txl.jit
#@txl.jit(diff_mode="ttgir")
def txl_smem_kernel6(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + offsets)
    x = x + 1

    buf = txl.smem_alloc([BLOCK_SIZE], dtype=tl.float32)
    view = txl.get_buffer(buf, 0)
    txl.smem_store(view, x)

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[BLOCK_SIZE//2], threads_per_warp=[32], warps_per_cta=[1], order=[0])

    #layout: tl.constexpr = txl.BlockedLayout(size_per_thread=[2], threads_per_warp=[32], warps_per_cta=[1], order=[0])
    #layout: tl.constexpr = txl.DistributedLinearLayout(
    #    reg_bases = [[1]],
    #    lane_bases = [[2], [4]],
    #    warp_bases = [],
    #    block_bases = [],
    #    shape = [8, ]
    #)
    #txl.to_linear_layout((8,), tl.bfloat16, layout)
    #layout: tl.constexpr = txl.DistributedLinearLayout(
    #    reg_bases = [[1], [0], [8], [16]],
    #    lane_bases = [[2], [4], [0], [0]],
    #    warp_bases = [],
    #    block_bases = [],
    #    shape = [32, ]
    #)
    #txl.to_linear_layout((32,), tl.bfloat16, layout)
    layout: tl.constexpr = txl.DistributedLinearLayout(
        reg_bases = [[1], [0], [8], [16], [32]],
        lane_bases = [[2], [4], [0], [0], [0]],
        warp_bases = [[0], [0]],
        block_bases = [],
        shape = [64, ]
    )
    txl.to_linear_layout((64,), tl.bfloat16, layout)

    a = txl.smem_load(view, layout_a)

    tid = txl.tid(0)
    if tid == 0:
        txl.print('a', a)

    #tl.store(out_ptr + offsets, a)

def test_smem_txl6():
    # Example usage
    BLOCK_SIZE = 32
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel6[(1,)](x, y, BLOCK_SIZE=32, num_warps=1)

    print("input :", x)
    print("output:", y)

@triton.jit
def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # program id (only one CTA here)
    pid = tl.program_id(0)

    # thread id within the CTA
    tid = tl.arange(0, BLOCK_SIZE)  # 0..127 if BLOCK=128

    # load input
    x = tl.load(x_ptr + tid)

    # do computation
    x = x + 1

    # find warp_id and lane_id
    warp_id = tid // 32       # 0..3
    lane_id = tid % 32        # 0..31

    # select only lane 0 in each warp
    mask = lane_id == 0

    # compact index for y (only 4 slots, one per warp)
    out_idx = warp_id

    # store only lane 0 values
    tl.store(y_ptr + out_idx, x, mask=mask)

def test_smem_triton():
    # Example usage
    BLOCK_SIZE = 128
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty(4, dtype=torch.float32, device='cuda')

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    kernel[(1,)](x, y, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    print("input :", x)
    print("output:", y)

import os
from triton import knobs

dump_dir=None
#os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions"
#os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"

#knobs.runtime.override_arch='sm100'
knobs.autotuning.print=True
knobs.compilation.always_compile=True

if dump_dir:
    knobs.compilation.dump_ir=True
    knobs.cache.dump_dir=dump_dir

#test_txl()
#test_smem_txl()
#test_smem_txl2()
#test_smem_txl3()
#test_smem_txl4()
#test_smem_txl5()
test_smem_txl6()
#test_smem_triton()
