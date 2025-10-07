import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import txl
import os; print(os.getpid())


DEVICE = triton.runtime.driver.active.get_active_torch_device()

"""
All kinds of load and store combinations
Note that smem_load is implicitly called
"""
#@txl.jit()
#@txl.jit(diff_mode='llir')
#@txl.jit(diff_mode='ttgir', diff_select=40)
def txl_smem_all_kernel(
        desc_q,
        desc_o,
        q_ptr,
        o_ptr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    qo_offset_y = 0

    # load 0
    #offs = tl.arange(0, 128)[:, None] * 64 + tl.arange(0, 64)[None, :]
    #q_ptrs = q_ptr + offs
    #q = tl.load(q_ptrs)

    # load 1
    #q = desc_q.load([qo_offset_y, 0])

    # load 2
    #q_buf = txl.smem_alloc([128, 64], dtype=tl.float16, num_stages=1); q = txl.get_buffer(q_buf, 0)
    #mbar_q_buf = txl.mbar_alloc(arr_count=1, num_stages=1); mbar_q = txl.get_buffer(mbar_q_buf, 0)
    #txl.mbar_expect(mbar_q, 128 * 64 * 2)
    #txl.tma_load(q, desc_q, [qo_offset_y, 0], mbar_q)
    #txl.mbar_wait(mbar_q, 0)

    # load 3
    q_buf = txl.smem_alloc([128, 64], dtype=tl.float16, num_stages=1); q = txl.get_buffer(q_buf, 0)
    offs = tl.arange(0, 128)[:, None] * 64 + tl.arange(0, 64)[None, :]
    q_ptrs = q_ptr + offs
    txl.async_load(q, q_ptrs)
    txl.async_load_wait(0)

    acc = q + 1.0

    # store 0
    tl.store(o_ptr+offs, acc)

    # store 1
    #desc_o.store([qo_offset_y, 0], acc)

    # store 2
    #o_buf = txl.smem_alloc([128, 64], dtype=tl.float16, num_stages=1); o = txl.get_buffer(o_buf, 0)
    #txl.smem_store(o, acc)
    #txl.tma_store(o, desc_o, [qo_offset_y, 0])
    #txl.tma_store_wait(0)

def test_smem_all():
    dtype = torch.float16
    dummy_block = [128, 64]
    q = torch.randn((128, 64), dtype=dtype, device=DEVICE)
    o = torch.empty_like(q)

    desc_q = TensorDescriptor(q, shape=[128, 64], strides=[64, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[128, 64], strides=[64, 1], block_shape=dummy_block)

    grid = lambda meta: (1,)
    txl_smem_all_kernel[grid](desc_q, desc_o, q, o, BLOCK_SIZE_M=128, BLOCK_SIZE_N=64)
    print(q)
    print(o)


"""
basic triton descriptor load store
"""
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

"""
BlockedLayout load
"""
@txl.jit
#@txl.jit(diff_mode="ttgir")
def txl_smem_kernel(
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

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[1], order=[0])

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

"""
Masked one per warp BlockedLayout load/store
Masked one thread 4 BlockedLayout load/store
const layout conversions
"""
@txl.jit
#@txl.jit(diff_mode='ttgir')
#@txl.jit(diff_mode='llir', log_dir='dump')
def txl_smem_kernel2(
        in_ptr,
        out_ptr,
        out_desc,
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + offsets)
    x = x + 1

    buf = txl.smem_alloc([4], dtype=tl.float32)
    view = txl.get_buffer(buf, 0)

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])

    layout_one_per_warp: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[4], order=[0])

    layout_one_thread_4: tl.constexpr = txl.BlockedLayout(size_per_thread=[4], threads_per_warp=[1], warps_per_cta=[1], order=[0])

    txl.frag_smem_store(view, x, layout_one_per_warp) # TODO: make layout_a not necessary

    if txl.lane_id() == 0:
        b = txl.frag_smem_load(view, layout_one_per_warp)
        txl.print("b:", b)

    if txl.tid(0) == 0:
        c = txl.frag_smem_load(view, layout_one_thread_4)
        c1 = c + 1.0
        sum_c = txl.sum(c)
        txl.print("c:", c)
        txl.print("c1:", c1)
        txl.print("sum_c", sum_c)

        txl.frag_smem_store(view, c, layout_one_thread_4)
        txl.tma_store(view, out_desc, [0])

def test_smem_txl2():
    # Example usage
    BLOCK_SIZE = 128
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)
    desc_y = TensorDescriptor(y, shape=[4,], strides=[1], block_shape=[4])

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel2[(1,)](x, y, desc_y, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    print("input :", x)
    print("output:", y)

"""
Single element store, and broadcast to all threads
"""
@txl.jit
def txl_smem_kernel3(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):

    layout_sum_c: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0])

    buf1 = txl.smem_alloc([1], dtype=tl.float32)
    view1 = txl.get_buffer(buf1, 0)

    if txl.tid(0) == 0:
        sum_c = tl.full((1,), 3.0, tl.float32)
        txl.frag_smem_store(view1, sum_c, layout_sum_c) # TODO: make layout_a not necessary

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])
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

"""
one lane save, one thread load, one value save, broadcast load
"""
@txl.jit
#@txl.jit(src_file='dump/smem4/txl_smem_kernel4.ptx')
def txl_smem_kernel4(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + offsets)
    x = x + 1

    buf = txl.smem_alloc([4], dtype=tl.float32)
    view = txl.get_buffer(buf, 0)
    buf1 = txl.smem_alloc([1], dtype=tl.float32)
    view1 = txl.get_buffer(buf1, 0)

    layout_b: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[4], order=[0])
    layout_c: tl.constexpr = txl.BlockedLayout(size_per_thread=[4], threads_per_warp=[1], warps_per_cta=[1], order=[0])
    layout_sum_c: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0])
    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0])

    # one lane save
    txl.frag_smem_store(view, x, layout_b) # TODO: make layout_a not necessary

    # one thread load
    c = txl.frag_smem_load(view, layout_c) # should never put into threadlevel if
    if txl.tid(0) == 0:
        sum_c = txl.sum(c)
        #sum_c = tl.full((1,), sum_c, tl.float32)
        # one value save
        txl.frag_smem_store(view1, sum_c, layout_sum_c) # TODO: make layout_a not necessary

    # all thread broadcast
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

"""
smem alloc, smem store matmul mask
"""
#@txl.jit
@txl.jit(diff_mode='ttgir')
def txl_smem_kernel5(
        a_desc, # 128, 128
        b_desc, # 128, 128
        out_desc, # 128, 128
        mask_ptr, # 128
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(mask_ptr + offsets) # mask

    buf = txl.smem_alloc([BLOCK_SIZE], dtype=tl.int1)
    view = txl.get_buffer(buf, 0)
    txl.smem_store(view, x)

    layout_a: tl.constexpr = txl.BlockedLayout(size_per_thread=[BLOCK_SIZE], threads_per_warp=[32], warps_per_cta=[4], order=[0])
    mask = txl.frag_smem_load(view, layout_a)

    a = a_desc.load([0, 0])
    b = b_desc.load([0, 0])
    accumulator = tl.dot(a, b.T)

    masked = tl.where(mask, accumulator, 0.0)
    out_desc.store([0, 0], masked)

    #a = a + tl.full(a.shape, tid, a.dtype) # TODO

def test_smem_txl5():
    # Example usage
    BLOCK_SIZE = 128

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

    dummy_block = [128, 128]
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

#test_txl()
#test_smem_txl()
#test_smem_txl2()
#test_smem_txl3()
#test_smem_txl4()
#test_smem_txl5()
#test_smem_txl6()
#test_smem_triton()

def test():
    #dump_dir='dump/smem/'
    dump_dir = None

    from triton import knobs
    import os
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-smem-alloc-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"
    knobs.runtime.override_arch='sm90'
    knobs.autotuning.print=True
    knobs.compilation.always_compile=True

    if dump_dir:
        knobs.compilation.dump_ir=True
        knobs.cache.dump_dir=dump_dir

    test_smem_txl6()


if __name__ == "__main__":
    test()
