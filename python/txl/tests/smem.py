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
@txl.jit()
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
FAILED: smem alloc, smem store matmul mask
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

"""
to_linear_layout
"""
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

"""
Shared Encoding
"""
@txl.jit
#@txl.jit(diff_mode="ttgir")
def txl_smem_kernel7(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + offsets)
    x = x + 1

    smem_enc: tl.constexpr = txl.SwizzledSharedLayout(1, 1, 1, [0])
    buf = txl.smem_alloc([BLOCK_SIZE], dtype=tl.float32, shared_enc=smem_enc)
    view = txl.get_buffer(buf, 0)
    txl.smem_store(view, x)

def test_smem_txl7():
    # Example usage
    BLOCK_SIZE = 32
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel7[(1,)](x, y, BLOCK_SIZE=32, num_warps=1)

    print("input :", x)
    print("output:", y)

"""
Shared Encoding + Reg Encoding verify, NVMMA repeat load
"""
@txl.jit
#@txl.jit(diff_mode="ttgir")
def txl_smem_kernel8(
        in_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
       ):
    tid = txl.tid(0)
    #smem_enc: tl.constexpr = txl.SwizzledSharedLayout(1, 1, 1, [0])
    buf = txl.smem_alloc([1, 64], dtype=tl.float32)
    view = txl.get_buffer(buf, 0)
    y = tl.arange(0, 64).to(tl.float32)[None, :]
    txl.smem_store(view, y)

    layout: tl.constexpr = txl.NVMMADistributedLayout([3, 0], [4, 1], [16, 64, 16])

    z = txl.frag_smem_load(view, (1, 64, ), layout)

    if tid == 0:
        txl.print('z', z)

    #x = tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :]
    #if tid == 0:
    #    txl.print('x1', x)

    #x = txl.relayout(x, (16, 64), layout)

    #if tid == 0:
    #    txl.print('x2', x)


def test_smem_txl8():
    # Example usage
    BLOCK_SIZE = 32
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel8[(1,)](x, y, BLOCK_SIZE=32, num_warps=4)

    print("input :", x)
    print("output:", y)


"""""""""""""""""""""""""""""
FlashMLA
"""""""""""""""""""""""""""""
    #indices_layout: tl.constexpr = txl.DistributedLinearLayout(
    #    reg_bases = [[16], [32]],
    #    lane_bases = [[0], [0], [0], [1], [2]],
    #    warp_bases = [[4], [8]],
    #    block_bases = [],
    #    shape = [64, ]
    #)

GROUP_SIZE = tl.constexpr(8)
@txl.jit
#@txl.jit(diff_mode="ttgir", log_dir='dump/')
def txl_mla0(
        q_nope_desc, q_pe_desc,
        kv_nope_ptr, kv_pe_ptr,
        o_desc,
        indices_ptr,

        S_Q : tl.constexpr,
        S_KV : tl.constexpr,
        B_H : tl.constexpr,
        D_Q : tl.constexpr,
        D_V : tl.constexpr,
        NUM_HEAD_BLOCKS : tl.constexpr,
        TOPK : tl.constexpr,
        B_TOPK : tl.constexpr,
        STRIDE_KV_NOPE_0: tl.constexpr,
        STRIDE_KV_PE_0: tl.constexpr,
        o0_desc, o1_desc,
        o_q_nope_desc, o_q_pe_desc,
       ):

    tid = txl.tid(0)
    pid = tl.program_id(0)
    s_q_idx = pid // NUM_HEAD_BLOCKS
    q_h_idx = pid % NUM_HEAD_BLOCKS
    NUM_TOPK_BLOCKS = TOPK // B_TOPK


    offs_q = pid * B_H

    q_pe_buf = txl.smem_alloc([64, 512], dtype=tl.bfloat16, num_stages=1); sQpe = txl.get_buffer(q_pe_buf, 0)
    q_nope_buf = txl.smem_alloc([64, 64], dtype=tl.bfloat16, num_stages=1); sQnope = txl.get_buffer(q_nope_buf, 0)
    mbar_q_pe_buf = txl.mbar_alloc(1, num_stages=1); mbar_q_pe = txl.get_buffer(mbar_q_pe_buf, 0)
    mbar_q_nope_buf = txl.mbar_alloc(1, num_stages=1); mbar_q_nope = txl.get_buffer(mbar_q_nope_buf, 0)


    #index_layout1d: tl.constexpr = txl.BlockedLayout([8], [32], [4], [0])
    #index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [32, 1], [4, 1], [1, 0])
    index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])

    Kpe_buf = txl.smem_alloc([B_TOPK, 512], dtype=tl.bfloat16, num_stages=2);
    Knope_buf = txl.smem_alloc([B_TOPK, 64], dtype=tl.bfloat16, num_stages=2);
    sKnope0 = txl.get_buffer(Knope_buf, 0)
    sKpe0 = txl.get_buffer(Kpe_buf, 0)
    sKnope1 = txl.get_buffer(Knope_buf, 1)
    sKpe1 = txl.get_buffer(Kpe_buf, 1)

    is_kv_valid_layout: tl.constexpr = txl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    is_kv_valid_buf = txl.smem_alloc([B_TOPK, 1], dtype=tl.int8, num_stages=2);
    is_kv_valid0 = txl.get_buffer(is_kv_valid_buf, 0)
    is_kv_valid1 = txl.get_buffer(is_kv_valid_buf, 1)

    # 2, 2, 8
    layout_rP_valid: tl.constexpr = txl.DistributedLinearLayout(
            reg_bases=[[1], [0], [8], [16], [32]],
            lane_bases=[[2], [4], [0], [0], [0]],
            warp_bases=[[0], [0]],
            shape=[64,]
    )

    # mma barriers
    mbar_k0_free = txl.mbar_alloc(128, num_stages=2); mbar_k0_free0 = txl.get_buffer(mbar_k0_free, 0); mbar_k0_free1 = txl.get_buffer(mbar_k0_free, 1)
    mbar_k0_ready = txl.mbar_alloc(128, num_stages=2); mbar_k0_ready0 = txl.get_buffer(mbar_k0_ready, 0); mbar_k0_ready1 = txl.get_buffer(mbar_k0_ready, 1)
    mbar_k1_free = txl.mbar_alloc(128, num_stages=2); mbar_k1_free0 = txl.get_buffer(mbar_k1_free, 0); mbar_k1_free1 = txl.get_buffer(mbar_k1_free, 1)
    mbar_k1_ready = txl.mbar_alloc(128, num_stages=2); mbar_k1_ready0 = txl.get_buffer(mbar_k1_ready, 0); mbar_k1_ready1 = txl.get_buffer(mbar_k1_ready, 1)

    mbar_is_kv_valid_ready_buf = txl.mbar_alloc(16, num_stages=1); mbar_is_kv_valid_ready = txl.get_buffer(mbar_is_kv_valid_ready_buf, 0)


    if txl.is_warpgroup([0, 1]):
        if txl.is_warpgroup([0]):
            txl.mbar_expect(mbar_q_pe, 64*512*2)
            txl.tma_load(sQpe, q_pe_desc, [offs_q, 0], mbar_q_pe)
            txl.mbar_expect(mbar_q_nope, 64*64*2)
            txl.tma_load(sQnope, q_nope_desc, [offs_q, 0], mbar_q_nope)

        txl.mbar_wait(mbar_q_pe, 0)
        txl.mbar_wait(mbar_q_nope, 0)

        cur_bar_wait_phase = 0

        #if txl.is_warpgroup([1]):
        #    txl.tma_store(sQnope, o_q_nope_desc, [offs_q, 0])
        #    txl.tma_store(sQpe, o_q_pe_desc, [offs_q, 0])

        sKpe0l = txl.smem_slice(sKpe0, 0, 256, dim=1)
        sV0l = sKpe0l.T
        sKpe1l = txl.smem_slice(sKpe1, 0, 256, dim=1)
        sV1l = sKpe1l.T

        rP = tl.zeros([B_H, B_TOPK], dtype=tl.float32)

        for block_idx in range(0, NUM_TOPK_BLOCKS, 2):
            if block_idx == 0:
                txl.mbar_wait(mbar_k0_ready0, cur_bar_wait_phase)
                # slice 0-3
                for tile_index in tl.static_range(0, 256, 64):
                    cur_sQ = txl.smem_slice(sQpe, tile_index, 64, dim=1)
                    cur_sK = txl.smem_slice(sKpe0, tile_index, 64, dim=1)
                    rP = tl.dot(cur_sQ, cur_sK.T, rP)
                txl.mbar_wait(mbar_k0_ready1, cur_bar_wait_phase)
                # slice 0-3
                for tile_index in tl.static_range(256, 512, 64):
                    cur_sQ = txl.smem_slice(sQpe, tile_index, 64, dim=1)
                    cur_sK = txl.smem_slice(sKpe0, tile_index, 64, dim=1)
                    rP = tl.dot(cur_sQ, cur_sK.T, rP)
                rP = tl.dot(sQnope, sKnope0.T, rP)
                txl.dot_wait(0)

            # mask_rP
            txl.mbar_wait(mbar_is_kv_valid_ready, cur_bar_wait_phase)
            reg_is_kv_valid_0 = txl.smem_load(is_kv_valid_0, layout_rP_valid)
            rP = tl.where(reg_is_kv_valid_0, rP, float('-inf'))


    if txl.is_warpgroup([2]):

        gIndices = indices_ptr + s_q_idx * TOPK
        index_offs = tl.arange(0, B_TOPK * GROUP_SIZE) // GROUP_SIZE

        inc_index_offs = tl.reshape(tl.arange(0, B_TOPK * 64) % 64, (64, 64))
        inc_index_offs = txl.relayout(inc_index_offs, (64, 64), index_layout2d)

        cur_bar_wait_phase = 1

        #for block_idx in range(0, 2, 2):
        for block_idx in range(0, NUM_TOPK_BLOCKS, 2):
            # buf_idx 0
            cur_index_nope_arr = ()
            cur_index_pe_arr = ()
            is_token_valid_arr = ()

            for buf_idx in tl.static_range(0, 2):
            #for buf_idx in tl.static_range(0, 1):
                cur_g_indices = gIndices + (block_idx + buf_idx) * B_TOPK

                cur_index = tl.load(cur_g_indices + index_offs)
                cur_index_pe0 = cur_index * STRIDE_KV_PE_0
                cur_index_nope0 = cur_index * STRIDE_KV_NOPE_0
                cur_index_pe0 = tl.broadcast_to(cur_index_pe0[:, None], (512, 8))
                cur_index_pe0 = tl.reshape(cur_index_pe0, (64, 64))
                cur_index_pe0 = txl.relayout(cur_index_pe0, (64, 64), index_layout2d)
                cur_index_pe0 = cur_index_pe0 + inc_index_offs
                cur_index_nope0 = tl.broadcast_to(cur_index_nope0[:, None], (512, 8))
                cur_index_nope0 = tl.reshape(cur_index_nope0, (64, 64))
                cur_index_nope0 = txl.relayout(cur_index_nope0, (64, 64), index_layout2d)
                cur_index_nope0 = cur_index_nope0 + inc_index_offs


                is_token_valid = (cur_index >=0) & (cur_index < S_KV)
                is_token_valid0 = tl.broadcast_to(is_token_valid[:, None], (512, 8))
                is_token_valid0 = tl.reshape(is_token_valid0, (64, 64))
                is_token_valid0 = txl.relayout(is_token_valid0, (64, 64), index_layout2d)

                cur_index_pe_arr = cur_index_pe_arr + (cur_index_pe0,)
                cur_index_nope_arr = cur_index_nope_arr + (cur_index_nope0,)
                is_token_valid_arr = is_token_valid_arr + (is_token_valid0,)



            # V0l
            txl.mbar_wait(mbar_k0_free0, cur_bar_wait_phase)
            is_token_valid = is_token_valid_arr[0]
            #txl.async_load_wait(0)
            for tile_index in tl.static_range(0, 256, 64):
            #for tile_index in tl.static_range(0, 64, 64):
                cur_sKpe = txl.smem_slice(sKpe0, tile_index, 64, dim=1)
                offs_kv = cur_index_nope_arr[0] + tile_index
                txl.async_load(cur_sKpe, kv_pe_ptr+offs_kv, mask=is_token_valid, contiguity=8)
                #txl.async_load(cur_sKpe, kv_pe_ptr+offs_kv, contiguity=8)
                #txl.async_load_wait(0)
            txl.mbar_arrive(mbar_k0_ready0, track_async_op=True)

            # V1r
            txl.mbar_wait(mbar_k1_free1, cur_bar_wait_phase)
            is_token_valid = is_token_valid_arr[1]
            for tile_index in tl.static_range(256, 512, 64):
            #for tile_index in tl.static_range(0, 64, 64):
                cur_sKpe = txl.smem_slice(sKpe1, tile_index, 64, dim=1)
                offs_kv = cur_index_nope_arr[1] + tile_index
                txl.async_load(cur_sKpe, kv_pe_ptr+offs_kv, mask=is_token_valid, contiguity=8)
                #txl.async_load(cur_sKpe, kv_pe_ptr+offs_kv, contiguity=8)
                #txl.async_load_wait(0)
            offs_kv = cur_index_nope_arr[1]
            txl.async_load(sKnope1, kv_nope_ptr+offs_kv, mask=is_token_valid, contiguity=8)
            txl.mbar_arrive(mbar_k1_ready1, track_async_op=True)

            # V0r
            txl.mbar_wait(mbar_k0_free1, cur_bar_wait_phase)
            is_token_valid = is_token_valid_arr[0]
            for tile_index in tl.static_range(256, 512, 64):
                cur_sKpe = txl.smem_slice(sKpe0, tile_index, 64, dim=1)
                offs_kv = cur_index_nope_arr[0] + tile_index
                txl.async_load(cur_sKpe, kv_pe_ptr+offs_kv, mask=is_token_valid, contiguity=8)
            offs_kv = cur_index_nope_arr[0]
            txl.async_load(sKnope0, kv_nope_ptr+offs_kv, mask=is_token_valid, contiguity=8)
            txl.mbar_arrive(mbar_k0_ready1, track_async_op=True)

            # V1l
            txl.mbar_wait(mbar_k1_free0, cur_bar_wait_phase)
            is_token_valid = is_token_valid_arr[1]
            for tile_index in tl.static_range(0, 256, 64):
                cur_sKpe = txl.smem_slice(sKpe1, tile_index, 64, dim=1)
                offs_kv = cur_index_nope_arr[1] + tile_index
                txl.async_load(cur_sKpe, kv_pe_ptr+offs_kv, mask=is_token_valid, contiguity=8)
            txl.mbar_arrive(mbar_k1_ready0, track_async_op=True)

            if tid % 8 == 0:
                txl.frag_smem_store(is_kv_valid0, is_token_valid_arr[0].to(tl.int8), is_kv_valid_layout)
                txl.frag_smem_store(is_kv_valid1, is_token_valid_arr[1].to(tl.int8), is_kv_valid_layout)
                txl.mbar_arrive(mbar_is_kv_valid_ready)

            cur_bar_wait_phase ^= 1


@txl.jit
#@txl.jit(diff_mode="ttgir", log_dir='dump/')
def txl_mla1(
        q_nope_desc, q_pe_desc,
        kv_nope_ptr, kv_pe_ptr,
        o_desc,
        indices_ptr,

        S_Q : tl.constexpr,
        S_KV : tl.constexpr,
        B_H : tl.constexpr,
        D_Q : tl.constexpr,
        D_V : tl.constexpr,
        NUM_HEAD_BLOCKS : tl.constexpr,
        TOPK : tl.constexpr,
        B_TOPK : tl.constexpr,
        STRIDE_KV_NOPE_0: tl.constexpr,
        STRIDE_KV_PE_0: tl.constexpr,
        o0_desc, o1_desc,
        o_q_nope_desc, o_q_pe_desc,
       ):

    tid = txl.tid(0)
    pid = tl.program_id(0)
    s_q_idx = pid // NUM_HEAD_BLOCKS
    q_h_idx = pid % NUM_HEAD_BLOCKS
    NUM_TOPK_BLOCKS = TOPK // B_TOPK


    offs_q = pid * B_H

    q_nope_buf = txl.smem_alloc([64, 64], dtype=tl.bfloat16, num_stages=1); sQnope = txl.get_buffer(q_nope_buf, 0)
    q_pe_buf = txl.smem_alloc([64, 512], dtype=tl.bfloat16, num_stages=1); sQpe = txl.get_buffer(q_pe_buf, 0)
    mbar_q_nope_buf = txl.mbar_alloc(1, num_stages=1); mbar_q_nope = txl.get_buffer(mbar_q_nope_buf, 0)
    mbar_q_pe_buf = txl.mbar_alloc(1, num_stages=1); mbar_q_pe = txl.get_buffer(mbar_q_pe_buf, 0)


    #index_layout1d: tl.constexpr = txl.BlockedLayout([8], [32], [4], [0])
    #index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [32, 1], [4, 1], [1, 0])
    index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])

    Knope_buf = txl.smem_alloc([B_TOPK, 64], dtype=tl.bfloat16, num_stages=2);
    Kpe_buf = txl.smem_alloc([B_TOPK, 512], dtype=tl.bfloat16, num_stages=2);
    is_kv_valid_layout: tl.constexpr = txl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    is_kv_valid_buf = txl.smem_alloc([B_TOPK, 1], dtype=tl.int8, num_stages=2);

    # mma barriers
    mbar_k0_free = txl.mbar_alloc(128, num_stages=2);
    mbar_k0_ready = txl.mbar_alloc(128, num_stages=2);
    mbar_k1_free = txl.mbar_alloc(128, num_stages=2);
    mbar_k1_ready = txl.mbar_alloc(128, num_stages=2);


    if txl.is_warpgroup([0, 1]):
        if txl.is_warpgroup([0]):
            txl.mbar_expect(mbar_q_nope, 64*64*2)
            txl.tma_load(sQnope, q_nope_desc, [offs_q, 0], mbar_q_nope)
            txl.mbar_expect(mbar_q_pe, 64*512*2)
            txl.tma_load(sQpe, q_pe_desc, [offs_q, 0], mbar_q_pe)

        txl.mbar_wait(mbar_q_nope, 0)
        txl.mbar_wait(mbar_q_pe, 0)

        cur_bar_wait_phase = 0

        if txl.is_warpgroup([1]):
            txl.tma_store(sQnope, o_q_nope_desc, [offs_q, 0])
            txl.tma_store(sQpe, o_q_pe_desc, [offs_q, 0])

        for block_idx in range(0, NUM_TOPK_BLOCKS, 2):
            pass


    if txl.is_warpgroup([2]):

        gIndices = indices_ptr + s_q_idx * TOPK
        index_offs = tl.arange(0, B_TOPK * GROUP_SIZE) // GROUP_SIZE

        inc_index_offs = tl.reshape(tl.arange(0, B_TOPK * 64) % 64, (64, 64))
        inc_index_offs = txl.relayout(inc_index_offs, (64, 64), index_layout2d)

        cur_bar_wait_phase = 1

        #for block_idx in range(0, 2, 2):
        for block_idx in range(0, NUM_TOPK_BLOCKS, 2):
            # buf_idx 0
            cur_index_pe_arr = ()
            for buf_idx in tl.static_range(0, 2):
            #for buf_idx in tl.static_range(0, 1):
                cur_g_indices = gIndices + (block_idx + buf_idx) * B_TOPK
                cur_index = tl.load(cur_g_indices + index_offs)
                cur_index_nope0 = cur_index * STRIDE_KV_NOPE_0
                cur_index_pe0 = cur_index * STRIDE_KV_PE_0
                cur_index_nope0 = tl.broadcast_to(cur_index_nope0[:, None], (512, 8))
                cur_index_nope0 = tl.reshape(cur_index_nope0, (64, 64))
                cur_index_nope0 = txl.relayout(cur_index_nope0, (64, 64), index_layout2d)
                cur_index_nope0 = cur_index_nope0 + inc_index_offs
                cur_index_pe0 = tl.broadcast_to(cur_index_pe0[:, None], (512, 8))
                cur_index_pe0 = tl.reshape(cur_index_pe0, (64, 64))
                cur_index_pe0 = txl.relayout(cur_index_pe0, (64, 64), index_layout2d)
                cur_index_pe0 = cur_index_pe0 + inc_index_offs

                cur_index_pe_arr = cur_index_pe_arr + (cur_index_pe0,)

                sKnope = txl.get_buffer(Knope_buf, buf_idx)
                sKpe = txl.get_buffer(Kpe_buf, buf_idx)


                is_token_valid = (cur_index >=0) & (cur_index < S_KV)
                is_token_valid0 = tl.broadcast_to(is_token_valid[:, None], (512, 8))
                is_token_valid0 = tl.reshape(is_token_valid0, (64, 64))
                is_token_valid0 = txl.relayout(is_token_valid0, (64, 64), index_layout2d)

                is_kv_valid0 = txl.get_buffer(is_kv_valid_buf, buf_idx)
                if tid % 8 == 0:
                    txl.frag_smem_store(is_kv_valid0, is_token_valid.to(tl.int8), is_kv_valid_layout)

                #for tile_index in range(0, 1):
                offs_kv = cur_index_nope0
                txl.async_load(sKnope, kv_nope_ptr+offs_kv, mask=is_token_valid0, contiguity=8)
                #txl.async_load(sKnope, kv_nope_ptr+offs_kv, contiguity=8)
                txl.async_load_wait(0)
                for tile_index in tl.static_range(0, 512, 64):
                #for tile_index in tl.static_range(0, 64, 64):
                    cur_sKpe = txl.smem_slice(sKpe, tile_index, 64, dim=1)
                    offs_kv = cur_index_pe_arr[buf_idx] + tile_index
                    txl.async_load(cur_sKpe, kv_pe_ptr+offs_kv, mask=is_token_valid0, contiguity=8)
                    #txl.async_load(cur_sKpe, kv_pe_ptr+offs_kv, contiguity=8)
                    txl.async_load_wait(0)

                ## for test
                #offs_store = s_q_idx * TOPK + (block_idx + buf_idx) * B_TOPK
                #txl.tma_store(sKnope, o0_desc, [offs_store, 0])
                #txl.tma_store(sKpe, o1_desc, [offs_store, 0])

@txl.jit
#@txl.jit(diff_mode="llir")
def txl_mla_load2(
        q_nope_desc, q_pe_desc,
        kv_nope_ptr, kv_pe_ptr,
        o_desc,
        indices_ptr,

        S_Q : tl.constexpr,
        S_KV : tl.constexpr,
        B_H : tl.constexpr,
        D_Q : tl.constexpr,
        D_V : tl.constexpr,
        NUM_HEAD_BLOCKS : tl.constexpr,
        TOPK : tl.constexpr,
        B_TOPK : tl.constexpr,
        STRIDE_KV_NOPE_0: tl.constexpr,
        STRIDE_KV_PE_0: tl.constexpr,
        o0_desc,
        o1_desc,
       ):

    tid = txl.tid(0)
    pid = tl.program_id(0)
    s_q_idx = pid // NUM_HEAD_BLOCKS
    q_h_idx = pid % NUM_HEAD_BLOCKS
    NUM_TOPK_BLOCKS = TOPK // B_TOPK

    Knope_buf = txl.smem_alloc([B_TOPK, 64], dtype=tl.bfloat16, num_stages=2);
    Kpe_buf = txl.smem_alloc([B_TOPK, 512], dtype=tl.bfloat16, num_stages=2);

    gIndices = indices_ptr + s_q_idx * TOPK
    index_offs = tl.arange(0, B_TOPK * GROUP_SIZE) // GROUP_SIZE

    index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])

    inc_index_offs = tl.reshape(tl.arange(0, B_TOPK * 64) % 64, (64, 64))
    inc_index_offs = txl.relayout(inc_index_offs, (64, 64), index_layout2d)

    k_nope_buf = txl.smem_alloc([64, 64], dtype=tl.bfloat16, num_stages=1); sKnope = txl.get_buffer(k_nope_buf, 0)
    k_pe_buf = txl.smem_alloc([64, 512], dtype=tl.bfloat16, num_stages=1); sKpe = txl.get_buffer(k_pe_buf, 0)

    #for block_idx in range(0, 2, 2):
    for block_idx in range(0, NUM_TOPK_BLOCKS, 2):
        # buf_idx 0
        #for buf_idx in tl.static_range(0, 2):
        for buf_idx in tl.static_range(0, 1):
            cur_g_indices = gIndices + (block_idx + buf_idx) * B_TOPK
            cur_index = tl.load(cur_g_indices + index_offs)
            cur_index_nope00 = cur_index * STRIDE_KV_NOPE_0
            cur_index_pe0 = cur_index * STRIDE_KV_PE_0
            cur_index_nope01 = tl.broadcast_to(cur_index_nope00[:, None], (512, 8))
            cur_index_nope02 = tl.reshape(cur_index_nope01, (64, 64))
            cur_index_nope03 = txl.relayout(cur_index_nope02, (64, 64), index_layout2d)
            cur_index_nope0 = cur_index_nope03 + inc_index_offs
            cur_index_pe0 = tl.broadcast_to(cur_index_pe0[:, None], (512, 8))
            cur_index_pe0 = tl.reshape(cur_index_pe0, (64, 64))
            cur_index_pe0 = txl.relayout(cur_index_pe0, (64, 64), index_layout2d)
            cur_index_pe0 = cur_index_pe0 + inc_index_offs

            sKnope = txl.get_buffer(Knope_buf, buf_idx)
            sKpe = txl.get_buffer(Kpe_buf, buf_idx)

            is_token_valid0 = cur_index >=0 and cur_index < S_KV
            is_token_valid0 = tl.broadcast_to(is_token_valid0[:, None], (512, 8))
            is_token_valid0 = tl.reshape(is_token_valid0, (64, 64))
            is_token_valid0 = txl.relayout(is_token_valid0, (64, 64), index_layout2d)

            #for tile_index in range(0, 1):
            offs_kv = cur_index_nope0

            #if tid == 1 and pid == 0:
            #    txl.print('cur_index:', cur_index)
            #    txl.print('cur_index_nope00:', cur_index_nope00)
            #    txl.print('cur_index_nope03:', cur_index_nope03)
            #    txl.print('cur_index_nope0:', cur_index_nope0)
            #    #txl.print('offs_kv:', offs_kv)
            #    #txl.print('inc_index_offs:', inc_index_offs)

            txl.async_load(sKnope, kv_nope_ptr+offs_kv, contiguity=8)
            #txl.async_load(sKnope, kv_nope_ptr+offs_kv, mask=is_token_valid0)
            #txl.async_load_wait(0)


@txl.jit
#@txl.jit(diff_mode="ttgir")
#@txl.jit(diff_mode="ttgir", log_dir='dump')
def txl_mla_load1(
        q_nope_desc, q_pe_desc,
        kv_nope_ptr, kv_pe_ptr,
        o_desc,
        indices_ptr,

        S_Q : tl.constexpr,
        S_KV : tl.constexpr,
        B_H : tl.constexpr,
        D_Q : tl.constexpr,
        D_V : tl.constexpr,
        NUM_HEAD_BLOCKS : tl.constexpr,
        TOPK : tl.constexpr,
        B_TOPK : tl.constexpr,
        STRIDE_KV_NOPE_0: tl.constexpr,
        STRIDE_KV_PE_0: tl.constexpr,
        o0_desc,
        o1_desc,
        is_kv_valid_desc
       ):

    tid = txl.tid(0)
    pid = tl.program_id(0)
    s_q_idx = pid // NUM_HEAD_BLOCKS
    q_h_idx = pid % NUM_HEAD_BLOCKS
    NUM_TOPK_BLOCKS = TOPK // B_TOPK


    gIndices = indices_ptr + s_q_idx * TOPK
    index_offs = tl.arange(0, B_TOPK * GROUP_SIZE) // GROUP_SIZE

    #index_layout1d: tl.constexpr = txl.BlockedLayout([8], [32], [4], [0])
    #index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [32, 1], [4, 1], [1, 0])
    index_layout2d: tl.constexpr = txl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    is_kv_valid_layout: tl.constexpr = txl.BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])

    #is_kv_valid_buf = txl.smem_alloc([B_TOPK, 8], dtype=tl.int8, num_stages=2);
    is_kv_valid_buf = txl.smem_alloc([B_TOPK, 1], dtype=tl.int8, num_stages=2);
    #is_kv_valid_buf = txl.smem_alloc([B_TOPK, 8], dtype=tl.int32, num_stages=2);

    is_token_valid = tl.arange(0, 64)[:, None] * 8 + tl.arange(0, 8)[None, :]

    is_token_valid = (is_token_valid < 32).to(tl.int8)
    is_token_valid = txl.relayout(is_token_valid, (64, 8), is_kv_valid_layout)

    is_kv_valid0 = txl.get_buffer(is_kv_valid_buf, 0)
    if tid % 8 == 0:
        if tid == 0 and pid == 0:
            txl.print('is_token_valid', is_token_valid)
        txl.frag_smem_store(is_kv_valid0, is_token_valid, is_kv_valid_layout)
        tmp = txl.frag_smem_load(is_kv_valid0, is_kv_valid_layout)
        if tid == 0 and pid == 0:
            txl.print('tmp', tmp)
    #for block_idx in range(0, 2, 2):
    ##for block_idx in range(0, NUM_TOPK_BLOCKS, 2):
    #    for buf_idx in tl.static_range(0, 1):
    #    #for buf_idx in tl.static_range(0, 2):
    #        cur_g_indices = gIndices + (block_idx + buf_idx) * B_TOPK
    #        cur_index = tl.load(cur_g_indices + index_offs)

    #        is_token_valid = (cur_index >=0) & (cur_index < S_KV)
    #        is_token_valid0 = tl.broadcast_to(is_token_valid[:, None], (512, 8))
    #        is_token_valid0 = tl.reshape(is_token_valid0, (64, 64))
    #        is_token_valid0 = txl.relayout(is_token_valid0, (64, 64), index_layout2d)

    #        is_kv_valid0 = txl.get_buffer(is_kv_valid_buf, buf_idx)
    #        is_token_valid1 = tl.reshape(is_token_valid, (64, 8))
    #        is_token_valid1 = txl.relayout(is_token_valid1, (64, 8), is_kv_valid_layout)
    #        txl.smem_store(is_kv_valid0, is_token_valid1)


def test_txl_mla1():
    # Example usage

    B_H = 64
    B_TOPK = 64
    D_Q = 576
    D_K = D_Q
    D_V = 512

    b0 = 1
    s_q = 64
    s_kv = 128
    top_k = 128
    h_q = 128

    h_kv = 1
    d_qk = D_Q
    d_v = D_V

    q = torch.randn((b0, s_q, h_q, d_qk), dtype=torch.bfloat16, device='cuda')/10

    kv = torch.randn((b0, s_kv, h_kv, d_qk), dtype=torch.bfloat16, device='cuda')/10

    q.clamp_(-10, 10)
    q_nope, q_pe = torch.split(q, [64, 512], dim=-1)
    q_nope = q_nope.contiguous()
    q_pe = q_pe.contiguous()

    kv.clamp_(-10, 10)
    kv_nope, kv_pe = torch.split(kv, [64, 512], dim=-1)
    kv_nope = kv_nope.contiguous()
    kv_pe = kv_pe.contiguous()

    indices = torch.full((b0, s_q, h_kv, top_k), s_kv, dtype=torch.int32, device='cuda')
    for b in range(b0):
        for s in range(s_q):
            for h in range(h_kv):
                # NOTE We use the following method to generate indices so that most indices lies within [s_kv-20000, s_kv), which is more realistic for sparse attention
                near_mask = torch.randint(0, 32, (min(top_k, s_kv),)) < 31
                cur_indices = torch.randperm(s_kv)[:top_k]
                cur_indices[near_mask] = torch.randint(max(0, s_kv-20000), s_kv-1, (near_mask.sum().item(),))
                if len(cur_indices) < top_k:
                    cur_indices = torch.cat([cur_indices, torch.full((top_k - len(cur_indices),), 2147480000)])
                cur_indices = cur_indices[torch.randperm(top_k)]
                indices[b, s, h] = cur_indices
    indices = indices.to(q.device)

    q = q.squeeze(0); kv = kv.squeeze(0); indices = indices.squeeze(0)
    q_nope = q_nope.squeeze(0); q_pe = q_pe.squeeze(0)
    kv_nope = kv_nope.squeeze(0); kv_pe = kv_pe.squeeze(0)

    out = torch.empty((s_q, h_q, d_v), dtype=q.dtype, device=q.device)

    q_nope_desc = TensorDescriptor(q_nope, (s_q*h_q, 64), (64, 1), [B_H, 64])
    q_pe_desc = TensorDescriptor(q_pe, (s_q*h_q, 512), (512, 1), [B_H, 512])
    #q_nope_desc = TensorDescriptor(q, (s_q*h_q, d_qk), (d_qk, 1), [B_H, D_V])
    o_desc = TensorDescriptor(out, (s_q*h_q, d_v), (d_v, 1), [B_H, D_V])

    out_q_nope = torch.empty((s_q, h_q, 64), dtype=q.dtype, device=q.device)
    o_q_nope_desc = TensorDescriptor(out_q_nope, (s_q*h_q, 64), (64, 1), [B_H, 64])
    out_q_pe = torch.empty((s_q, h_q, 512), dtype=q.dtype, device=q.device)
    o_q_pe_desc = TensorDescriptor(out_q_pe, (s_q*h_q, 512), (512, 1), [B_H, 512])

    out0 = torch.empty((s_q, top_k, 64), dtype=q.dtype, device=q.device)
    o0_desc = TensorDescriptor(out0, (s_q*top_k, 64), (64, 1), [B_TOPK, 64])
    out1 = torch.empty((s_q, top_k, 512), dtype=q.dtype, device=q.device)
    o1_desc = TensorDescriptor(out1, (s_q*top_k, 512), (512, 1), [B_TOPK, 512])
    is_kv_valid = torch.empty((s_q, top_k), dtype=torch.int8, device=q.device)
    is_kv_valid_desc = TensorDescriptor(is_kv_valid, (s_q*top_k,), (1,), [B_TOPK])

    #s_q = q.size(0)
    #s_kv = kv.size(0)
    #h_q = q.size(1)
    #h_kv = kv.size(1)
    #d_qk = q.size(2) # kv.size(2) is the same
    #top_k = indices.size(2)

    #assert h_kv == 1
    #assert top_k % (2*B_TOPK) == 0
    #assert top_k > 0
    #assert h_q % B_H == 0

    NUM_HEAD_BLOCKS = h_q // B_H
    txl_mla1[(NUM_HEAD_BLOCKS * s_q,)](
    #txl_mla2[(NUM_HEAD_BLOCKS * s_q,)](
    #txl_mla1[(NUM_HEAD_BLOCKS * s_q,)](
            q_nope_desc, q_pe_desc,
            kv_nope, kv_pe,
            o_desc,
            indices,
            s_q, s_kv,
            B_H, D_Q, D_V,
            NUM_HEAD_BLOCKS,
            top_k,
            B_TOPK,
            kv_nope.stride(0),
            kv_pe.stride(0),
            o0_desc,
            o1_desc,
            o_q_nope_desc,
            o_q_pe_desc,
            num_warps=4, num_warpgroups=3)

    # REF
    kv_nope_ref = kv_nope.squeeze(1)  # (s_kv, 64)
    kv_pe_ref = kv_pe.squeeze(1)      # (s_kv, 512)
    indices_ref = indices.squeeze(1)  # (s_q, top_k)
    #out0_ref = torch.empty((s_q, top_k, 64), dtype=kv_nope.dtype, device=kv_nope_ref.device)
    #out1_ref = torch.empty((s_q, top_k, 512), dtype=kv_pe.dtype, device=kv_pe_ref.device)
    kv_nope_sel = []
    kv_pe_sel = []

    idx = indices_ref.unsqueeze(-1).expand(-1, -1, 64).to(dtype=torch.int64)     # (s_q, top_k, 64)
    kv_nope_sel = torch.gather(kv_nope_ref.unsqueeze(0).expand(s_q, -1, -1), 1, idx)  # (s_q, top_k, 64)

    idx_pe = indices_ref.unsqueeze(-1).expand(-1, -1, 512).to(dtype=torch.int64) # (s_q, top_k, 512)
    kv_pe_sel = torch.gather(kv_pe_ref.unsqueeze(0).expand(s_q, -1, -1), 1, idx_pe)   # (s_q, top_k, 512)

    #print(q0-out)
    print((out0-kv_nope_sel).sum())
    print((out1-kv_pe_sel).sum())
    print((out_q_nope-q_nope).sum())
    print((out_q_pe-q_pe).sum())
    import pdb;pdb.set_trace()



"""""""""""""""""""""""""""""
TXL
"""""""""""""""""""""""""""""


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
    dump_dir='dump/smem/'
    #dump_dir = None

    from triton import knobs
    import os
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "tritongpu-remove-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-smem-alloc-legalize"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-smem-alloc-layout-conversions"
    #os.environ["TRITON_LLVM_DEBUG_ONLY"] = "txlgpu-pipeliner"
    knobs.runtime.override_arch='sm90'
    knobs.autotuning.print=True
    knobs.compilation.always_compile=True

    if dump_dir:
        knobs.compilation.dump_ir=True
        knobs.cache.dump_dir=dump_dir

    #test_txl_mla1()
    #test_smem_all()
    test_smem_txl8()


if __name__ == "__main__":
    test()
