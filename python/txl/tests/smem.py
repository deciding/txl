import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import txl

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
#@txl.jit(diff_mode='llir')
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

    layout_sum_c: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[1], order=[0])

    txl.frag_smem_store(view, x, layout_b) # TODO: make layout_a not necessary

    if txl.lane_id() == 0:
        b = txl.frag_smem_load(view, layout_b)
        txl.print("b:", b)

    buf1 = txl.smem_alloc([1], dtype=tl.float32)
    view1 = txl.get_buffer(buf1, 0)

    if txl.tid(0) == 0:
        c = txl.frag_smem_load(view, layout_c)
        sum_c = txl.sum(c)
        txl.print("c:", c)
        txl.print("sum_c", sum_c)

        sum_c = tl.full((1,), sum_c, tl.float32)
        txl.frag_smem_store(view1, sum_c, layout_sum_c) # TODO: make layout_a not necessary

    a = txl.frag_smem_load(view1, layout_a)
    if txl.tid(0) == 33:
        txl.print("a:", a)




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

    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + offsets)

    buf = txl.smem_alloc([4], dtype=tl.float32)
    view = txl.get_buffer(buf, 0)
    x = x + 1

    lane_id = txl.lane_id()
    mask = lane_id == 0

    layout_b: tl.constexpr = txl.BlockedLayout(size_per_thread=[1], threads_per_warp=[1], warps_per_cta=[4], order=[0])

    txl.smem_store(view, x, layout_b) # TODO: make layout_a not necessary
    tl.inline_asm_elementwise(
        asm="""
        {@mask bra END; }            // skip if not lane0
        st.shared.f32 [{smem}], {val};
        END:
        """,
        constraints="r,r,r",  # smem address, value, mask
        args=[smem_ptr, val, mask],
    )

    #if txl.lane_id() == 0:
    #    a = txl.smem_load(view, layout_b)
    #    txl.print("hello", a)

def test_smem_txl2():
    # Example usage
    BLOCK_SIZE = 128
    x = torch.arange(BLOCK_SIZE, dtype=torch.float32, device='cuda')
    y = torch.empty_like(x)

    # Launch kernel: 1 block, 1 warp (BLOCK=32)
    txl_smem_kernel2[(1,)](x, y, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

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
test_smem_txl2()
#test_smem_triton()
