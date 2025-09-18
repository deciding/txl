import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import txl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@txl.jit()
def txl_kernel(
        desc_q,
        desc_o,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
       ):

    if txl.tid(0) == 0:
        txl.print('hello t0')
    #if txl.warp_id() == 0:
    #    txl.print('hello w0', txl.tidx.x())
    #if txl.warpgroup_id() == 0:
    #    txl.print('hello wg0', txl.tidx.x())
    qo_offset_y = 0
    q = desc_q.load([qo_offset_y, 0])
    acc = q + 1.0
    desc_o.store([qo_offset_y, 0], acc)

def test_txl():
    dtype = torch.float16
    dummy_block = [128, 64]
    q = torch.randn((128, 64), dtype=dtype, device=DEVICE)
    o = torch.empty_like(q)

    desc_q = TensorDescriptor(q, shape=[128, 64], strides=[64, 1], block_shape=dummy_block)
    desc_o = TensorDescriptor(o, shape=[128, 64], strides=[64, 1], block_shape=dummy_block)

    grid = lambda meta: (1,)
    txl_kernel[grid](desc_q, desc_o, BLOCK_SIZE_M=128, BLOCK_SIZE_N=64)
    print(q)
    print(o)

test_txl()
