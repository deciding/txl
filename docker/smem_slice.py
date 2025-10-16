import triton
import triton.language as tl
import txl
import torch
import os
import sys
import math
import pytest
from triton.tools.tensor_descriptor import TensorDescriptor

# torch.set_printoptions(profile="full")

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@txl.jit
def smem_slice_test(q_desc, k_desc, o_desc,
                    size_m:tl.constexpr, 
                    size_n:tl.constexpr, 
                    r0:tl.constexpr, 
                    ):
    bQ = txl.smem_alloc([size_m, r0], dtype=tl.float16)
    mQ = txl.mbar_alloc(1)
    bK = txl.smem_alloc([size_n, r0], dtype=tl.float16)
    mK = txl.mbar_alloc(1)

    bQi = txl.get_buffer(bQ, 0)
    bKi = txl.get_buffer(bK, 0)
    mQi = txl.get_buffer(mQ, 0)
    mKi = txl.get_buffer(mK, 0)
    txl.mbar_expect(mQi, size_m*r0*2)
    txl.tma_load(bQi, q_desc, [0, 0], mQi)
    txl.mbar_wait(mQi, 0)
    txl.mbar_expect(mKi, size_n*r0*2)
    txl.tma_load(bKi, k_desc, [0, 0], mKi)
    txl.mbar_wait(mKi, 0)

    kup = txl.smem_slice(bKi, 0, size_n//2, 0)
    kdown = txl.smem_slice(bKi, size_n//2, size_n//2, 0)

    qkL = tl.dot(bQi, kup.T)
    txl.dot_wait(0)
    # txl.print(qkL)

    o_desc.store([0, 0], qkL.to(tl.float16))

def test(q, k, m, n, r):
    o = torch.empty((m, n//2), dtype=torch.float16, device="cuda")

    desc_q = TensorDescriptor(q, shape=[m, r], strides=[r, 1], block_shape=[m, r])
    desc_k = TensorDescriptor(k, shape=[n, r], strides=[r, 1], block_shape=[n, r])
    desc_o = TensorDescriptor(o, shape=[m, n//2], strides=[n//2, 1], block_shape=[m, n//2])

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    grid = lambda meta: (1,)

    smem_slice_test[grid](desc_q, desc_k, desc_o, m, n, r)

    return o

def test_op(m, n, r0, dtype=torch.float16):
    q = torch.eye(m, r0, dtype=dtype, device=DEVICE)
    kvu = (torch.ones((n//2, r0), dtype=dtype, device=DEVICE)*0.5)
    kvd = (torch.ones((n//2, r0), dtype=dtype, device=DEVICE)*2.0)
    kv = torch.cat([kvu, kvd], dim=0)

    tri_out = test(q, kv, m, n, r0)
    ref_out = torch.matmul(q, kvu.T)

    max_err = (tri_out - ref_out).abs().max().item()
    print(f"smem_slice out: {tri_out}")
    print(f"ref out: {ref_out}")
    print(f"max err: {max_err}")

if __name__ == "__main__":

    test_op(128, 128, 256)